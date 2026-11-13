"""Map ingestion off the LIO thread.

The acquisition/LIO loop hands each odometry result to one worker thread that
runs OnlineMap.ingest_odom (2D grid and 3D voxel projection). Ingestion order is
preserved. Live sessions use ``drop_stale=True``: while the worker is busy only
the newest pending result is kept, so a slow map update never delays LIO.
Offline sessions use ``drop_stale=False``: the producer waits for the single
pending slot, every result is ingested, and the map equals synchronous
ingestion while LIO still overlaps with the previous map update.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from .timing import FrameTimings


class MapIngestWorker:
    def __init__(
        self,
        online_map: Any,
        *,
        drop_stale: bool,
        on_keyframe: Callable[[], None] | None = None,
    ) -> None:
        self._map = online_map
        self._drop_stale = bool(drop_stale)
        self._on_keyframe = on_keyframe or (lambda: None)
        self._condition = threading.Condition()
        self._pending: tuple[Any, int | None] | None = None
        self._busy = False
        self._closing = False
        self._thread: threading.Thread | None = None
        self.error: BaseException | None = None
        self.submitted = 0
        self.dropped = 0
        self.ingested = 0
        self.timings = FrameTimings()
        self.last_ms: dict[str, float] = {"map_ms": 0.0, "grid_ms": 0.0, "voxel_ms": 0.0}

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("map ingest worker already started")
        self._thread = threading.Thread(target=self._run, name="mapping-ingest", daemon=True)
        self._thread.start()

    def submit(self, odom: Any, generation: int | None) -> None:
        """Queue one odometry result; raises the worker's error if it failed."""
        with self._condition:
            if self.error is not None:
                raise RuntimeError(f"地图更新失败：{self.error}") from self.error
            if self._closing:
                raise RuntimeError("map ingest worker is closed")
            if self._drop_stale:
                if self._pending is not None:
                    self.dropped += 1
            else:
                while self._pending is not None and self.error is None:
                    self._condition.wait()
                if self.error is not None:
                    raise RuntimeError(f"地图更新失败：{self.error}") from self.error
            self._pending = (odom, generation)
            self.submitted += 1
            self._condition.notify_all()

    def check(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"地图更新失败：{self.error}") from self.error

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Wait until nothing is pending or running (tests and orderly shutdown)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while (self._pending is not None or self._busy) and self.error is None:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def stop(self, timeout: float = 15.0) -> None:
        """Ingest what is pending, then stop. Raises the worker's error, if any."""
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise TimeoutError("map ingest worker did not finish")
        self.check()

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._closing:
                    self._condition.wait()
                if self._pending is None:
                    return
                odom, generation = self._pending
                self._pending = None
                self._busy = True
                self._condition.notify_all()
            try:
                started = time.monotonic()
                keyframe = self._map.ingest_odom(odom, generation=generation)
                sample = {"map_ms": (time.monotonic() - started) * 1000.0}
                sample.update(self._map.projection_timings)
                self.last_ms = sample
                self.timings.add(sample)
                self.ingested += 1
                if keyframe:
                    self._on_keyframe()
            except BaseException as exc:  # surfaced to the producer on its next call
                with self._condition:
                    self.error = exc
                    self._busy = False
                    self._condition.notify_all()
                return
            with self._condition:
                self._busy = False
                self._condition.notify_all()
