"""Map ingestion worker: ordering, stale-drop policy, error propagation, shutdown."""

import threading
import time

import pytest

from robot_mapping.runtime.map_ingest import MapIngestWorker


class SlowMap:
    def __init__(self, delay=0.0, fail_on=None):
        self.delay = delay
        self.fail_on = fail_on
        self.seen = []
        self.release = threading.Event()
        self.release.set()

    def ingest_odom(self, odom, *, generation=None):
        self.release.wait(5)
        time.sleep(self.delay)
        if odom == self.fail_on:
            raise ValueError("boom")
        self.seen.append((odom, generation))
        return odom % 3 == 0

    @property
    def projection_timings(self):
        return {"grid_ms": 1.0, "voxel_ms": 2.0}


def test_blocking_mode_ingests_every_result_in_order():
    fake = SlowMap(delay=0.002)
    keyframes = []
    worker = MapIngestWorker(fake, drop_stale=False, on_keyframe=lambda: keyframes.append(1))
    worker.start()
    for i in range(50):
        worker.submit(i, 7)
    worker.stop()
    assert fake.seen == [(i, 7) for i in range(50)]
    assert worker.dropped == 0 and worker.ingested == 50
    assert len(keyframes) == len([i for i in range(50) if i % 3 == 0])
    assert worker.last_ms["grid_ms"] == 1.0 and worker.timings.summary()["frames"] == 50


def test_live_mode_never_blocks_and_keeps_the_newest():
    fake = SlowMap()
    fake.release.clear()  # the worker is stuck inside the first update
    worker = MapIngestWorker(fake, drop_stale=True)
    worker.start()
    worker.submit(0, 1)
    time.sleep(0.05)
    started = time.monotonic()
    for i in range(1, 20):
        worker.submit(i, 1)
    assert time.monotonic() - started < 0.05  # the producer did not wait
    fake.release.set()
    worker.stop()
    assert [odom for odom, _ in fake.seen] == [0, 19]
    assert worker.dropped == 18


def test_worker_error_reaches_the_producer():
    fake = SlowMap(fail_on=3)
    worker = MapIngestWorker(fake, drop_stale=False)
    worker.start()
    for i in range(4):
        worker.submit(i, None)
    assert worker.wait_idle(5)
    with pytest.raises(RuntimeError, match="地图更新失败"):
        worker.submit(4, None)
    with pytest.raises(RuntimeError):
        worker.stop()
    assert [odom for odom, _ in fake.seen] == [0, 1, 2]


def test_stop_ingests_the_pending_result():
    fake = SlowMap()
    fake.release.clear()
    worker = MapIngestWorker(fake, drop_stale=True)
    worker.start()
    worker.submit(0, None)
    time.sleep(0.05)
    worker.submit(1, None)
    fake.release.set()
    worker.stop()
    assert [odom for odom, _ in fake.seen] == [0, 1]
    with pytest.raises(RuntimeError):
        worker.submit(2, None)
