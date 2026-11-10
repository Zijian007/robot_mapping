"""Bounded wall-time samples, shared with session status readers."""

import threading
from collections import deque

import numpy as np


class FrameTimings:
    def __init__(self, window=256):
        self._samples = deque(maxlen=window)
        self._lock = threading.Lock()

    def add(self, sample):
        with self._lock:
            self._samples.append(dict(sample))

    def summary(self):
        with self._lock:
            samples = list(self._samples)
        if not samples:
            return {"frames": 0, "last_ms": {}, "stages_ms": {}}
        stages = {}
        for name in samples[-1]:
            values = np.array([s[name] for s in samples])
            p50, p95, p99 = np.percentile(values, [50, 95, 99])
            stages[name] = dict(
                p50=float(p50), p95=float(p95), p99=float(p99), max=float(values.max())
            )
        return {"frames": len(samples), "last_ms": dict(samples[-1]), "stages_ms": stages}
