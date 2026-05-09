"""Pure-Python spike detection + throttling for cluster nodes.

These helpers carry no ROS dependencies so they can be unit-tested and reused
across the action and service variants of the cluster_poses nodes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class SpikeReading:
    rate: float  # detections per second over the rolling window
    is_spike: bool


class SpikeDetector:
    """Tracks detection rate over a rolling time window.

    `update` is called with the current monotonic-ish time (in seconds, float)
    and the cumulative count of detections seen so far. Returns the current
    rate and whether it crosses the configured threshold.

    Set `rate_threshold <= 0` to disable spike detection (is_spike always False).
    """

    def __init__(self, window_sec: float, rate_threshold: float) -> None:
        self._window_sec = float(window_sec)
        self._rate_threshold = float(rate_threshold)
        self._samples: deque[tuple[float, int]] = deque()

    def reset(self) -> None:
        self._samples.clear()

    def update(self, now_sec: float, count: int) -> SpikeReading:
        self._samples.append((now_sec, count))
        while self._samples and (now_sec - self._samples[0][0]) > self._window_sec:
            self._samples.popleft()

        if len(self._samples) < 2:
            return SpikeReading(rate=0.0, is_spike=False)

        dt = self._samples[-1][0] - self._samples[0][0]
        if dt <= 0.0:
            return SpikeReading(rate=0.0, is_spike=False)

        rate = (self._samples[-1][1] - self._samples[0][1]) / dt
        is_spike = self._rate_threshold > 0.0 and rate >= self._rate_threshold
        return SpikeReading(rate=rate, is_spike=is_spike)


class ThrottledTrigger:
    """Allows an event to fire at most once per `min_interval_sec`."""

    def __init__(self, min_interval_sec: float) -> None:
        self._min_interval_sec = float(min_interval_sec)
        self._last_fire_sec: float | None = None

    def reset(self) -> None:
        self._last_fire_sec = None

    def test(self, now_sec: float) -> bool:
        if (
            self._last_fire_sec is None
            or (now_sec - self._last_fire_sec) >= self._min_interval_sec
        ):
            self._last_fire_sec = now_sec
            return True
        return False
