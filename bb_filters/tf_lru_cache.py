#!/usr/bin/env python3

from geometry_msgs.msg import TransformStamped
from rclpy.time import Time


# TODO: use a fixed sized numpy array then can make everything faster
class TfLruCache:
    def __init__(self, size: int, logger):
        self.size = size

        # idx is the current insertion index (the open spot in the circular buffer)
        self.idx = 0

        self.cache = [None] * self.size
        self.logger = logger

        self.oldest_time = Time()
        self.latest_time = Time()

        self.is_empty_flag = True
        self.count = 0  # number of elements in cache

    @property
    def is_full(self) -> bool:
        return self.count >= self.size

    def _get(self, idx: int) -> TransformStamped:
        return self.cache[idx % self.size]

    def _set(self, tf: TransformStamped):
        self.cache[self.idx] = tf
        self.latest_time = Time.from_msg(tf.header.stamp)
        self.idx = (self.idx + 1) % self.size
        self.count += 1

    def add(self, tf: TransformStamped):
        if self.is_empty_flag:
            self.oldest_time = Time.from_msg(tf.header.stamp)
            self.is_empty_flag = False
            self._set(tf)
            return True

        prev_tf = self._get(self.idx - 1)

        if Time.from_msg(tf.header.stamp) == Time.from_msg(prev_tf.header.stamp):
            self.logger.warn(
                f"Skipping TF with timestamp {tf.header.stamp} as it is the same as the previous one."
            )
            return False

        if self.is_full:
            self.oldest_time = Time.from_msg(self._get(self.idx + 1).header.stamp)

        self._set(tf)
        return True

    def get_oldest_time(self) -> Time:
        return self.oldest_time

    def get_latest_time(self) -> Time:
        return self.latest_time

    def is_empty(self) -> bool:
        return self.is_empty_flag

    def get_all(self) -> tuple[list[TransformStamped], Time]:
        return (
            [self.cache[i] for i in range(self.size) if self.cache[i] is not None],
            self.get_latest_time(),
        )

    def get_count(self) -> int:
        return self.count
