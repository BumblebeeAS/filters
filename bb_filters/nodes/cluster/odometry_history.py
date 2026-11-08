"""Retain odometry between collections without retaining detection poses."""
from collections import deque

from frames.utils.sync_diagnostics import stamp_ns
from message_filters import SimpleFilter
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data


class OdometryHistory:
    """One persistent subscription and a fresh synchronizer relay per collection."""

    def __init__(self, node, seconds=5., capacity=2000):
        self.node, self.seconds, self.capacity = node, seconds, capacity
        self.messages = deque(maxlen=capacity)
        self.subscription = self.topic = self.relay = None

    def connect(self, topic):
        if self.topic != topic:
            if self.subscription is not None:
                self.node.destroy_subscription(self.subscription)
            self.messages.clear()
            self.topic = topic
            self.subscription = self.node.create_subscription(
                Odometry, topic, self.remember, qos_profile_sensor_data)
        self.relay = SimpleFilter()
        return self.relay

    def remember(self, message):
        stamp = stamp_ns(message)
        if self.messages and stamp < stamp_ns(self.messages[-1]):
            self.messages.clear()  # Never replay across a clock/epoch reset.
        self.messages.append(message)
        cutoff = stamp - self.seconds * 1e9
        while self.messages and stamp_ns(self.messages[0]) < cutoff:
            self.messages.popleft()
        if self.relay is not None:
            self.relay.signalMessage(message)

    def replay(self):
        # Replay odometry only into the new synchronizers. Detection samples
        # always belong to the current collection and are never replayed.
        now = self.node.get_clock().now().nanoseconds
        for message in self.messages:
            if now - self.seconds * 1e9 <= stamp_ns(message) <= now + 1e8:
                self.relay.signalMessage(message)

    def disconnect(self):
        self.relay = None  # Old synchronizers must not receive future odometry.
