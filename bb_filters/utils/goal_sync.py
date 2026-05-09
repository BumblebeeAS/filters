import threading

from geometry_msgs.msg import (
    PoseStamped,
)
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import (
    qos_profile_sensor_data,
)


class GoalSynchronizer:
    """Per-goal subscribers + synchronizer + cleanup guard, all on a private
    MutuallyExclusiveCallbackGroup.

    The guard's destroy callback shares the group with the subs, so the
    executor's own scheduling serializes destroy against any in-flight take.
    Bare destroy_subscription from another thread would race rcl_wait; this
    pattern doesn't. Ensure that the on_synchronized callback is thread safe
    """

    def __init__(
        self,
        node: Node,
        odom_topic: str,
        pose_topic: str,
        slop: float,
        queue_size: int,
        on_synchronized,
    ):
        self._node = node
        self._on_synchronized = on_synchronized
        self._accepting = True
        self._destroyed = threading.Event()

        self.group = MutuallyExclusiveCallbackGroup()

        self.odom_sub = Subscriber(
            node,
            Odometry,
            odom_topic,
            qos_profile=qos_profile_sensor_data,
            callback_group=self.group,
        )
        self.pose_sub = Subscriber(
            node,
            PoseStamped,
            pose_topic,
            qos_profile=qos_profile_sensor_data,
            callback_group=self.group,
        )
        self.sync = ApproximateTimeSynchronizer(
            [self.odom_sub, self.pose_sub],
            queue_size=queue_size,
            slop=slop,
        )
        self.sync.registerCallback(self._gated_callback)

        self._cleanup_guard = node.create_guard_condition(
            self._do_destroy, callback_group=self.group
        )

    def _gated_callback(self, odom_msg: Odometry, pose_msg: PoseStamped) -> None:
        if not self._accepting:
            return
        self._on_synchronized(odom_msg, pose_msg)

    def shutdown(self) -> None:
        """Request destruction. Returns immediately; actual destroy runs on
        the executor when the group is free."""
        self._accepting = False
        try:
            self._cleanup_guard.trigger()
        except Exception as e:
            self._node.get_logger().warning(f"cleanup trigger failed: {e}")

    def wait_destroyed(self, timeout: float | None = None) -> bool:
        return self._destroyed.wait(timeout)

    def _do_destroy(self) -> None:
        # Runs on an executor worker, holding this group's mutex slot.
        # No sub take can be in flight while this runs.
        for sub, name in (
            (self.odom_sub.sub, "odom"),
            (self.pose_sub.sub, "pose"),
        ):
            try:
                self._node.destroy_subscription(sub)
            except Exception as e:
                self._node.get_logger().warning(f"{name} destroy failed: {e}")
        try:
            self._node.destroy_guard_condition(self._cleanup_guard)
        except Exception as e:
            self._node.get_logger().warning(f"cleanup guard destroy failed: {e}")
        self._destroyed.set()
