import threading
import uuid

from geometry_msgs.msg import (
    PoseStamped,
)
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    qos_profile_sensor_data,
)


class GoalSynchronizer:
    """Per-goal child Node spun on its own SingleThreadedExecutor.

    Why a private executor: the parent's MultiThreadedExecutor can return a
    batch from rcl_wait that already references the child's subscriptions,
    so destroying those subs from any callback running on the parent races
    the in-flight batch ("cannot use destroyable" on the next take).
    Executor.shutdown() + thread join is a real synchronization point: once
    the spin thread has exited, no rcl_wait is in flight on the child's
    entities, so destroy_node() is safe.
    """

    def __init__(
        self,
        parent_node: Node,
        odom_topic: str,
        pose_topic: str,
        slop: float,
        queue_size: int,
        on_synchronized,
    ):
        self._parent = parent_node
        self._on_synchronized = on_synchronized
        self._accepting = True

        self._child = Node(
            f"_cluster_poses_channel_{uuid.uuid4().hex[:8]}",
            namespace=parent_node.get_namespace(),
            start_parameter_services=False,
        )

        self.group = MutuallyExclusiveCallbackGroup()

        self.odom_sub = Subscriber(
            self._child,
            Odometry,
            odom_topic,
            qos_profile=qos_profile_sensor_data,
            callback_group=self.group,
        )
        self.pose_sub = Subscriber(
            self._child,
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

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._child)
        self._spin_thread = threading.Thread(
            target=self._executor.spin,
            name="goal_sync_spin",
            daemon=True,
        )
        self._spin_thread.start()

    def _gated_callback(self, odom_msg: Odometry, pose_msg: PoseStamped) -> None:
        if not self._accepting:
            return
        self._on_synchronized(odom_msg, pose_msg)

    def shutdown(self, join_timeout: float = 2.0) -> None:
        self._accepting = False
        threading.Thread(
            target=self.cleanup,
            name="goal_sync_cleanup",
            daemon=True,
            kwargs={"timeout": join_timeout},
        ).start()

    def cleanup(self, timeout: float = 2.0) -> None:
        try:
            self._executor.shutdown()
        except Exception as e:
            self._parent.get_logger().warning(f"executor shutdown failed: {e}")
        self._spin_thread.join(timeout=timeout)
        if self._spin_thread.is_alive():
            self._parent.get_logger().warning(
                "goal_sync spin thread did not exit within timeout; "
                "skipping destroy_node to avoid racing a live wait loop"
            )
            return
        try:
            self._child.destroy_node()
        except Exception as e:
            self._parent.get_logger().warning(f"destroy_node failed: {e}")
