import uuid

from geometry_msgs.msg import (
    PoseStamped,
)
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import Executor
from rclpy.node import Node
from rclpy.qos import (
    qos_profile_sensor_data,
)


class GoalSynchronizer:
    """Per-goal subscribers + synchronizer hosted on a dedicated child Node.

    Teardown uses Executor.remove_node + Node.destroy_node, which fences
    against the executor's wait loop: after remove_node returns, no entity
    on the child node can be dispatched, so destroying the subscriptions is
    free of the stale-ready-list race that the previous guard-condition
    pattern could not prevent.
    """

    def __init__(
        self,
        parent_node: Node,
        executor: Executor,
        odom_topic: str,
        pose_topic: str,
        slop: float,
        queue_size: int,
        on_synchronized,
    ):
        self._parent = parent_node
        self._executor = executor
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

        executor.add_node(self._child)

    def _gated_callback(self, odom_msg: Odometry, pose_msg: PoseStamped) -> None:
        if not self._accepting:
            return
        self._on_synchronized(odom_msg, pose_msg)

    def shutdown(self) -> None:
        """Tear down the channel. remove_node fences the executor's wait
        loop; destroy_node then reaps rcl handles without racing any take."""
        self._accepting = False
        try:
            self._executor.remove_node(self._child)
        except Exception as e:
            self._parent.get_logger().warning(f"remove_node failed: {e}")
        try:
            self._child.destroy_node()
        except Exception as e:
            self._parent.get_logger().warning(f"destroy_node failed: {e}")
