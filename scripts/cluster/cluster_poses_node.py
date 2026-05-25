#!/usr/bin/env python3
from dataclasses import dataclass
from operator import attrgetter

import numpy as np
import tf2_ros
from bb_filters.clustering.cluster import ClusterResult, get_largest_cluster
from bb_filters.clustering.pose import get_average_pose
from geometry_msgs.msg import (
    PoseArray,
    PoseStamped,
    Quaternion,
    TransformStamped,
    Vector3,
)
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sklearn.cluster import HDBSCAN
from tf2_msgs.msg import TFMessage


def seconds_to_duration(seconds: float) -> Duration:
    """Convert float seconds to rclpy Duration."""
    sec_int, sec_frac = divmod(seconds, 1)
    return Duration(seconds=int(sec_int), nanoseconds=int(round(sec_frac * 1e9)))


@dataclass(frozen=True)
class ClusterParams:
    min_cluster_size: int
    min_samples: int
    cluster_selection_epsilon: float


class ClusterPosesNode(Node):
    """Base node that collects synchronized pose + odom pairs and clusters them."""

    def __init__(self, node_name: str) -> None:
        super().__init__(node_name)

        self._tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10))
        # Subscribe only to /tf_static to avoid processing dynamic TF
        static_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._tf_static_sub = self.create_subscription(
            TFMessage,
            "/tf_static",
            self._handle_tf_static,
            qos_profile=static_qos,
        )

        self._synchronized_data: list[tuple[Odometry, PoseStamped]] = []
        self._camera_to_odom_transform: TransformStamped | None = None
        self._odom_subscriber: Subscriber | None = None
        self._pose_subscriber: Subscriber | None = None
        self._time_synchronizer: ApproximateTimeSynchronizer | None = None

        # Declare parameters
        output_pose_array_topic = (
            self.declare_parameter("output_pose_array_topic", "clustered_poses")
            .get_parameter_value()
            .string_value
        )
        self._sync_queue_size = (
            self.declare_parameter("sync_queue_size", 100)
            .get_parameter_value()
            .integer_value
        )

        # Publishers
        self._pose_array_publisher = self.create_publisher(
            PoseArray,
            output_pose_array_topic,
            10,
        )
        self._static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

    def _handle_tf_static(self, msg: TFMessage) -> None:
        for transform in msg.transforms:
            self._tf_buffer.set_transform_static(transform, "default_authority")

    def _synchronized_callback(self, odom_msg: Odometry, pose_msg: PoseStamped) -> None:
        self._synchronized_data.append((odom_msg, pose_msg))

    def _reset_collection(self) -> None:
        self._synchronized_data = []
        self._camera_to_odom_transform = None

    def _ensure_camera_to_odom(
        self,
        synchronized_data: list[tuple[Odometry, PoseStamped]],
    ) -> bool:
        if self._camera_to_odom_transform is not None:
            return True
        if not synchronized_data:
            return False

        odom_child_frame = synchronized_data[0][0].child_frame_id
        camera_frame_id = synchronized_data[0][1].header.frame_id
        try:
            self._camera_to_odom_transform = self._tf_buffer.lookup_transform(
                odom_child_frame,
                camera_frame_id,
                Time(),
                timeout=Duration(seconds=5),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            self.get_logger().error(f"Failed to lookup transform: {exc}")
            return False

        self.get_logger().info(
            f"Found transform from {camera_frame_id} to {odom_child_frame}"
        )
        return True

    def _start_subscribers(
        self,
        *,
        odom_topic: str,
        pose_topic: str,
        sync_tolerance: float,
        sync_queue_size: int | None = None,
    ) -> None:
        self._odom_subscriber = Subscriber(
            self,
            Odometry,
            odom_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self._pose_subscriber = Subscriber(
            self,
            PoseStamped,
            pose_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self._time_synchronizer = ApproximateTimeSynchronizer(
            [self._odom_subscriber, self._pose_subscriber],
            queue_size=int(sync_queue_size or self._sync_queue_size),
            slop=float(sync_tolerance),
        )
        self._time_synchronizer.registerCallback(self._synchronized_callback)

    def _cleanup_subscribers(self) -> None:
        for sub in (self._odom_subscriber, self._pose_subscriber):
            if sub is None:
                continue
            try:
                self.destroy_subscription(sub.sub)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warning(f"Subscriber cleanup failed: {exc}")
        self._odom_subscriber = None
        self._pose_subscriber = None
        self._time_synchronizer = None

    def _cluster_poses(
        self,
        transformed_poses: list[PoseStamped],
        params: ClusterParams,
    ) -> tuple[PoseStamped | None, ClusterResult]:
        if len(transformed_poses) < max(params.min_cluster_size, params.min_samples):
            self.get_logger().error("Not enough poses for clustering")
            return None, ClusterResult.empty(num_input_poses=len(transformed_poses))

        hdbscan = HDBSCAN(
            min_cluster_size=int(params.min_cluster_size),
            min_samples=int(params.min_samples),
            cluster_selection_epsilon=float(params.cluster_selection_epsilon),
            allow_single_cluster=True,
            store_centers="centroid",
        )
        positions = np.array(
            [
                attrgetter("x", "y", "z")(pose.pose.position)
                for pose in transformed_poses
            ]
        )
        cluster_result = get_largest_cluster(hdbscan, positions)
        if len(cluster_result.idxs) == 0:
            self.get_logger().error("No clusters found")
            return None, cluster_result

        filtered_pose_msgs = [transformed_poses[i].pose for i in cluster_result.idxs]
        avg_pose = PoseStamped()
        avg_pose.pose = get_average_pose(filtered_pose_msgs)
        avg_pose.header = transformed_poses[cluster_result.idxs[0]].header
        return avg_pose, cluster_result

    def _publish_results(
        self,
        avg_pose: PoseStamped,
        transformed_poses: list[PoseStamped],
        clustered_child_frame_id: str,
    ) -> None:
        pose_array_msg = PoseArray()
        pose_array_msg.header = avg_pose.header
        pose_array_msg.poses = [pose.pose for pose in transformed_poses]
        self._pose_array_publisher.publish(pose_array_msg)

        transform_stamped = TransformStamped()
        transform_stamped.header = avg_pose.header
        transform_stamped.child_frame_id = clustered_child_frame_id
        t = attrgetter("x", "y", "z")(avg_pose.pose.position)
        qx, qy, qz, qw = attrgetter("x", "y", "z", "w")(avg_pose.pose.orientation)
        transform_stamped.transform.translation = Vector3(x=t[0], y=t[1], z=t[2])
        transform_stamped.transform.rotation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        self._static_tf_broadcaster.sendTransform(transform_stamped)
