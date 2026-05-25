#!/usr/bin/env python3
from dataclasses import dataclass
from operator import attrgetter

import numpy as np
import tf2_ros
from bb_filters.clustering.cluster import (
    ClusterResult,
    ClusterSortKey,
    get_all_clusters,
    sort_clusters,
)
from bb_filters.clustering.pose import get_average_pose
from bb_perception_msgs.msg import ClusterPoseResult
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
    top_k: int = 1
    sort_key: ClusterSortKey = ClusterSortKey.NUM_CLUSTER_POSES


def build_cluster_pose_result(
    avg_pose: PoseStamped,
    cluster_result: ClusterResult,
    num_input_poses: int,
) -> ClusterPoseResult:
    """Project a clustered (PoseStamped, ClusterResult) pair into a ROS message.

    The PoseStamped's header is dropped — the array carries the timestamp once.
    """
    msg = ClusterPoseResult()
    msg.clustered_pose = avg_pose.pose
    msg.clustered_position_std = float(cluster_result.clustered_position_std)
    msg.num_cluster_poses = int(cluster_result.num_cluster_poses)
    msg.num_input_poses = int(num_input_poses)
    msg.mean_probability = float(cluster_result.mean_probability)
    return msg


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
        self._camera_to_odom_transforms: dict[str, TransformStamped] = {}
        self._odom_subscriber: Subscriber | None = None
        self._pose_subscribers: list[Subscriber] = []
        self._time_synchronizers: list[ApproximateTimeSynchronizer] = []

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
        self._camera_to_odom_transforms = {}

    def _ensure_camera_to_odom(
        self,
        synchronized_data: list[tuple[Odometry, PoseStamped]],
    ) -> bool:
        """Look up odom_child -> camera transforms for every unique source frame."""
        if not synchronized_data:
            return False

        odom_child_frame = synchronized_data[0][0].child_frame_id
        camera_frames = {pose.header.frame_id for _, pose in synchronized_data}
        for camera_frame_id in camera_frames:
            if camera_frame_id in self._camera_to_odom_transforms:
                continue
            try:
                tf = self._tf_buffer.lookup_transform(
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
                self.get_logger().error(
                    f"Failed to lookup transform from {camera_frame_id} to "
                    f"{odom_child_frame}: {exc}"
                )
                return False
            self._camera_to_odom_transforms[camera_frame_id] = tf
            self.get_logger().info(
                f"Found transform from {camera_frame_id} to {odom_child_frame}"
            )
        return True

    def _start_subscribers(
        self,
        *,
        odom_topic: str,
        pose_topics: list[str],
        sync_tolerance: float,
        sync_queue_size: int | None = None,
    ) -> None:
        """Subscribe to one odom topic and N pose topics, all feeding one buffer."""
        if not pose_topics:
            raise ValueError("pose_topics must contain at least one topic")
        qsize = int(sync_queue_size or self._sync_queue_size)
        self._odom_subscriber = Subscriber(
            self,
            Odometry,
            odom_topic,
            qos_profile=qos_profile_sensor_data,
        )
        for topic in pose_topics:
            pose_sub = Subscriber(
                self,
                PoseStamped,
                topic,
                qos_profile=qos_profile_sensor_data,
            )
            self._pose_subscribers.append(pose_sub)
            time_synchronizer = ApproximateTimeSynchronizer(
                [self._odom_subscriber, pose_sub],
                queue_size=qsize,
                slop=float(sync_tolerance),
            )
            time_synchronizer.registerCallback(self._synchronized_callback)
            self._time_synchronizers.append(time_synchronizer)

    def _cleanup_subscribers(self) -> None:
        subs: list[Subscriber] = []
        if self._odom_subscriber is not None:
            subs.append(self._odom_subscriber)
        subs.extend(self._pose_subscribers)
        for sub in subs:
            try:
                self.destroy_subscription(sub.sub)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warning(f"Subscriber cleanup failed: {exc}")
        self._odom_subscriber = None
        self._pose_subscribers = []
        self._time_synchronizers = []

    def _cluster_poses(
        self,
        transformed_poses: list[PoseStamped],
        params: ClusterParams,
    ) -> list[tuple[PoseStamped, ClusterResult]]:
        """Cluster `transformed_poses` and return the top_k results.

        Each entry is the average PoseStamped of a cluster paired with its
        `ClusterResult`. The list is ordered by `params.sort_key`, best-first,
        and truncated to `params.top_k` (0 = keep all).
        """
        if len(transformed_poses) < max(params.min_cluster_size, params.min_samples):
            self.get_logger().error("Not enough poses for clustering")
            return []

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
        clusters = get_all_clusters(hdbscan, positions)
        if not clusters:
            self.get_logger().error("No clusters found")
            return []

        clusters = sort_clusters(clusters, params.sort_key)
        if int(params.top_k) > 0:
            clusters = clusters[: int(params.top_k)]

        results: list[tuple[PoseStamped, ClusterResult]] = []
        for cluster in clusters:
            cluster_poses = [transformed_poses[i].pose for i in cluster.idxs]
            avg_pose = PoseStamped()
            avg_pose.pose = get_average_pose(cluster_poses)
            avg_pose.header = transformed_poses[cluster.idxs[0]].header
            results.append((avg_pose, cluster))
        return results

    def _publish_results(
        self,
        clustered: list[tuple[PoseStamped, ClusterResult]],
        transformed_poses: list[PoseStamped],
        clustered_child_frame_id: str,
    ) -> None:
        if not clustered:
            return

        # Pose array of every input pose — still useful for debugging/visualization.
        pose_array_msg = PoseArray()
        pose_array_msg.header = clustered[0][0].header
        pose_array_msg.poses = [pose.pose for pose in transformed_poses]
        self._pose_array_publisher.publish(pose_array_msg)

        transforms: list[TransformStamped] = []
        for i, (avg_pose, _) in enumerate(clustered):
            transform_stamped = TransformStamped()
            transform_stamped.header = avg_pose.header
            transform_stamped.child_frame_id = f"{clustered_child_frame_id}_{i}"
            t = attrgetter("x", "y", "z")(avg_pose.pose.position)
            qx, qy, qz, qw = attrgetter("x", "y", "z", "w")(avg_pose.pose.orientation)
            transform_stamped.transform.translation = Vector3(x=t[0], y=t[1], z=t[2])
            transform_stamped.transform.rotation = Quaternion(x=qx, y=qy, z=qz, w=qw)
            transforms.append(transform_stamped)
        self._static_tf_broadcaster.sendTransform(transforms)
