#!/usr/bin/env python3
from dataclasses import dataclass
from operator import attrgetter

import numpy as np
import tf2_ros
from bb_filters.utils.cluster.cluster import (
    ClusterResult,
    ClusterSortKey,
    get_all_clusters,
    sort_clusters,
)
from bb_filters.utils.pose import get_average_pose
from bb_perception_msgs.msg import ClusterPoseResult, ClusterPoseResultArray
from frames.utils.transform_ros_msgs import transform_pose_to_odom
from geometry_msgs.msg import (
    Pose,
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
from rclpy.publisher import Publisher
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
    # Minimum buffer size before clustering is even attempted. Acts as a
    # pre-clustering rejection threshold, separate from HDBSCAN's own knobs.
    min_poses: int = 10
    top_k: int = 1
    sort_key: ClusterSortKey = ClusterSortKey.NUM_CLUSTER_POSES


def build_cluster_pose_result(
    avg_pose: Pose, cluster_result: ClusterResult, num_input_poses: int
) -> ClusterPoseResult:
    """Project a clustered (Pose, ClusterResult) pair into a ROS message."""
    msg = ClusterPoseResult()
    msg.clustered_pose = avg_pose
    msg.clustered_position_std = float(cluster_result.clustered_position_std)
    msg.num_cluster_poses = int(cluster_result.num_cluster_poses)
    msg.num_input_poses = int(num_input_poses)
    msg.mean_probability = float(cluster_result.mean_probability)
    return msg


def fill_cluster_result_array(
    msg: ClusterPoseResultArray,
    clustered: list[tuple[Pose, ClusterResult]],
    num_input_poses: int,
    header,
    sort_key: int,
) -> None:
    """Populate a ClusterPoseResultArray from clustered (Pose, ClusterResult) pairs."""
    msg.header = header
    msg.sort_key = int(sort_key)
    msg.results = [
        build_cluster_pose_result(avg_pose, cluster_result, num_input_poses)
        for avg_pose, cluster_result in clustered
    ]


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
        self._max_detection_age_s = 0.0

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
        self._cluster_pose_array_publishers: dict[str, Publisher] = {}
        self._static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

    def _handle_tf_static(self, msg: TFMessage) -> None:
        for transform in msg.transforms:
            self._tf_buffer.set_transform_static(transform, "default_authority")

    def _synchronized_callback(self, odom_msg: Odometry, pose_msg: PoseStamped) -> None:
        self._synchronized_data.append((odom_msg, pose_msg))

    def _is_detection_too_old(
        self, clustering_time: Time, pose_msg: PoseStamped
    ) -> bool:
        if self._max_detection_age_s <= 0.0:
            return False

        detection_age_s = (
            clustering_time - Time.from_msg(pose_msg.header.stamp)
        ).nanoseconds / 1e9
        return detection_age_s >= self._max_detection_age_s

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
        max_detection_age_s: float = 0.0,
    ) -> None:
        """Subscribe to one odom topic and N pose topics, all feeding one buffer."""
        if not pose_topics:
            raise ValueError("pose_topics must contain at least one topic")
        self._max_detection_age_s = float(max_detection_age_s)
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

    def _run_clustering(
        self,
        params: ClusterParams,
    ) -> tuple[
        list[tuple[Pose, ClusterResult]],
        list[PoseStamped],
        int,
    ]:
        """Run the full clustering pipeline against the current buffer.

        Returns `(clustered, transformed_poses, total_collected)`
        """
        if not self._synchronized_data:
            total_collected = 0
            self.get_logger().error(
                "Not enough synchronized poses collected. "
                f"Got {total_collected}, need {int(params.min_poses)}"
            )
            return [], [], total_collected

        # Match the result PoseArray/ClusterPoseResultArray timestamp, which is
        # taken from the last transformed pose after this age filter.
        clustering_time = Time.from_msg(self._synchronized_data[-1][1].header.stamp)
        self._synchronized_data = [
            (odom_msg, pose_msg)
            for odom_msg, pose_msg in self._synchronized_data
            if not self._is_detection_too_old(clustering_time, pose_msg)
        ]
        synchronized_data = self._synchronized_data
        total_collected = len(synchronized_data)

        if total_collected < int(params.min_poses):
            self.get_logger().error(
                "Not enough synchronized poses collected. "
                f"Got {total_collected}, need {int(params.min_poses)}"
            )
            return [], [], total_collected

        if not self._ensure_camera_to_odom(synchronized_data):
            return [], [], total_collected

        transformed_poses = [
            transform_pose_to_odom(
                odom_msg,
                pose_msg,
                self._camera_to_odom_transforms[pose_msg.header.frame_id],
            )
            for odom_msg, pose_msg in synchronized_data
        ]
        clustered = self._cluster_poses(transformed_poses, params)
        return clustered, transformed_poses, total_collected

    def _cluster_poses(
        self,
        transformed_poses: list[PoseStamped],
        params: ClusterParams,
    ) -> list[tuple[Pose, ClusterResult]]:
        """Cluster `transformed_poses` and return the top_k results.

        Each entry is the averaged Pose of a cluster paired with its
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

        return [
            (
                get_average_pose([transformed_poses[i].pose for i in cluster.idxs]),
                cluster,
            )
            for cluster in clusters
        ]

    def _publish_results(
        self,
        clustered: list[tuple[Pose, ClusterResult]],
        transformed_poses: list[PoseStamped],
        clustered_child_frame_id: str,
    ) -> None:
        if not clustered:
            return

        # All cluster TFs and the input PoseArray share the most recent
        # transformed pose's header — clustering is a snapshot, so the
        # timestamp is one value, not one per cluster.
        header = transformed_poses[-1].header

        all_transformed_poses_msg = PoseArray()
        all_transformed_poses_msg.header = header
        all_transformed_poses_msg.poses = [pose.pose for pose in transformed_poses]
        self._pose_array_publisher.publish(all_transformed_poses_msg)

        transforms: list[TransformStamped] = []
        for cluster_index, (avg_pose, cluster_result) in enumerate(clustered):
            cluster_child_frame_id = f"{clustered_child_frame_id}_{cluster_index}"
            self._publish_cluster_pose_array(
                cluster_child_frame_id,
                transformed_poses,
                cluster_result,
                header,
            )

            transform_stamped = TransformStamped()
            transform_stamped.header = header
            transform_stamped.child_frame_id = cluster_child_frame_id
            t = attrgetter("x", "y", "z")(avg_pose.position)
            qx, qy, qz, qw = attrgetter("x", "y", "z", "w")(avg_pose.orientation)
            transform_stamped.transform.translation = Vector3(x=t[0], y=t[1], z=t[2])
            transform_stamped.transform.rotation = Quaternion(x=qx, y=qy, z=qz, w=qw)
            transforms.append(transform_stamped)
        self._static_tf_broadcaster.sendTransform(transforms)

    def _publish_cluster_pose_array(
        self,
        cluster_child_frame_id: str,
        transformed_poses: list[PoseStamped],
        cluster_result: ClusterResult,
        header,
    ) -> None:
        cluster_pose_array_msg = PoseArray()
        cluster_pose_array_msg.header = header
        cluster_pose_array_msg.poses = [
            transformed_poses[pose_index].pose for pose_index in cluster_result.idxs
        ]

        topic_name = f"{cluster_child_frame_id}/poses"
        if topic_name not in self._cluster_pose_array_publishers:
            self._cluster_pose_array_publishers[topic_name] = self.create_publisher(
                PoseArray, topic_name, 10
            )

        self._cluster_pose_array_publishers[topic_name].publish(cluster_pose_array_msg)

    def _cleanup_cluster_pose_publishers(self) -> None:
        for topic_name in list(self._cluster_pose_array_publishers):
            publisher = self._cluster_pose_array_publishers.pop(topic_name)
            if not self.destroy_publisher(publisher):
                self.get_logger().warning(
                    f"Failed to destroy cluster pose publisher: {topic_name}"
                )
