#!/usr/bin/env python3
from __future__ import annotations

from operator import attrgetter
from typing import Optional

import numpy as np
import rclpy
import tf2_ros
from bb_perception_msgs.srv import ClusterTfSrv
from frames.utils.transform_ros_msgs import transform_pose_to_odom
from geometry_msgs.msg import (
    PoseArray,
    PoseStamped,
    Quaternion,
    TransformStamped,
    Vector3,
)
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
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

from bb_filters.clustering.cluster import get_largest_cluster
from bb_filters.clustering.pose import get_average_pose


def seconds_to_duration(seconds: float) -> Duration:
    """Convert float seconds to rclpy Duration."""
    sec_int, sec_frac = divmod(seconds, 1)
    return Duration(seconds=int(sec_int), nanoseconds=int(round(sec_frac * 1e9)))


class ClusterPosesServiceNode(Node):
    def __init__(self) -> None:
        super().__init__("cluster_poses_service_node")

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10))

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

        self._service_group = ReentrantCallbackGroup()

        # Single service callback pattern (enabled=True to start, enabled=False to stop+cluster)
        self._service_server = self.create_service(
            ClusterTfSrv,
            "cluster_poses_srv",
            self.cluster_srv_callback,
            callback_group=self._service_group,
        )

        # --------------------- Topics and synchronization parameters --------------------
        self.output_pose_array_topic = (
            self.declare_parameter("output_pose_array_topic", "clustered_poses")
            .get_parameter_value()
            .string_value
        )
        self.odom_topic = (
            self.declare_parameter("odom_topic", "/odom")
            .get_parameter_value()
            .string_value
        )
        self.pose_stamped_topic = (
            self.declare_parameter("pose_stamped_topic", "/pose")
            .get_parameter_value()
            .string_value
        )
        self.clustered_child_frame_id = (
            self.declare_parameter("clustered_child_frame_id", "clustered_pose")
            .get_parameter_value()
            .string_value
        )
        self.sync_queue_size = (
            self.declare_parameter("sync_queue_size", 100)
            .get_parameter_value()
            .integer_value
        )
        self.sync_tolerance = (
            self.declare_parameter("sync_tolerance", 0.05)
            .get_parameter_value()
            .double_value
        )

        # -------------------- Clustering parameters --------------------
        self.min_poses = (
            self.declare_parameter("min_poses", 10).get_parameter_value().integer_value
        )
        self.min_cluster_size = (
            self.declare_parameter("min_cluster_size", 5)
            .get_parameter_value()
            .integer_value
        )
        self.min_samples = (
            self.declare_parameter("min_samples", 5).get_parameter_value().integer_value
        )
        self.cluster_selection_epsilon = (
            self.declare_parameter("cluster_selection_epsilon", 0.0)
            .get_parameter_value()
            .double_value
        )

        self.pose_array_publisher = self.create_publisher(
            PoseArray, self.output_pose_array_topic, 10
        )
        self._static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        # State
        self.enabled: bool = False
        self._synchronized_data: list[tuple[Odometry, PoseStamped]] = []
        self._camera_to_odom_transform: Optional[TransformStamped] = None
        self._odom_subscriber: Optional[Subscriber] = None
        self._pose_subscriber: Optional[Subscriber] = None
        self._time_synchronizer: Optional[ApproximateTimeSynchronizer] = None

        self.get_logger().info("Cluster Poses Service Node initialized")

    def cluster_srv_callback(
        self, request: ClusterTfSrv.Request, response: ClusterTfSrv.Response
    ) -> ClusterTfSrv.Response:
        """Service callback.

        - request.enabled == True: start accumulating
        - request.enabled == False: stop + cluster + publish

        This mirrors the enable/disable pattern used by `cluster_tf_service_server.py`.
        """

        if not request.enabled:
            # Stop accumulating and perform clustering
            if not self.enabled:
                response.is_enabled = False
                response.is_cluster_success = False
                response.cluster_spread = 0.0
                return response

            self.enabled = False

            total_collected = len(self._synchronized_data)
            if total_collected < int(self.min_poses):
                self._cleanup_subscribers()
                response.is_enabled = False
                response.is_cluster_success = False
                response.cluster_spread = 0.0
                return response

            odom_child_frame = self._synchronized_data[0][0].child_frame_id
            camera_frame_id = self._synchronized_data[0][1].header.frame_id

            try:
                self._camera_to_odom_transform = self.tf_buffer.lookup_transform(
                    odom_child_frame,
                    camera_frame_id,
                    Time(),
                    timeout=Duration(seconds=5),
                )
            except Exception as e:
                self.get_logger().error(f"Failed to lookup transform: {e}")
                self._cleanup_subscribers()
                response.is_enabled = False
                response.is_cluster_success = False
                response.cluster_spread = 0.0
                return response

            transformed_poses = [
                transform_pose_to_odom(
                    odom_msg, pose_msg, self._camera_to_odom_transform
                )
                for odom_msg, pose_msg in self._synchronized_data
            ]

            avg_pose, num_poses_in_cluster = self._cluster_poses(transformed_poses)
            if avg_pose is None:
                self._cleanup_subscribers()
                response.is_enabled = False
                response.is_cluster_success = False
                response.cluster_spread = 0.0
                return response

            avg_pose.header = transformed_poses[-1].header
            self._publish_results(
                avg_pose, transformed_poses, self.clustered_child_frame_id
            )
            self._cleanup_subscribers()

            response.is_enabled = False
            response.is_cluster_success = True
            # Reuse cluster_spread field to report how many poses were used in the cluster.
            response.cluster_spread = float(num_poses_in_cluster)
            return response

        # Start accumulating
        self._synchronized_data = []

        self._odom_subscriber = Subscriber(
            self, Odometry, self.odom_topic, qos_profile=qos_profile_sensor_data
        )
        self._pose_subscriber = Subscriber(
            self,
            PoseStamped,
            self.pose_stamped_topic,
            qos_profile=qos_profile_sensor_data,
        )

        self._time_synchronizer = ApproximateTimeSynchronizer(
            [self._odom_subscriber, self._pose_subscriber],
            queue_size=self.sync_queue_size,
            slop=self.sync_tolerance,
        )
        self._time_synchronizer.registerCallback(self._synchronized_callback)

        self.enabled = True
        response.is_enabled = True
        response.is_cluster_success = False
        response.cluster_spread = 0.0
        return response

    def _synchronized_callback(self, odom_msg: Odometry, pose_msg: PoseStamped) -> None:
        if not self.enabled:
            return
        self._synchronized_data.append((odom_msg, pose_msg))

    def _cluster_poses(
        self, transformed_poses: list[PoseStamped]
    ) -> tuple[Optional[PoseStamped], int]:
        if len(transformed_poses) < max(
            int(self.min_cluster_size), int(self.min_samples)
        ):
            self.get_logger().error("Not enough poses for clustering")
            return None, 0

        hdbscan = HDBSCAN(
            min_cluster_size=int(self.min_cluster_size),
            min_samples=int(self.min_samples),
            cluster_selection_epsilon=float(self.cluster_selection_epsilon),
            allow_single_cluster=True,
            store_centers="centroid",
        )

        positions = np.array(
            [
                attrgetter("x", "y", "z")(pose.pose.position)
                for pose in transformed_poses
            ]
        )
        filtered_idxs = get_largest_cluster(hdbscan, positions).idxs

        if len(filtered_idxs) == 0:
            self.get_logger().error("No clusters found")
            return None, 0

        filtered_pose_msgs = [transformed_poses[i].pose for i in filtered_idxs]
        avg_pose = get_average_pose(filtered_pose_msgs)
        avg_pose_stamped = PoseStamped()
        avg_pose_stamped.pose = avg_pose
        avg_pose_stamped.header = transformed_poses[filtered_idxs[0]].header

        return avg_pose_stamped, len(filtered_idxs)

    def _publish_results(
        self,
        avg_pose: PoseStamped,
        transformed_poses: list[PoseStamped],
        clustered_child_frame_id: str,
    ) -> None:
        pose_array_msg = PoseArray()
        pose_array_msg.header = avg_pose.header
        pose_array_msg.poses = [pose.pose for pose in transformed_poses]
        self.pose_array_publisher.publish(pose_array_msg)

        transform_stamped = TransformStamped()
        transform_stamped.header = avg_pose.header
        transform_stamped.child_frame_id = clustered_child_frame_id
        t = attrgetter("x", "y", "z")(avg_pose.pose.position)
        qx, qy, qz, qw = attrgetter("x", "y", "z", "w")(avg_pose.pose.orientation)
        transform_stamped.transform.translation = Vector3(x=t[0], y=t[1], z=t[2])
        transform_stamped.transform.rotation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        self._static_tf_broadcaster.sendTransform(transform_stamped)

    def _cleanup_subscribers(self) -> None:
        if self._time_synchronizer is not None:
            self._time_synchronizer = None

        if self._odom_subscriber is not None:
            try:
                self.destroy_subscription(self._odom_subscriber.sub)
            except Exception as e:
                self.get_logger().warning(f"Error destroying odom subscriber: {e}")
            self._odom_subscriber = None

        if self._pose_subscriber is not None:
            try:
                self.destroy_subscription(self._pose_subscriber.sub)
            except Exception as e:
                self.get_logger().warning(f"Error destroying pose subscriber: {e}")
            self._pose_subscriber = None

    def _handle_tf_static(self, msg: TFMessage) -> None:
        for transform in msg.transforms:
            self.tf_buffer.set_transform_static(transform, "default_authority")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ClusterPosesServiceNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    # Keep same rclpy workaround style as other scripts in this repo.
    try:
        while rclpy.ok():
            try:
                executor.spin_once()
            except KeyboardInterrupt:
                raise
            except rclpy._rclpy_pybind11.InvalidHandle as e:  # type: ignore
                node.get_logger().error(f"Invalid handle rclpy bug: {e}\nignoring...")
            except Exception as e:
                node.get_logger().error(f"Exception in main: {e}")
                raise
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"Unhandled exception in main: {e}")
    finally:
        executor.shutdown()
        node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
