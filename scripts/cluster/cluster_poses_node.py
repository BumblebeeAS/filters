#!/usr/bin/env python3
from operator import attrgetter

import numpy as np
import rclpy
import tf2_ros
from bb_filters.clustering.cluster import ClusterResult, get_largest_cluster
from bb_filters.clustering.pose import get_average_pose
from bb_perception_msgs.action import ClusterPosesAction
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
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
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

CONFIDENCE_KEY_BY_METRIC = {
    0: "mean_probability",
    1: "inlier_ratio",
    2: "position_std",
}


def seconds_to_duration(seconds: float) -> Duration:
    """Convert float seconds to rclpy Duration."""
    sec_int, sec_frac = divmod(seconds, 1)
    return Duration(seconds=int(sec_int), nanoseconds=int(round(sec_frac * 1e9)))


class ClusterPosesNode(Node):
    def __init__(self):
        super().__init__("cluster_poses_node")

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10))

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

        # Callback group for action server
        self.action_callback_group = ReentrantCallbackGroup()

        # Action server
        self._action_server = ActionServer(
            self,
            ClusterPosesAction,
            "cluster_poses",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.action_callback_group,
        )

        # State for current goal execution
        self._current_goal_handle: ServerGoalHandle | None = None
        self._synchronized_data: list[tuple] = []
        self._camera_to_odom_transform: tf2_ros.TransformStamped | None = None
        self._odom_subscriber: Subscriber | None = None
        self._pose_subscriber: Subscriber | None = None
        self._time_synchronizer: ApproximateTimeSynchronizer | None = None

        # Declare parameters
        output_pose_array_topic = (
            self.declare_parameter(
                "output_pose_array_topic",
                "clustered_poses",
            )
            .get_parameter_value()
            .string_value
        )
        self.sync_queue_size = (
            self.declare_parameter("sync_queue_size", 100)
            .get_parameter_value()
            .integer_value
        )
        self.feedback_rate = (
            self.declare_parameter("feedback_rate_hz", 10)
            .get_parameter_value()
            .integer_value
        )

        # Publishers
        self.pose_array_publisher = self.create_publisher(
            PoseArray, output_pose_array_topic, 10
        )
        self._static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        self.get_logger().info("Cluster Poses Action Server initialized")

    def goal_callback(self, goal_request: ClusterPosesAction.Goal) -> GoalResponse:
        """Accept or reject a new goal."""
        self.get_logger().info("Received new goal request")
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle: ServerGoalHandle) -> CancelResponse:
        """Handle goal cancellation."""
        self.get_logger().info("Received cancel request")
        return CancelResponse.ACCEPT

    def synchronized_callback(self, odom_msg: Odometry, pose_msg: PoseStamped):
        """Callback for synchronized odom and pose messages."""
        self._synchronized_data.append((odom_msg, pose_msg))

    async def execute_callback(
        self, goal_handle: ServerGoalHandle
    ) -> ClusterPosesAction.Result:
        """Execute the clustering action."""
        self.get_logger().info("Executing goal...")
        self._current_goal_handle = goal_handle
        self._synchronized_data = []

        goal: ClusterPosesAction.Goal = goal_handle.request
        feedback_msg = ClusterPosesAction.Feedback()

        try:
            # Set up subscribers with time synchronization
            feedback_msg.current_status = "Setting up subscribers"
            feedback_msg.collection_progress = 0.0
            feedback_msg.poses_collected_so_far = 0
            goal_handle.publish_feedback(feedback_msg)

            # Set up subscribers with time synchronization
            self._odom_subscriber = Subscriber(
                self, Odometry, goal.odom_topic, qos_profile=qos_profile_sensor_data
            )
            self._pose_subscriber = Subscriber(
                self,
                PoseStamped,
                goal.pose_stamped_topic,
                qos_profile=qos_profile_sensor_data,
            )

            self._time_synchronizer = ApproximateTimeSynchronizer(
                [self._odom_subscriber, self._pose_subscriber],
                queue_size=self.sync_queue_size,
                slop=goal.sync_tolerance,
            )
            self._time_synchronizer.registerCallback(self.synchronized_callback)

            # Collect synchronized messages for the specified duration
            feedback_msg.current_status = "Collecting synchronized messages"
            goal_handle.publish_feedback(feedback_msg)

            collection_start_time = self.get_clock().now()
            collection_duration = seconds_to_duration(goal.collection_duration)

            rate = self.create_rate(self.feedback_rate)
            while rclpy.ok():
                elapsed_time = self.get_clock().now() - collection_start_time

                if elapsed_time >= collection_duration:
                    break

                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    self.get_logger().info("Goal canceled")
                    self._cleanup_subscribers()
                    return ClusterPosesAction.Result()

                # Update feedback
                feedback_msg.collection_progress = min(
                    elapsed_time.nanoseconds / collection_duration.nanoseconds, 1.0
                )
                feedback_msg.poses_collected_so_far = len(self._synchronized_data)
                goal_handle.publish_feedback(feedback_msg)

                rate.sleep()

            # Check if we have enough data
            total_collected = len(self._synchronized_data)
            self.get_logger().info(
                f"Collected {total_collected} synchronized pose pairs"
            )

            if total_collected < goal.min_poses:
                self.get_logger().error(
                    f"Not enough synchronized poses collected. Got {total_collected}, need {goal.min_poses}"
                )
                goal_handle.abort()
                self._cleanup_subscribers()
                return ClusterPosesAction.Result()

            # Get odom child frame and camera frame from first message
            odom_child_frame = self._synchronized_data[0][0].child_frame_id
            camera_frame_id = self._synchronized_data[0][1].header.frame_id

            # Lookup static transform from camera to odom child frame
            feedback_msg.current_status = "Looking up static transform"
            goal_handle.publish_feedback(feedback_msg)

            try:
                self._camera_to_odom_transform = self.tf_buffer.lookup_transform(
                    odom_child_frame,
                    camera_frame_id,
                    Time(),
                    timeout=Duration(seconds=5),
                )
                self.get_logger().info(
                    f"Found transform from {camera_frame_id} to {odom_child_frame}"
                )
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as e:
                self.get_logger().error(f"Failed to lookup transform: {e}")
                goal_handle.abort()
                self._cleanup_subscribers()
                return ClusterPosesAction.Result()

            # Transform poses to odom frame and cluster
            feedback_msg.current_status = "Transforming and clustering poses"
            feedback_msg.collection_progress = 1.0
            goal_handle.publish_feedback(feedback_msg)

            transformed_poses = [
                transform_pose_to_odom(
                    odom_msg, pose_msg, self._camera_to_odom_transform
                )
                for odom_msg, pose_msg in self._synchronized_data
            ]

            self.get_logger().info("Trying to cluster...")
            # Cluster the transformed poses
            avg_pose, cluster_result = self._cluster_poses(transformed_poses, goal)
            if avg_pose is None:
                goal_handle.abort()
                self._cleanup_subscribers()
                return ClusterPosesAction.Result()

            self.get_logger().info("Finished clustering...")
            # Use the latest timestamp
            avg_pose.header = transformed_poses[-1].header

            # Publish results
            self._publish_results(
                avg_pose, transformed_poses, goal.clustered_child_frame_id
            )

            # Return result
            result = ClusterPosesAction.Result()
            result.clustered_pose = avg_pose
            result.total_poses_collected = total_collected
            result.poses_in_cluster = len(cluster_result.idxs)
            result.mean_probability = cluster_result.mean_probability
            result.inlier_ratio = cluster_result.inlier_ratio
            result.position_std = cluster_result.position_std
            result.primary_confidence = getattr(
                cluster_result,
                CONFIDENCE_KEY_BY_METRIC.get(int(goal.primary_confidence_metric), ""),
                0.0,
            )

            self.get_logger().info(
                f"Clustering complete: {result.poses_in_cluster}/{result.total_poses_collected} poses in cluster"
            )

            goal_handle.succeed()
            self._cleanup_subscribers()
            return result

        except Exception as e:
            self.get_logger().error(f"Error during execution: {e}")
            goal_handle.abort()
            self._cleanup_subscribers()
            return ClusterPosesAction.Result()

    def _cluster_poses(
        self, transformed_poses: list[PoseStamped], goal: ClusterPosesAction.Goal
    ) -> tuple[PoseStamped | None, ClusterResult]:
        """Cluster transformed poses using HDBSCAN and return the average pose.

        Args:
            transformed_poses: List of transformed poses to cluster
            goal: Goal request containing clustering parameters

        Returns:
            Tuple of (average_pose, num_poses_in_cluster) or (None, 0) if clustering fails
        """
        # Check if there are enough poses for clustering
        if len(transformed_poses) < max(goal.min_cluster_size, goal.min_samples):
            self.get_logger().error("Not enough poses for clustering")
            return None, ClusterResult.empty()

        # Create HDBSCAN clustering instance
        hdbscan = HDBSCAN(
            min_cluster_size=goal.min_cluster_size,
            min_samples=goal.min_samples,
            cluster_selection_epsilon=goal.cluster_selection_epsilon,
            allow_single_cluster=True,
            store_centers="centroid",
        )

        # Extract positions and cluster
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

        # Get average pose from filtered poses
        filtered_pose_msgs = [transformed_poses[i].pose for i in cluster_result.idxs]
        avg_pose = get_average_pose(filtered_pose_msgs)
        avg_pose_stamped = PoseStamped()
        avg_pose_stamped.pose = avg_pose
        avg_pose_stamped.header = transformed_poses[cluster_result.idxs[0]].header

        return avg_pose_stamped, cluster_result

    def _publish_results(
        self,
        avg_pose: PoseStamped,
        transformed_poses: list[PoseStamped],
        clustered_child_frame_id: str,
    ) -> None:
        """Publish clustered results as pose array and static transform.

        Args:
            avg_pose: The averaged clustered pose
            transformed_poses: List of all transformed poses for publishing as array
                    clustered_child_frame_id: Frame ID for the clustered transform
        """
        # Publish pose array of all transformed poses
        pose_array_msg = PoseArray()
        pose_array_msg.header = avg_pose.header
        pose_array_msg.poses = [pose.pose for pose in transformed_poses]
        self.pose_array_publisher.publish(pose_array_msg)

        # Publish clustered pose as a static transform
        transform_stamped = TransformStamped()
        transform_stamped.header = avg_pose.header
        transform_stamped.child_frame_id = clustered_child_frame_id
        t = attrgetter("x", "y", "z")(avg_pose.pose.position)
        qx, qy, qz, qw = attrgetter("x", "y", "z", "w")(avg_pose.pose.orientation)
        transform_stamped.transform.translation = Vector3(x=t[0], y=t[1], z=t[2])
        transform_stamped.transform.rotation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        self._static_tf_broadcaster.sendTransform(transform_stamped)

    def _cleanup_subscribers(self):
        """Clean up subscribers after goal completion."""
        if self._time_synchronizer is not None:
            self._time_synchronizer = None

        # try-excepts are needed as unused subscriptions may already be destroyed
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
        """Store static transforms without subscribing to dynamic TF."""
        for transform in msg.transforms:
            self.tf_buffer.set_transform_static(transform, "default_authority")


def main(args=None):
    rclpy.init(args=args)
    node = ClusterPosesNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    # https://github.com/CMU-cabot/cabot-navigation/commit/18fe9330c6b0c02c1e52344b7ed6df32bfaf01e7
    try:
        while rclpy.ok():
            try:
                executor.spin_once()
            except KeyboardInterrupt:
                raise
            except rclpy._rclpy_pybind11.InvalidHandle as e:  # type: ignore
                node.get_logger().error(f"Invalid handle rclpy bug: {e}\nignoring...")
            except Exception as e:
                # https://github.com/ros2/rclpy/issues/1206
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
