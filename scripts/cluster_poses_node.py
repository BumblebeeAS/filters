#!/usr/bin/env python3
from operator import attrgetter
from typing import List, Optional

import numpy as np
import rclpy
import tf2_ros
from bb_filters.clustering.cluster import (
    get_average_pose,
    get_idxs_in_largest_cluster,
    get_position_tuple_from_pose,
)
from bb_perception_msgs.action import ClusterPosesAction
from geometry_msgs.msg import (
    PoseArray,
    PoseStamped,
    PoseWithCovarianceStamped,
    Quaternion,
    TransformStamped,
    Vector3,
)
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sklearn.cluster import HDBSCAN
from tf2_geometry_msgs import do_transform_pose
from tf2_msgs.msg import TFMessage


def transform_pose_to_odom(
    data: tuple[Odometry, PoseStamped],
    camera_to_odom_transform: tf2_ros.TransformStamped,
) -> PoseWithCovarianceStamped:
    """Transform a pose from camera frame to odom parent frame.

    Args:
        data: Tuple of (odom_msg, pose_msg) from synchronized callback
        camera_to_odom_transform: Static transform from camera to odom_child frame

    Returns:
        PoseWithCovarianceStamped in odom parent frame
    """
    odom_msg, pose_stamped_msg = data

    # Transform pose from camera frame to odom child frame
    pose_msg = pose_stamped_msg.pose
    transformed_pose = do_transform_pose(pose_msg, camera_to_odom_transform)

    # Create transform from odom child to odom parent using odometry
    odom_transform = tf2_ros.TransformStamped()
    odom_transform.transform.translation.x = odom_msg.pose.pose.position.x
    odom_transform.transform.translation.y = odom_msg.pose.pose.position.y
    odom_transform.transform.translation.z = odom_msg.pose.pose.position.z
    odom_transform.transform.rotation = odom_msg.pose.pose.orientation

    # Apply odom transform to get final pose in odom parent frame
    final_pose = do_transform_pose(transformed_pose, odom_transform)

    # Convert to PoseWithCovarianceStamped
    pose_with_cov = PoseWithCovarianceStamped()
    pose_with_cov.header.frame_id = odom_msg.header.frame_id
    pose_with_cov.header.stamp = odom_msg.header.stamp
    pose_with_cov.pose.pose = final_pose

    return pose_with_cov


class ClusterPosesNode(Node):
    def __init__(self):
        super().__init__("cluster_poses_node")

        self.tf_buffer = tf2_ros.Buffer(
            cache_time=rclpy.duration.Duration(seconds=10.0)
        )

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
        self._current_goal_handle = None
        self._synchronized_data: List[tuple] = []
        self._camera_to_odom_transform: Optional[tf2_ros.TransformStamped] = None
        self._odom_subscriber: Optional[Subscriber] = None
        self._pose_subscriber: Optional[Subscriber] = None
        self._time_synchronizer: Optional[ApproximateTimeSynchronizer] = None

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

    def goal_callback(self, goal_request):
        """Accept or reject a new goal."""
        self.get_logger().info("Received new goal request")
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Handle goal cancellation."""
        self.get_logger().info("Received cancel request")
        return CancelResponse.ACCEPT

    def synchronized_callback(self, odom_msg: Odometry, pose_msg: PoseStamped):
        """Callback for synchronized odom and pose messages."""
        self._synchronized_data.append((odom_msg, pose_msg))

    async def execute_callback(self, goal_handle):
        """Execute the clustering action."""
        self.get_logger().info("Executing goal...")
        self._current_goal_handle = goal_handle
        self._synchronized_data = []

        goal = goal_handle.request
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
            collection_duration = rclpy.duration.Duration(
                seconds=goal.collection_duration
            )

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
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=5.0),
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
                transform_pose_to_odom(data, self._camera_to_odom_transform)
                for data in self._synchronized_data
            ]

            self.get_logger().info("Trying to cluster...")
            # Cluster the transformed poses
            avg_pose, num_poses_in_cluster = self._cluster_poses(
                transformed_poses, goal
            )
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
            result.poses_in_cluster = num_poses_in_cluster

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
        self,
        transformed_poses: List[PoseWithCovarianceStamped],
        goal,
    ) -> tuple[Optional[PoseWithCovarianceStamped], int]:
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
            return None, 0

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
            [get_position_tuple_from_pose(pose) for pose in transformed_poses]
        )
        filtered_idxs = get_idxs_in_largest_cluster(hdbscan, positions)

        if len(filtered_idxs) == 0:
            self.get_logger().error("No clusters found")
            return None, 0

        # Get average pose from filtered poses
        filtered_poses = [transformed_poses[i] for i in filtered_idxs]
        avg_pose = get_average_pose(filtered_poses)

        return avg_pose, len(filtered_idxs)

    def _publish_results(
        self,
        avg_pose: PoseWithCovarianceStamped,
        transformed_poses: List[PoseWithCovarianceStamped],
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
        pose_array_msg.poses = [pose.pose.pose for pose in transformed_poses]
        self.pose_array_publisher.publish(pose_array_msg)

        # Publish clustered pose as a static transform
        transform_stamped = TransformStamped()
        transform_stamped.header = avg_pose.header
        transform_stamped.child_frame_id = clustered_child_frame_id
        t = attrgetter("x", "y", "z")(avg_pose.pose.pose.position)
        qx, qy, qz, qw = attrgetter("x", "y", "z", "w")(avg_pose.pose.pose.orientation)
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
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
