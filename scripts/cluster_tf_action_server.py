#!/usr/bin/env python3
import time
import traceback
from operator import attrgetter

import numpy as np
import rclpy
import rclpy.logging
import tf2_geometry_msgs
import tf2_ros
from bb_perception_msgs.action import ClusterTf
from geometry_msgs.msg import (
    Quaternion,
    TransformStamped,
    Vector3,
)
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sklearn.cluster import HDBSCAN

from bb_filters.cluster import (
    get_average_pose,
    get_idxs_in_largest_cluster,
    get_position_tuple_from_pose,
    tf_to_pose_with_covariance_stamped,
)


class ClusterTfActionServer(Node):
    def __init__(self):
        super().__init__("cluster_tfs_action_server")

        # TF components
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        self._action_server = ActionServer(
            self, ClusterTf, "/auv4/cluster_tf", self.execute_callback
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info("Cluster TFs Action Server initialized")

    def goal_callback(self, goal_request):
        """Accept or reject a client request to begin an action."""
        self.get_logger().info("Received goal request, accepting")
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject a client request to cancel an action."""
        self.get_logger().info("Received cancel request, accepting")
        return CancelResponse.ACCEPT

    def handle_accepted(self, goal_handle):
        """Handle the accepted goal."""
        self.get_logger().info("Goal accepted, executing callback")
        goal_handle.execute()

    async def execute_callback(self, goal_handle):
        # Extract goal parameters
        goal = goal_handle.request
        input_parent = goal.input_parent_frame_id
        input_child = goal.input_child_frame_id
        output_parent = goal.output_parent_frame_id
        output_child = goal.output_child_frame_id
        clustering_duration = goal.clustering_duration
        lookup_interval = goal.tf_lookup_interval
        use_cache = goal.use_cache
        min_cluster_size = goal.min_cluster_size
        min_samples = goal.min_samples

        # Initialize feedback
        feedback_msg = ClusterTf.Feedback()
        feedback_msg.current_status = "Starting TF collection"
        feedback_msg.collection_progress = 0.0
        feedback_msg.transforms_collected_so_far = 0
        goal_handle.publish_feedback(feedback_msg)

        result = ClusterTf.Result()

        if use_cache:
            cache_size = goal.cache_size
        else:
            cache_size = int(clustering_duration / lookup_interval) + 10
        cache = TfLruCache(
            size=cache_size,
        )

        start_time = self.get_clock().now()
        end_time = start_time + Duration(seconds=clustering_duration)

        self.get_logger().info(
            f"Collecting TF from {input_parent} to {input_child} for {clustering_duration} seconds"
        )

        lookup_hz = max(1, int(1 / lookup_interval))
        count = 0

        while self.get_clock().now() < end_time:
            rclpy.spin_once(self)
            count += 1

            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return result

            # Try to lookup transform
            try:
                tf = self.tf_buffer.lookup_transform(
                    source_frame=input_child,
                    target_frame=input_parent,
                    time=Time(),
                )
                cache.add(tf)
            except Exception as e:
                self.get_logger().error(f"Failed to lookup transform: {e}")
                self.get_logger().error(f"Traceback: {traceback.format_exc()}")
                goal_handle.abort()
                return result

            if count % lookup_hz == 0:
                # Update feedback every second
                feedback_msg.current_status = "Collecting transforms"
                feedback_msg.collection_progress = (
                    (self.get_clock().now() - start_time).nanoseconds
                    * 10e9
                    / clustering_duration
                )
                feedback_msg.transforms_collected_so_far = cache.get_count()
                goal_handle.publish_feedback(feedback_msg)

            time.sleep(lookup_interval)

        # Update feedback for clustering phase
        feedback_msg.current_status = (
            "Finished collecting transforms, starting clustering"
        )
        feedback_msg.collection_progress = 1.0
        feedback_msg.transforms_collected_so_far = cache.get_count()
        goal_handle.publish_feedback(feedback_msg)

        self.get_logger().info(
            f"Collected {cache.get_count()} transforms from {input_parent} to {input_child}"
        )

        # Perform clustering
        if cache.is_empty():
            self.get_logger().error(
                "No transforms collected, cannot perform clustering."
            )
            goal_handle.abort()
            return result

        min_num_poses = max(min_cluster_size, min_samples)
        if cache.get_count() < min_num_poses:
            self.get_logger().error(
                f"Not enough transforms collected to perform clustering. Collected: {cache.get_count()}, Required: {min_num_poses}."
            )
            goal_handle.abort()
            return result

        tfs, latest_time = cache.get_all()
        pose_msgs = [tf_to_pose_with_covariance_stamped(tf) for tf in tfs]

        # Perform clustering
        positions = np.array([get_position_tuple_from_pose(pose) for pose in pose_msgs])

        hdbscan = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=0.05,
            allow_single_cluster=True,
            store_centers="centroid",
        )

        filtered_idxs = get_idxs_in_largest_cluster(hdbscan, positions)
        if len(filtered_idxs) == 0:
            self.get_logger().error(
                "No clusters found, cannot create clustered transform."
            )
            goal_handle.abort()
            return result

        # Calculate average pose from largest cluster
        filtered_poses = [pose_msgs[i] for i in filtered_idxs]
        avg_pose = get_average_pose(filtered_poses, self.get_logger())

        # Create the clustered transform
        clustered_transform = TransformStamped()
        clustered_transform.header.stamp = latest_time.to_msg()
        clustered_transform.header.frame_id = output_parent
        clustered_transform.child_frame_id = output_child

        # If we need to transform from input parent to output parent frame, do it here
        if input_parent != output_parent:
            try:
                # Get transform from input parent to output parent
                parent_transform = self.tf_buffer.lookup_transform(
                    target_frame=output_parent,
                    source_frame=input_parent,
                    time=Time(),
                )

                avg_pose = tf2_geometry_msgs.do_transform_pose_with_covariance_stamped(
                    avg_pose, parent_transform
                )

            except Exception as e:
                self.get_logger().error(
                    f"Could not get transform from {input_parent} to {output_parent}: {e}"
                )
                goal_handle.abort()
                return result

        # Extract position and orientation
        t = attrgetter("x", "y", "z")(avg_pose.pose.pose.position)
        qx, qy, qz, qw = attrgetter("x", "y", "z", "w")(avg_pose.pose.pose.orientation)
        clustered_transform.transform.translation = Vector3(x=t[0], y=t[1], z=t[2])
        clustered_transform.transform.rotation = Quaternion(x=qx, y=qy, z=qz, w=qw)

        # Publish as static transform
        self.static_tf_broadcaster.sendTransform(clustered_transform)

        self.get_logger().info(
            f"Clustered transform from {output_parent} to {output_child} with average position: \n"
            f"{avg_pose.pose.pose.position.x}, {avg_pose.pose.pose.position.y}, {avg_pose.pose.pose.position.z}"
        )

        # Final feedback
        feedback_msg.current_status = "Clustering complete, static transform published"
        goal_handle.publish_feedback(feedback_msg)

        # Return success result
        result.clustered_transform = clustered_transform
        result.total_transforms_collected = len(pose_msgs)
        result.transforms_in_cluster = len(filtered_poses)

        goal_handle.succeed()

        return result


class TfLruCache:
    def __init__(self, size: int):
        self.size = size

        # idx is the current insertion index (the open spot in the circular buffer)
        self.idx = 0

        self.cache = [None for i in range(self.size)]
        self.logger = rclpy.logging.get_logger("cluster_tf_node")

        self.oldest_time = Time()
        self.latest_time = Time()

        self.is_empty_flag = True
        self.count = 0

    @property
    def is_full(self) -> bool:
        return self.count >= self.size

    def _get(self, idx: int) -> TransformStamped:
        return self.cache[idx % self.size]

    def _set(self, tf: TransformStamped):
        self.cache[self.idx] = tf
        self.latest_time = Time.from_msg(tf.header.stamp)
        self.idx = (self.idx + 1) % self.size
        self.count += 1

    def add(self, tf: TransformStamped):
        if self.is_empty_flag:
            self.oldest_time = Time.from_msg(tf.header.stamp)
            self.is_empty_flag = False
            self._set(tf)
            return True

        prev_tf = self._get(self.idx - 1)
        # self.logger.info(f"current: {id(tf)}")
        # self.logger.info(f"prev: {id(prev_tf)}")

        if Time.from_msg(tf.header.stamp) == Time.from_msg(prev_tf.header.stamp):
            self.logger.warn(
                f"Skipping TF with timestamp {tf.header.stamp} as it is the same as the previous one."
            )
            return False

        if self.is_full:
            self.oldest_time = Time.from_msg(self._get(self.idx + 1).header.stamp)

        self._set(tf)
        return True

    def get_oldest_time(self) -> Time:
        return self.oldest_time

    def get_latest_time(self) -> Time:
        return self.latest_time

    def is_empty(self) -> bool:
        return self.is_empty_flag

    def get_all(self) -> tuple[list[TransformStamped], Time]:
        return [
            self.cache[i] for i in range(self.size) if self.cache[i] is not None
        ], self.get_latest_time()

    def get_count(self) -> int:
        return self.count


def main(args=None):
    rclpy.init(args=args)
    node = ClusterTfActionServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
