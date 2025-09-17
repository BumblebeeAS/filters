#!/usr/bin/env python3
import traceback

import numpy as np
import rclpy
import tf2_ros
from bb_perception_msgs.action import ClusterTfAction
from geometry_msgs.msg import PoseArray, TransformStamped
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.time import Time
from sklearn.cluster import HDBSCAN
from std_srvs.srv import Trigger

from bb_filters.clustering.cluster import (
    average_transforms,
    get_position_from_transform,
    tf_to_pose,
)
from bb_filters.utils.tf_lru_cache import TfLruCache


class ClusterTfActionServer(Node):
    """
    ROS2 Action Server that collects TF transforms over a duration,
    clusters them using HDBSCAN, and publishes the centroid of the
    largest cluster as a static TF transform.
    """

    PERSISTENT_CACHE_SIZE = 5000

    def __init__(self):
        super().__init__("cluster_tfs_action_server")

        # TF components
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=False
        )

        self._action_server = ActionServer(
            self,
            ClusterTfAction,
            "/auv4/cluster_tf",
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

        self.reset_cache_srv = self.create_service(
            srv_type=Trigger,
            srv_name="/auv4/cluster_tf/reset_caches",
            callback=self.reset_callback,
        )

        self.caches: dict[tuple[str, str], TfLruCache] = dict()

        self.get_logger().info("Cluster TFs action server initialized")

    def reset_callback(self, request: Trigger.Request, response: Trigger.Response):
        response = Trigger.Response()
        self.caches = dict()
        response.success = True
        response.message = "Caches resetted"
        self.get_logger().info(response.message)

        return response

    def goal_callback(self, goal_request):
        self.get_logger().info("Received goal request, accepting")
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info("Received cancel request, accepting")
        return CancelResponse.ACCEPT

    def handle_accepted(self, goal_handle):
        self.get_logger().info("Goal accepted, executing callback")
        goal_handle.execute()

    @staticmethod
    def _get_idxs_in_largest_cluster(
        hdbscan: HDBSCAN, positions: np.ndarray
    ) -> np.ndarray:
        """Returns an array of indices belonging to the largest, non-noise cluster."""
        hdbscan.fit(positions)

        labels = np.array(hdbscan.labels_)
        non_noise_labels = labels[labels >= 0]

        if len(non_noise_labels) == 0:
            return np.array([])

        unique_labels, unique_label_counts = np.unique(
            non_noise_labels, return_counts=True
        )
        largest_cluster_label = unique_labels[np.argmax(unique_label_counts)]
        largest_cluster_idxs = np.where(labels == largest_cluster_label)[0]

        return largest_cluster_idxs

    async def execute_callback(self, goal_handle):
        goal: ClusterTfAction.Goal = goal_handle.request
        input_children = goal.input_child_frame_ids
        output_parents = goal.output_parent_frame_ids
        output_children = goal.output_child_frame_ids
        clustering_duration = goal.clustering_duration  # seconds
        lookup_interval = goal.tf_lookup_interval  # seconds
        use_cache = goal.use_cache
        min_cluster_size = goal.min_cluster_size
        min_samples = goal.min_samples

        feedback_msg = ClusterTfAction.Feedback()
        result = ClusterTfAction.Result()

        for output_parent, input_child in zip(output_parents, input_children):
            cache_key = (output_parent, input_child)

            if goal.persistent:
                if cache_key not in self.caches:
                    self.caches[cache_key] = TfLruCache(
                        self.PERSISTENT_CACHE_SIZE, logger=self.get_logger()
                    )
            else:
                cache_size = (
                    goal.cache_size
                    if use_cache
                    else int(clustering_duration / lookup_interval) + 10
                )
                self.caches[cache_key] = TfLruCache(
                    size=cache_size, logger=self.get_logger()
                )

        num_old_tfs = 0
        num_duplicated_tfs = 0
        start_time = self.get_clock().now()
        end_time = start_time + Duration(seconds=clustering_duration)

        self.get_logger().info(f"Collecting TFs for {clustering_duration} seconds")

        rate = self.create_rate(1.0 / lookup_interval)

        while self.get_clock().now() < end_time:
            if goal_handle.is_cancel_requested:
                self.get_logger().info("Goal canceled during TF collection.")

                self.destroy_rate(rate=rate)  # destroy the rate
                goal_handle.canceled()
                return result

            for output_parent, input_child in zip(output_parents, input_children):
                try:
                    tf = self.tf_buffer.lookup_transform(
                        target_frame=output_parent,
                        source_frame=input_child,
                        time=Time(),  # TODO check if a timeout is needed
                    )

                    succeeded, is_duplicated, is_old = self.caches[
                        (output_parent, input_child)
                    ].add(tf, start_time)

                    num_duplicated_tfs += int(is_duplicated)
                    num_old_tfs += int(is_old)

                except Exception as e:
                    self.get_logger().warn(f"Failed to lookup transform: {e}")
                    self.get_logger().warn(f"Traceback: {traceback.format_exc()}")

            try:
                rate.sleep()
            except:
                self.get_logger().info("Interrupted during TF collection.")

                self.destroy_rate(rate=rate)  # destroy the rate
                goal_handle.canceled()
                return result

        self.get_logger().warn(f"{num_old_tfs} old TFs collected")
        self.get_logger().warn(f"{num_duplicated_tfs} duplicate TFs collected")

        min_num_poses = max(min_cluster_size, min_samples)
        worked = False
        pub_list = []

        for output_parent, input_child, output_child in zip(
            output_parents, input_children, output_children
        ):
            cache = self.caches[(output_parent, input_child)]

            self.get_logger().info(
                f"Collected {cache.get_count()} valid transforms from {output_parent} to {input_child}"
            )

            self.get_logger().warn(
                f"Clustering transforms: {[((tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z) if tf is not None else 0) for tf in cache.cache]}",
            )

            # Perform clustering
            if cache.is_empty():
                self.get_logger().warn(
                    f"No transforms collected for {output_parent} to {input_child}, cannot perform clustering."
                )
                continue

            if cache.get_count() < min_num_poses:
                self.get_logger().warn(
                    f"Not enough transforms collected to perform clustering. "
                    f"Collected: {cache.get_count()}, Required: {min_num_poses}."
                )
                continue

            tfs, latest_time = cache.get_all()

            pub = self.create_publisher(PoseArray, f"/auv4/{output_child}/poses", 10)
            pub_list.append(pub)

            self._pub_debug_poses(tfs, pub)

            # Extract positions directly from transforms for clustering
            positions = np.array([get_position_from_transform(tf) for tf in tfs])

            hdbscan = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=0.2,
                allow_single_cluster=True,
                store_centers="centroid",
            )

            filtered_idxs = self._get_idxs_in_largest_cluster(hdbscan, positions)
            if len(filtered_idxs) == 0:
                self.get_logger().warn(
                    "No clusters found, cannot create clustered transform."
                )
                continue

            # Calculate average transform from largest cluster
            filtered_tfs = [tfs[i] for i in filtered_idxs]
            avg_translation, avg_rotation = average_transforms(filtered_tfs)

            # Create the clustered transform
            clustered_transform = TransformStamped()
            clustered_transform.header.stamp = latest_time.to_msg()
            clustered_transform.header.frame_id = output_parent
            clustered_transform.child_frame_id = output_child
            clustered_transform.transform.translation = avg_translation
            clustered_transform.transform.rotation = avg_rotation

            self.get_logger().info(
                f"Clustered transform from {output_parent} to {output_child} "
                f"with average position: {avg_translation.x:.3f}, {avg_translation.y:.3f}, {avg_translation.z:.3f}"
            )
            worked = True

            self.static_tf_broadcaster.sendTransform(clustered_transform)

        ##### CLEANUP #####
        if not goal.persistent:
            for output_parent, input_child in zip(output_parents, input_children):
                del self.caches[(output_parent, input_child)]

        if not self.destroy_rate(rate=rate):
            self.get_logger().warn(f"Failed to destroy rate: {rate}")

        # for pub in pub_list:
        #     if not self.destroy_publisher(publisher=pub):
        #         self.get_logger().warn(f"Failed to destroy publisher: {pub}")
        ##################

        if not worked:
            self.get_logger().error(
                "No clusters possible for all provided input transforms"
            )
            goal_handle.abort()
            return result

        goal_handle.succeed()
        return result

    def _pub_debug_poses(self, tfs: list[TransformStamped], publisher: Publisher):
        """Publish debug poses for a specific input child."""
        pose_stamped_msgs = map(tf_to_pose, tfs)
        pose_array_msg = PoseArray()
        if len(tfs) == 0:
            return
        pose_array_msg.header = tfs[-1].header
        pose_array_msg.poses = list(pose_stamped_msgs)
        publisher.publish(pose_array_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ClusterTfActionServer()
    try:
        rclpy.spin(node, executor=MultiThreadedExecutor())
    except KeyboardInterrupt:
        pass
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
