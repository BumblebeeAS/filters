#!/usr/bin/env python3
import traceback

import numpy as np
import rclpy
import tf2_ros
from bb_filters.cluster import (
    average_transforms,
    get_position_from_transform,
    tf_to_pose_stamped,
)
from bb_filters.tf_lru_cache import TfLruCache
from bb_perception_msgs.action import ClusterTfAction
from geometry_msgs.msg import PoseArray, Quaternion, TransformStamped, Vector3
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sklearn.cluster import HDBSCAN


class ClusterTfMultiActionServer(Node):
    def __init__(self):
        super().__init__("cluster_tf_multi_action_server")

        self.declare_parameter(name="output_parent_frame", value="world_ned")
        self.output_parent_frame = (
            self.get_parameter("output_parent_frame").get_parameter_value().string_value
        )

        self.declare_parameter(name="base_link_frame", value="auv4/base_link_ned")
        self.base_link_frame = (
            self.get_parameter("base_link_frame").get_parameter_value().string_value
        )

        self.declare_parameter(name="cache_size", value=10000)
        self.cache_size = (
            self.get_parameter("cache_size").get_parameter_value().integer_value
        )

        # TF components
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=False
        )

        self._action_server = ActionServer(
            self,
            ClusterTfAction,
            "/auv4/cluster_tf_multi",
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

        self.pose_array_publisher_all = self.create_publisher(
            PoseArray, "/auv4/cluster_multi/poses", 10
        )

        self.caches = dict()

        # the below two vars will be created every time action is called
        self.num_tfs = 0
        self.pub_list = []

        self.get_logger().info("Multi cluster action server initialized")

    def goal_callback(self, goal_request):
        self.get_logger().info("Received goal request, accepting")
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info("Received cancel request, accepting")
        return CancelResponse.ACCEPT

    def handle_accepted(self, goal_handle):
        self.get_logger().info("Goal accepted, executing callback")
        goal_handle.execute()

    def _collect_transform(self, input_child, cache, counts):
        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame=self.output_parent_frame,
                source_frame=input_child,
                time=Time(),
            )
            if cache.add(tf):
                counts[input_child] += 1
        except Exception as e:
            self.get_logger().warn(f"Failed to lookup transform for {input_child}: {e}")
            self.get_logger().warn(f"Traceback: {traceback.format_exc()}")

    def collect_transforms(
        self,
        goal_handle,
        cache,
        clustering_duration,
        tf_list_in,
        tf_lookup_interval=0.05,
    ):
        rate = self.create_rate(1.0 / tf_lookup_interval)
        counts = {tf: 0 for tf in tf_list_in}  # purely for debugging

        self.get_logger().info(f"Collecting TFs for {clustering_duration} seconds")

        start_time = self.get_clock().now()
        end_time = start_time + Duration(seconds=clustering_duration)
        while self.get_clock().now() < end_time:
            if goal_handle.is_cancel_requested:
                self.get_logger().info("Goal canceled during TF collection.")
                return "CANCEL"

            for input_child in tf_list_in:
                self._collect_transform(input_child, cache, counts)

            try:
                rate.sleep()
            except:
                self.get_logger().info("Interrupted during TF collection.")
                return "ABORT"

        [
            self.get_logger().info(
                f"Collected {counts[input_child]} tfs from {self.output_parent_frame} to {input_child}"
            )
            for input_child in tf_list_in
        ]

        return "SUCCESS"

    def cluster_transforms(self, tfs, min_cluster_size, min_samples) -> list[list[int]]:
        min_num_poses = max(min_cluster_size, min_samples)

        if len(tfs) < min_num_poses:
            self.get_logger().warn(
                f"Not enough transforms collected to perform clustering. "
                f"Collected: {len(tfs)}, Required: {min_num_poses}."
            )
            return [[] for _ in range(self.num_tfs)]

        positions = np.array([get_position_from_transform(tf) for tf in tfs])

        hdbscan = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=0.5,
            allow_single_cluster=True,
            store_centers="centroid",
        )

        labels = hdbscan.fit_predict(positions)
        valid_mask = labels >= 0

        if not np.any(valid_mask):
            return [[] for _ in range(self.num_tfs)]

        valid_labels = labels[valid_mask]

        max_label = valid_labels.max()
        counts = np.bincount(valid_labels, minlength=max_label + 1)

        # negate counts to get descending order TODO: check if need stable sort
        top_n_indices = np.argsort(-counts)[: self.num_tfs]

        result = [
            np.where(labels == cluster_id)[0].tolist()
            for cluster_id in top_n_indices
            if counts[cluster_id] > 0
        ]

        while len(result) < self.num_tfs:
            result.append([])

        return result

    def _pub_debug_poses(self, tfs, publisher):
        """Publish debug poses for a specific input child."""
        pose_stamped_msgs = map(tf_to_pose_stamped, tfs)
        pose_array_msg = PoseArray()
        pose_array_msg.header = tfs[-1].header
        pose_array_msg.poses = [
            pose_stamped_msg.pose for pose_stamped_msg in pose_stamped_msgs
        ]
        publisher.publish(pose_array_msg)

    def order_transforms(self, tfs, top_indices, curr_pos):
        ordered_tfs = []

        # Publish the array of poses for debugging
        self._pub_debug_poses(tfs, self.pose_array_publisher_all)

        for i, indices in enumerate(top_indices):
            if len(indices) == 0:
                break  # top_indices should have been sorted by argsort previously

            filtered_tfs = [tfs[i] for i in indices]

            self._pub_debug_poses(
                filtered_tfs, self.pub_list[i]
            )  # pub_list created everytime action is called

            avg_translation, avg_rotation = average_transforms(filtered_tfs)
            ordered_tfs.append((avg_translation, avg_rotation))

        ordered_tfs.sort(key=self._comparator(curr_pos))
        return ordered_tfs

    def publish_transform(self, translation, rotation, latest_time, output_child):
        clustered_transform = TransformStamped()
        clustered_transform.header.stamp = latest_time.to_msg()
        clustered_transform.header.frame_id = self.output_parent_frame
        clustered_transform.child_frame_id = output_child
        clustered_transform.transform.translation = translation
        clustered_transform.transform.rotation = rotation

        self.get_logger().info(
            f"Clustered transform from {self.output_parent_frame} to {output_child} "
            f"with average position: {translation.x:.3f}, {translation.y:.3f}, {translation.z:.3f}"
        )

        self.static_tf_broadcaster.sendTransform(clustered_transform)

    async def execute_callback(self, goal_handle):
        goal: ClusterTfAction.Goal = goal_handle.request
        clustering_duration = goal.clustering_duration  # seconds
        min_cluster_size = goal.min_cluster_size
        min_samples = goal.min_samples
        tf_list_in = goal.input_child_frame_ids
        tf_list_out = goal.output_child_frame_ids
        tf_lookup_interval = goal.tf_lookup_interval
        use_cache = goal.use_cache
        persistent = goal.persistent

        self.num_tfs = len(tf_list_in)
        self.pub_list = [
            self.create_publisher(PoseArray, f"/auv4/{out}/poses", 10)
            for out in tf_list_out
        ]

        feedback_msg = ClusterTfAction.Feedback()
        result = ClusterTfAction.Result()

        cache_key = (tuple(sorted(tf_list_in)), tuple(sorted(tf_list_out)))

        if cache_key not in self.caches:
            if persistent:
                self.caches[cache_key] = TfLruCache(
                    self.cache_size, logger=self.get_logger()
                )
            else:
                cache_size = (
                    goal.cache_size
                    if use_cache
                    else int(clustering_duration / tf_lookup_interval) + 10
                )
                self.caches[cache_key] = TfLruCache(
                    size=cache_size, logger=self.get_logger()
                )

        cache = self.caches[cache_key]

        collection_result = self.collect_transforms(
            goal_handle=goal_handle,
            cache=cache,
            clustering_duration=clustering_duration,
            tf_list_in=tf_list_in,
            tf_lookup_interval=tf_lookup_interval,
        )

        if collection_result == "ABORT":
            goal_handle.abort()
            return result
        elif collection_result == "CANCEL":
            goal_handle.canceled()
            return result

        tfs, latest_time = cache.get_all()

        top_n_indices = self.cluster_transforms(
            tfs=tfs,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )

        try:
            curr_pos = self.tf_buffer.lookup_transform(
                target_frame=self.output_parent_frame,
                source_frame=self.base_link_frame,
                time=Time(),
            ).transform.translation
        except Exception as e:
            self.get_logger().error(
                f"Failed to lookup transform from world to base link frame: {e}"
                f"Traceback: {traceback.format_exc()}"
            )
            goal_handle.abort()
            return result

        ordered_tfs = self.order_transforms(tfs, top_n_indices, curr_pos)

        if len(ordered_tfs) == 0:
            # for loop below will not run in this case
            self.get_logger().warn("No valid transforms found.")

        self.get_logger().info(
            f"min_cluster_size: {min_cluster_size}, min_samples: {min_samples}"
        )

        # faithfully pub all tfs that we have collected + clustered + ordered dont do any post processing
        for (avg_translation, avg_rotation), tf_out in zip(ordered_tfs, tf_list_out):
            self.publish_transform(
                translation=avg_translation,
                rotation=avg_rotation,
                latest_time=latest_time,
                output_child=tf_out,
            )

        if not persistent:
            del self.caches[cache_key]

        goal_handle.succeed()
        return result

    @staticmethod
    def _comparator(origin: Vector3):
        def d(transform: tuple[Vector3, Quaternion, str]):
            translation = transform[0]
            return (
                ((translation.x - origin.x) ** 2)
                + ((translation.y - origin.y) ** 2)
                + ((translation.z - origin.z) ** 2)
            )

        return d


def main(args=None):
    rclpy.init(args=args)
    node = ClusterTfMultiActionServer()
    try:
        rclpy.spin(node, executor=MultiThreadedExecutor())
    except KeyboardInterrupt:
        pass
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
