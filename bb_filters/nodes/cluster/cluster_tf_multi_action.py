#!/usr/bin/env python3

import traceback

import rclpy
from action_msgs.msg import GoalStatus
from bb_perception_msgs.action import ClusterTfAction
from geometry_msgs.msg import PoseArray, Quaternion, TransformStamped, Vector3
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time
from sklearn.cluster import HDBSCAN

from bb_filters.utils.cluster.cluster import (
    average_transforms,
    euclidean_metric,
    get_top_k_clusters,
    tf_to_pose,
)
from bb_filters.utils.cluster.cluster_tf_action_base import ClusterTfActionBase
from bb_filters.utils.tf_lru_cache import TfCacheDict, TfLruCache


class ClusterTfMultiActionS(ClusterTfActionBase):
    def __init__(self, node_name: str = "cluster_tf_multi_action"):
        super().__init__(node_name)

        self.declare_parameter(name="output_parent_frame", value="world_ned")
        self.output_parent_frame = (
            self.get_parameter("output_parent_frame").get_parameter_value().string_value
        )

        self.declare_parameter(name="base_link_frame", value="auv4/base_link_ned")
        self.base_link_frame = (
            self.get_parameter("base_link_frame").get_parameter_value().string_value
        )

    def setup_caches(self, goal: ClusterTfAction.Goal) -> TfCacheDict:
        clustering_duration = goal.clustering_duration  # seconds
        tf_list_in = goal.input_child_frame_ids
        tf_list_out = goal.output_child_frame_ids
        tf_lookup_interval = goal.tf_lookup_interval
        use_cache = goal.use_cache
        persistent = goal.persistent

        cache_key = (tuple(sorted(tf_list_in)), tuple(sorted(tf_list_out)))

        if persistent:
            if cache_key not in self.caches:
                self.caches[cache_key] = TfLruCache(
                    self.cache_size, logger=self.get_logger()
                )
            return {cache_key: self.caches[cache_key]}
        else:
            cache_size = (
                goal.cache_size
                if use_cache
                else int(clustering_duration / tf_lookup_interval) + 10
            )
            return {cache_key: TfLruCache(size=cache_size, logger=self.get_logger())}

    def collect_once(
        self,
        goal: ClusterTfAction.Goal,
        caches: TfCacheDict,
        start_time: Time,
    ) -> None:
        tf_list_in = goal.input_child_frame_ids

        cache = next(iter(caches.values()))

        for input_child in tf_list_in:
            try:
                tf = self.tf_buffer.lookup_transform(
                    target_frame=self.output_parent_frame,
                    source_frame=input_child,
                    time=Time(),
                )
                success, is_duplicated, is_old = cache.add(tf, start_time)
            except Exception as e:
                self.get_logger().warn(
                    f"Failed to lookup transform for {input_child}: {e}"
                )
                self.get_logger().warn(f"Traceback: {traceback.format_exc()}")

    def process_transforms(
        self,
        goal_handle,
        goal: ClusterTfAction.Goal,
        filled_caches: TfCacheDict,
    ) -> tuple[list[TransformStamped], dict[str, PoseArray], int]:
        # TODO: Split into cluster and order methods

        min_cluster_size = goal.min_cluster_size
        min_samples = goal.min_samples
        tf_list_out = goal.output_child_frame_ids

        output_tfs = []
        debug_poses = dict()

        cache = next(iter(filled_caches.values()))
        tfs, latest_time = cache.get_all()

        min_num_poses = max(min_cluster_size, min_samples)
        if len(tfs) < min_num_poses:
            self.get_logger().warn(
                f"Not enough transforms collected to perform clustering. "
                f"Collected: {len(tfs)}, Required: {min_num_poses}."
            )
            return [], dict(), GoalStatus.STATUS_SUCCEEDED

        hdbscan = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=0.5,
            allow_single_cluster=True,
            store_centers="centroid",
        )

        clusters = get_top_k_clusters(hdbscan, tfs, k=len(tf_list_out))
        ordered_centroids = []

        for tf_out, cluster in zip(tf_list_out, clusters):
            # Calculate centroid
            ordered_centroids.append(average_transforms(cluster))

            # Generate debug pose array
            pose_array = PoseArray()
            pose_array.header = cluster[0].header
            pose_array.poses = [tf_to_pose(tf) for tf in cluster]
            debug_poses[tf_out] = pose_array

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
            return [], dict(), GoalStatus.STATUS_ABORTED

        def comparator(v: tuple[Vector3, Quaternion]):
            return euclidean_metric((curr_pos, Quaternion()), v)

        ordered_centroids.sort(key=comparator)

        for (translation, rotation), tf_out in zip(ordered_centroids, tf_list_out):
            clustered_transform = TransformStamped()
            clustered_transform.header.stamp = latest_time.to_msg()
            clustered_transform.header.frame_id = self.output_parent_frame
            clustered_transform.child_frame_id = tf_out
            clustered_transform.transform.translation = translation
            clustered_transform.transform.rotation = rotation

            self.get_logger().info(
                f"Clustered transform from {self.output_parent_frame} to {tf_out} "
                f"with average position: {translation.x:.3f}, {translation.y:.3f}, {translation.z:.3f}"
            )

            output_tfs.append(clustered_transform)

        return output_tfs, debug_poses, GoalStatus.STATUS_SUCCEEDED


def main(args=None):
    rclpy.init(args=args)
    node = ClusterTfMultiActionS()
    try:
        rclpy.spin(node, executor=MultiThreadedExecutor())
    except KeyboardInterrupt:
        pass
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
