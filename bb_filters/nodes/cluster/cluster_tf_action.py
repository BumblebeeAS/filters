#!/usr/bin/env python3

import traceback

import rclpy
from action_msgs.msg import GoalStatus
from bb_perception_msgs.action import ClusterTfAction
from geometry_msgs.msg import PoseArray, TransformStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time
from sklearn.cluster import HDBSCAN

from bb_filters.utils.cluster.cluster import (
    average_transforms,
    get_top_k_clusters,
    tf_to_pose,
)
from bb_filters.utils.cluster.cluster_tf_action_base import ClusterTfActionBase
from bb_filters.utils.tf_lru_cache import TfCacheDict, TfLruCache


class ClusterTfActionS(ClusterTfActionBase):
    def __init__(self, node_name: str = "cluster_tf_action"):
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
        input_children = goal.input_child_frame_ids
        output_parents = goal.output_parent_frame_ids
        clustering_duration = goal.clustering_duration  # seconds
        lookup_interval = goal.tf_lookup_interval  # seconds
        use_cache = goal.use_cache

        caches: TfCacheDict = dict()

        for output_parent, input_child in zip(output_parents, input_children):
            cache_key = (output_parent, input_child)

            if goal.persistent:
                if cache_key not in self.caches:
                    self.caches[cache_key] = TfLruCache(
                        self.cache_size, logger=self.get_logger()
                    )
                caches[cache_key] = self.caches[cache_key]
            else:
                cache_size = (
                    goal.cache_size
                    if use_cache
                    else int(clustering_duration / lookup_interval) + 10
                )
                caches[cache_key] = TfLruCache(
                    size=cache_size, logger=self.get_logger()
                )

        return caches

    def collect_once(
        self,
        goal: ClusterTfAction.Goal,
        caches: TfCacheDict,
        start_time: Time,
    ) -> None:
        input_children = goal.input_child_frame_ids
        output_parents = goal.output_parent_frame_ids

        for output_parent, input_child in zip(output_parents, input_children):
            try:
                tf = self.tf_buffer.lookup_transform(
                    target_frame=output_parent,
                    source_frame=input_child,
                    time=Time(),
                )

                succeeded, is_duplicated, is_old = caches[
                    (output_parent, input_child)
                ].add(tf, start_time)

            except Exception as e:
                self.get_logger().warn(f"Failed to lookup transform: {e}")
                self.get_logger().warn(f"Traceback: {traceback.format_exc()}")

    def process_transforms(
        self,
        goal_handle,
        goal: ClusterTfAction.Goal,
        filled_caches: TfCacheDict,
    ) -> tuple[list[TransformStamped], dict[str, PoseArray], int]:
        input_children = goal.input_child_frame_ids
        output_parents = goal.output_parent_frame_ids
        output_children = goal.output_child_frame_ids
        min_cluster_size = goal.min_cluster_size
        min_samples = goal.min_samples

        output_tfs = []
        debug_poses = dict()

        is_at_least_one_success = False

        for output_parent, input_child, output_child in zip(
            output_parents, input_children, output_children
        ):
            cache = filled_caches[(output_parent, input_child)]
            tfs, latest_time = cache.get_all()

            min_num_poses = max(min_cluster_size, min_samples)
            if len(tfs) < min_num_poses:
                self.get_logger().warn(
                    f"Not enough transforms collected from {output_parent} to {input_child} to perform clustering. "
                    f"Collected: {len(tfs)}, Required: {min_num_poses}."
                )

            if len(tfs) > 0:
                pose_array = PoseArray()
                pose_array.header = tfs[0].header
                pose_array.poses = [tf_to_pose(tf) for tf in tfs]
                debug_poses[input_child] = pose_array

            hdbscan = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=0.2,
                allow_single_cluster=True,
                store_centers="centroid",
            )

            cluster = get_top_k_clusters(hdbscan, tfs, k=1)[0]
            avg_translation, avg_rotation = average_transforms(cluster)

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
            is_at_least_one_success = True

            output_tfs.append(clustered_transform)

        if not is_at_least_one_success:
            self.get_logger().error(
                "No clusters possible for all provided input transforms"
            )
            return [], dict(), GoalStatus.STATUS_ABORTED

        return output_tfs, debug_poses, GoalStatus.STATUS_SUCCEEDED


def main(args=None):
    rclpy.init(args=args)
    node = ClusterTfActionS()
    try:
        rclpy.spin(node, executor=MultiThreadedExecutor())
    except KeyboardInterrupt:
        pass
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
