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
from bb_perception_msgs.srv import ClusterTfSrv
from geometry_msgs.msg import PoseArray, Quaternion, TransformStamped, Vector3
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sklearn.cluster import HDBSCAN


class ClusterMultiServiceServer(Node):
    def __init__(self):
        super().__init__("cluster_tf_multi_service_server")

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

        self.service_server = self.create_service(
            ClusterTfSrv,
            "/auv4/cluster_tfs_multi_srv",
            self.cluster_srv_callback,
        )
        self.timer = self.create_timer(0.01, self.collect_tfs)

        self.pose_array_publisher_all = self.create_publisher(
            PoseArray, "output_pose_array_all_srv", 10
        )

        self.cache = TfLruCache(size=self.cache_size, logger=self.get_logger())
        self.enabled = False
        self.get_logger().info("Multi cluster service server initialized")

    def collect_tfs(self):
        if not self.enabled:
            return

        for input_child in self.tf_list_in:
            try:
                tf = self.tf_buffer.lookup_transform(
                    target_frame=self.output_parent_frame,
                    source_frame=input_child,
                    time=Time(),
                )
                self.cache.add(tf)
            except Exception as e:
                self.get_logger().warn(
                    f"Failed to lookup transform for {input_child}: {e}"
                )
                self.get_logger().warn(f"Traceback: {traceback.format_exc()}")

    def cluster_srv_callback(
        self, request: ClusterTfSrv.Request, response: ClusterTfSrv.Response
    ):
        if not request.enabled:
            self.enabled = False
            response.is_enabled = False
            response.is_cluster_success = False
            tfs, latest_time = self.cache.get_all()

            if len(tfs) == 0:
                self.get_logger().warn("No transforms available in cache.")
                return response

            top_n_indices = self.cluster_transforms(
                tfs=tfs,
                min_cluster_size=request.min_cluster_size,
                min_samples=request.min_samples,
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
                return response

            ordered_tfs = self.order_transforms(tfs, top_n_indices, curr_pos)

            if len(ordered_tfs) == 0:
                self.get_logger().warn("No valid transforms found.")
                return response

            # faithfully pub all tfs that we have collected + clustered + ordered dont do any post processing
            for (avg_translation, avg_rotation), tf_out in zip(
                ordered_tfs, request.output_child_frame_ids
            ):
                self.publish_transform(
                    translation=avg_translation,
                    rotation=avg_rotation,
                    latest_time=latest_time,
                    output_child=tf_out,
                )

            if not self.persistent:
                self.cache = TfLruCache(size=self.cache_size, logger=self.get_logger())

            return response

        self.enabled = True
        self.tf_list_in = request.input_child_frame_ids.copy()
        self.tf_list_out = request.output_child_frame_ids.copy()
        self.min_cluster_size = request.min_cluster_size
        self.min_samples = request.min_samples
        self.persistent = request.persistent
        self.num_tfs = len(self.tf_list_in)

        # for debugging purposes
        self.pub_list = [
            self.create_publisher(PoseArray, f"/auv4/{out}_cluster_multi_srv", 10)
            for out in self.tf_list_out
        ]

        return response

    def cluster_transforms(self, tfs, min_cluster_size, min_samples) -> list[list[int]]:
        min_num_poses = max(min_cluster_size, min_samples)

        if self.cache.get_count() < min_num_poses:
            self.get_logger().warn(
                f"Not enough transforms collected to perform clustering. "
                f"Collected: {self.cache.get_count()}, Required: {min_num_poses}."
            )
            return []

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
            return []

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
    node = ClusterMultiServiceServer()
    try:
        rclpy.spin(node, executor=MultiThreadedExecutor())
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
