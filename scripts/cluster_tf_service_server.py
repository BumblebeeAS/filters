#!/usr/bin/env python3
import traceback

import numpy as np
import rclpy
import tf2_ros
from bb_perception_msgs.srv import ClusterTf
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.time import Time
from sklearn.cluster import HDBSCAN

from bb_filters.cluster import (
    average_transforms,
    get_idxs_in_largest_cluster,
    get_position_from_transform,
)
from bb_filters.tf_lru_cache import TfLruCache


class ClusterTfServiceServer(Node):
    PERSISTENT_CACHE_SIZE = 10000

    def __init__(self):
        super().__init__("cluster_tfs_service_server")

        # TF components
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=False
        )

        self.service_server = self.create_service(
            ClusterTf,
            "/auv4/cluster_tfs_srv",
            self.cluster_srv_callback,
        )

        self.caches = dict()
        self.timer = self.create_timer(
            0.01, self.collect_tfs
        )  # 100 Hz, if our detections are faster than this then I give you $100

        # shared flag between the service callback and timer callback
        self.enabled = False

        self.get_logger().info("Cluster TFs service server initialized")

    def collect_tfs(self):
        if not self.enabled:
            return

        for output_parent, input_child in zip(self.output_parents, self.input_children):
            try:
                tf = self.tf_buffer.lookup_transform(
                    target_frame=output_parent,
                    source_frame=input_child,
                    time=Time(),
                )
                self.get_logger().info(
                    "Transform found: "
                    f"{output_parent} -> {input_child} at {tf.header.stamp}"
                )
                self.caches[(output_parent, input_child)].add(tf)
            except Exception as e:
                self.get_logger().warn(f"Failed to lookup transform: {e}")
                self.get_logger().warn(f"Traceback: {traceback.format_exc()}")

    def cluster_and_respond(
        self,
        response,
    ):
        min_num_poses = max(self.min_cluster_size, self.min_samples)
        worked = False

        for output_parent, input_child, output_child in zip(
            self.output_parents, self.input_children, self.output_children
        ):
            cache = self.caches[(output_parent, input_child)]

            self.get_logger().info(
                f"Collected {cache.get_count()} transforms from {output_parent} to {input_child}"
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

            # Extract positions directly from transforms for clustering
            positions = np.array([get_position_from_transform(tf) for tf in tfs])

            hdbscan = HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                cluster_selection_epsilon=0.2,
                allow_single_cluster=True,
                store_centers="centroid",
            )

            filtered_idxs = get_idxs_in_largest_cluster(hdbscan, positions)
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

            message = (
                f"Clustered transform from {output_parent} to {output_child} "
                f"with average position: {avg_translation.x:.3f}, {avg_translation.y:.3f}, {avg_translation.z:.3f}"
            )
            self.get_logger().info(message)
            worked = True

            self.static_tf_broadcaster.sendTransform(clustered_transform)

        response.is_enabled = False
        response.is_cluster_success = worked

    def cluster_srv_callback(
        self, request: ClusterTf.Request, response: ClusterTf.Response
    ):
        if not request.enabled:  # not enabled do the clustering
            self.cluster_and_respond(
                response
            )  # pass by reference, modifies the reference

            if not self.persistent:
                for output_parent, input_child in zip(
                    self.output_parents, self.input_children
                ):
                    del self.caches[(output_parent, input_child)]

            self.enabled = False
            return response

        self.enabled = request.enabled
        self.output_parents = request.output_parent_frame_ids.copy()
        self.input_children = request.input_child_frame_ids.copy()
        self.output_children = request.output_child_frame_ids.copy()
        self.min_cluster_size = request.min_cluster_size
        self.min_samples = request.min_samples
        self.persistent = request.persistent

        for output_parent, input_child in zip(self.output_parents, self.input_children):
            if (output_parent, input_child) in self.caches:
                continue
            self.caches[(output_parent, input_child)] = TfLruCache(
                size=self.PERSISTENT_CACHE_SIZE, logger=self.get_logger()
            )

        response.is_enabled = True
        response.is_cluster_success = False
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ClusterTfServiceServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
