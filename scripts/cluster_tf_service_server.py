#!/usr/bin/env python3
import traceback

import numpy as np
import rclpy
import tf2_ros
from bb_filters.cluster import average_transforms, get_position_from_transform
from bb_filters.tf_lru_cache import TfLruCache

# from bb_perception_msgs.action import ClusterTf
from bb_perception_msgs.srv import ClusterTf
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.time import Time
from sklearn.cluster import HDBSCAN


class ClusterTfServiceServer(Node):
    """
    ROS2 Action Server that collects TF transforms over a duration,
    clusters them using HDBSCAN, and publishes the centroid of the
    largest cluster as a static TF transform.
    """

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
        self.timer = self.create_timer(0.01, self.timer_callback)

        # shared flag between the service callback and timer callback
        self.enabled = False

        self.get_logger().info("Cluster TFs service server initialized")

    def timer_callback(self):
        if not self.enabled:
            return

        for output_parent, input_child in zip(self.output_parents, self.input_children):
            try:
                tf = self.tf_buffer.lookup_transform(
                    target_frame=output_parent,
                    source_frame=input_child,
                    time=Time(),  # TODO check if a timeout is needed
                )
                self.get_logger().info(
                    "Transform found: "
                    f"{output_parent} -> {input_child} at {tf.header.stamp}"
                )
                self.caches[(output_parent, input_child)].add(tf)
            except Exception as e:
                self.get_logger().warn(f"Failed to lookup transform: {e}")
                self.get_logger().warn(f"Traceback: {traceback.format_exc()}")

    def create_finish_response(
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

            message = (
                f"Clustered transform from {output_parent} to {output_child} "
                f"with average position: {avg_translation.x:.3f}, {avg_translation.y:.3f}, {avg_translation.z:.3f}"
            )
            self.get_logger().info(message)
            worked = True

            self.static_tf_broadcaster.sendTransform(clustered_transform)

        response.is_cluster_success = worked
        response.message = (
            message if worked else "No clusters found, no transforms created."
        )

    def cluster_srv_callback(
        self, request: ClusterTf.Request, response: ClusterTf.Response
    ):
        if not request.enabled:  # not enabled do the clustering
            self.create_finish_response(response)
            self.enabled = False

            return response

        self.enabled = request.enabled
        self.output_parents = request.output_parent_frame_ids.copy()
        self.input_children = request.input_child_frame_ids.copy()
        self.output_children = request.output_child_frame_ids.copy()
        self.tf_lookup_interval = request.tf_lookup_interval
        self.use_cache = request.use_cache
        self.min_cluster_size = request.min_cluster_size
        self.min_samples = request.min_samples
        self.cache_size = request.cache_size

        for output_parent, input_child in zip(self.output_parents, self.input_children):
            if (output_parent, input_child) in self.caches:
                continue
            self.get_logger().info(
                f"Using cache of size {self.cache_size} for {output_parent} to {input_child}"
            )
            self.caches[(output_parent, input_child)] = TfLruCache(
                size=self.PERSISTENT_CACHE_SIZE, logger=self.get_logger()
            )

        response.message = (
            "Cluster TFs service server is enabled, collecting transforms."
        )
        response.is_cluster_success = True
        return response

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
