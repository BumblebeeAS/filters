#!/usr/bin/env python3

from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from bb_perception_msgs.msg import DetectedObject3DArray
from bb_uav_msgs.srv import LatLonConverter
from geographic_msgs.msg import GeoPointStamped
from geometry_msgs.msg import PoseStamped
from message_filters import Subscriber
from ml_detector.schema_validator import get_config, load_schema
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sklearn.cluster import HDBSCAN


class ClusterDetectedObject3D(Node):
    def __init__(self):
        super().__init__("cluster_detected_object_3d_node")
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.declare_parameter("objects_config", "drone.yaml")
        objects_schema_path = (
            Path(get_package_share_directory("ml_detector"))
            / "configs"
            / "objects_schema.json"
        )
        self.objects_schema = load_schema(objects_schema_path)
        self.objects_config = get_config(
            Path(get_package_share_directory("ml_detector"))
            / "configs"
            / "objects"
            / self.get_parameter("objects_config").get_parameter_value().string_value,
            self.objects_schema,
        )
        self.id_to_name = {
            obj["label"]: obj["name"] for obj in self.objects_config["objects"]
        }
        self.declare_parameter("pose_frame", "odom_ned")
        self.pose_frame = (
            self.get_parameter("pose_frame").get_parameter_value().string_value
        )
        self.declare_parameter(
            "detected_objects_3d_topic", "/uav2/bottom_cam/projected_3d"
        )
        self.detection_topic = (
            self.get_parameter("detected_objects_3d_topic")
            .get_parameter_value()
            .string_value
        )
        self.declare_parameter("cluster_interval", 2.0)
        cluster_interval = (
            self.get_parameter("cluster_interval").get_parameter_value().double_value
        )
        self.declare_parameter("queue_size", 10)
        queue_size = (
            self.get_parameter("queue_size").get_parameter_value().integer_value
        )
        self.declare_parameter("min_cluster_size", 2)
        min_cluster_size = (
            self.get_parameter("min_cluster_size").get_parameter_value().integer_value
        )
        self.declare_parameter("min_samples", 1)
        min_samples = (
            self.get_parameter("min_samples").get_parameter_value().integer_value
        )
        self.declare_parameter("estimate_tolerance", 8.0)
        self.estimate_tolerance = (
            self.get_parameter("estimate_tolerance").get_parameter_value().double_value
        )
        self.declare_parameter("DBCV_threshold", 0.8)
        self.DBCV_threshold = (
            self.get_parameter("DBCV_threshold").get_parameter_value().double_value
        )
        self.detections_subscriber = Subscriber(
            self, DetectedObject3DArray, self.detection_topic, qos_profile=qos
        )
        self.detections_subscriber.registerCallback(self.detection_callback)
        # Publish multiple topics for each detection
        self.cluster_pose_publishers = {}
        # Create dict for a queue of every class
        self.class_queues = defaultdict(lambda: deque(maxlen=queue_size))
        self.hdb = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=0.0,
            allow_single_cluster=True,
            store_centers="centroid",
        )
        self.cluster_timer = self.create_timer(
            cluster_interval, self.cluster_timer_callback
        )

        # For dropping detections which are very far from the estimated locations
        self.estimate_n = None
        self.estimate_r = None
        self.estimate_r_sub = self.create_subscription(
            GeoPointStamped,
            "/uav2/search_report/estimate/robot_r",
            self.estimate_r_cb,
            10,
        )
        self.estimate_n_sub = self.create_subscription(
            GeoPointStamped,
            "/uav2/search_report/estimate/robot_n",
            self.estimate_n_cb,
            10,
        )

        self.home_latlon = None
        self.home_latlon_sub = self.create_subscription(
            GeoPointStamped, "/uav2/home_lat_lon", self.home_latlon_cb, 10
        )

        self.converter_client = self.create_client(
            LatLonConverter, "/uav2/lat_lon_converter_srv"
        )

    def detection_callback(self, detections: DetectedObject3DArray):
        """Adds respective detections into their respective queues"""
        if not detections.objects:
            self.get_logger().error("No detections to cluster")
            return
        for obj in detections.objects:
            position = np.array(
                [
                    obj.hypothesis.kinematics.pose_with_covariance.pose.position.x,
                    obj.hypothesis.kinematics.pose_with_covariance.pose.position.y,
                ]
            )
            # Do not enqueue if too far from estimate
            if not self.detection_pred(position, obj.hypothesis.class_id):
                self.get_logger().info(f"dropping detection too far away: {position}")
                continue

            self.get_logger().info(
                f"Enqueue detection for {self.id_to_name[obj.hypothesis.class_id]} with {position}"
            )
            # Add publisher for each class
            if obj.hypothesis.class_id not in self.cluster_pose_publishers:
                self.cluster_pose_publishers[obj.hypothesis.class_id] = (
                    self.create_publisher(
                        PoseStamped,
                        f"{self.detection_topic}/clustered/{self.id_to_name[obj.hypothesis.class_id]}",
                        10,
                    )
                )
            self.class_queues[obj.hypothesis.class_id].append(position)

    def merge_cluster(self, positions) -> PoseStamped:
        """Average out the positions in the cluster"""
        average_pose = PoseStamped()
        average_pose.header.frame_id = self.pose_frame
        average_pose.header.stamp = self.get_clock().now().to_msg()
        average_position = [0.0, 0.0]
        for position in positions:
            average_position[0] += position[0]
            average_position[1] += position[1]
        average_pose.pose.position.x = average_position[0] / len(positions)
        average_pose.pose.position.y = average_position[1] / len(positions)
        average_pose.pose.position.z = 0.0
        return average_pose

    def cluster_timer_callback(self):
        """Cluster detections for each object in queue"""
        for class_id, position_queue in self.class_queues.items():
            if not position_queue:
                self.get_logger().warn(
                    f"No detections to cluster for class {self.id_to_name[class_id]}"
                )
                continue
            # if only one detection, no need for clustering
            if len(position_queue) == 1:
                single_pose = PoseStamped()
                single_pose.header.frame_id = self.pose_frame
                single_pose.header.stamp = self.get_clock().now().to_msg()
                single_pose.pose.position.x = position_queue[0][0]
                single_pose.pose.position.y = position_queue[0][1]
                single_pose.pose.position.z = 0.0
                self.cluster_pose_publishers[class_id].publish(single_pose)
                continue

            # Returns a new list where each index in position_queue is replaced by the cluster label
            clusterer = self.hdb.fit(position_queue)
            labels = clusterer.labels_
            # Create a dict for the clusters
            clusters = defaultdict(list)
            for i, label in enumerate(labels):
                clusters[label].append(position_queue[i])
            # Find the biggest cluster
            largest_cluster_size = 0
            largest_cluster_label = 0
            for label, cluster in clusters.items():
                # Remove noise
                if label == -1:
                    continue
                if len(cluster) > largest_cluster_size:
                    largest_cluster_size = len(cluster)
                    largest_cluster_label = label
            largest_cluster_positions = clusters[largest_cluster_label]
            if len(largest_cluster_positions) == 0:
                self.get_logger().info(f"Cluster has no positions")
                continue
            clustered_average_pose = self.merge_cluster(largest_cluster_positions)
            self.get_logger().info(
                f"Publishing clustered pose with {self.id_to_name[class_id]} with {clustered_average_pose.pose.position}"
            )
            self.cluster_pose_publishers[class_id].publish(clustered_average_pose)

    # Methods for dropping detections far from estimates
    def estimate_r_cb(self, msg: GeoPointStamped) -> None:
        self.estimate_r = msg

    def estimate_n_cb(self, msg: GeoPointStamped) -> None:
        self.estimate_n = msg

    def home_latlon_cb(self, msg: GeoPointStamped) -> None:
        self.home_latlon = msg

    def detection_pred(self, position: list[float], class_id: int) -> bool:
        """Returns true if position (of detection) is near the estimates.

        Always returns true for anything that isn't robot_r or robot_n (ids 1 and 2).

        Depends on the following topics for the estimate:
            - /uav2/search_report/estimate/robot_r
            - /uav2/search_report/estimate/robot_n
            - /uav2/home_lat_lon

        Estimate to position conversion is done through LatLonCoverter.

        Tolerance is metres in euclidean distance. The z value is ignored.
        """
        # No home lat lon, should not happen, but defensive check
        if self.home_latlon is None:
            self.get_logger().info("home_latlon is none")
            return True

        # Not R or N marker
        if class_id != 1 and class_id != 2:
            self.get_logger().info("Neither R or N")
            return True

        # No estimates yet, just accept all
        if class_id == 1 and self.estimate_n is None:
            self.get_logger().info("No estimate for N")
            return True
        if class_id == 2 and self.estimate_r is None:
            self.get_logger().info("No estimate for R")
            return True

        if (
            self.estimate_r.position.latitude == 0.0
            or self.estimate_r.position.longitude == 0.0
        ):
            self.get_logger().info("Null land")
            return True

        # Convert estimate to position
        estimate_latlon = LatLonConverter.Request()
        estimate_latlon.input_converter = self.home_latlon
        estimate_latlon.lat_lon = (None, self.estimate_r, self.estimate_n)[class_id]
        estimate_position_future = self.converter_client.call_async(estimate_latlon)
        rclpy.spin_until_future_complete(self, estimate_position_future)
        estimate_position: LatLonConverter.Response = estimate_position_future.result()

        # Get euclidean distance
        dist = (
            (estimate_position.local_pose.pose.position.x - position[0]) ** 2
            + (estimate_position.local_pose.pose.position.y - position[1]) ** 2
        ) ** 0.5

        return dist <= self.estimate_tolerance


def main(args=None):
    rclpy.init(args=args)
    node = ClusterDetectedObject3D()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
