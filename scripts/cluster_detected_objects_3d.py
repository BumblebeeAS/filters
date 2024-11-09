#!/usr/bin/env python3

from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from bb_perception_msgs.msg import DetectedObject3DArray
from geometry_msgs.msg import PoseStamped
from message_filters import Subscriber
from ml_detector.schema_validator import get_config, load_schema
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sklearn.cluster import HDBSCAN

from std_srvs.srv import Trigger

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
        self.queue_size = (
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
        self.class_queues = defaultdict(lambda: deque(maxlen=self.queue_size))
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

        # trigger to flush queue
        self.flush_queue_srv = self.create_service(
            Trigger, "/uav2/clusters/flush_queue", self.flush_queue_cb)

    def detection_callback(self, detections: DetectedObject3DArray):
        """Adds respective detections into their respective queues"""
        if not detections.objects:
            # self.get_logger().error("No detections to cluster")
            return
        for obj in detections.objects:
            position = np.array(
                [
                    obj.hypothesis.kinematics.pose_with_covariance.pose.position.x,
                    obj.hypothesis.kinematics.pose_with_covariance.pose.position.y,
                ]
            )

            # self.get_logger().info(
            #     f"Enqueue detection for {self.id_to_name[obj.hypothesis.class_id]} with {position}"
            # )
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
                self.get_logger().info(f"Publishing single pose.... {single_pose.pose.position.x}, {single_pose.pose.position.y}")
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

    def flush_queue_cb(self, req: Trigger.Request, 
            resp: Trigger.Response) -> Trigger.Response:
        self.class_queues = defaultdict(lambda: deque(maxlen=self.queue_size))
        self.get_logger().info(f"Class queue {self.class_queues}....")
        resp.success = True
        return resp

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
