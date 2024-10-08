#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from bb_perception_msgs.msg import DetectedObject3DArray, DetectedObject3D
import numpy as np
from collections import deque
from ament_index_python.packages import get_package_share_directory
from pathlib import Path

from ml_detector.schema_validator import get_config, load_schema


class HypothesisManager:
    def __init__(self, name, max_distance, max_num_hypothesis):
        self.name = name
        self.max_distance = max_distance
        self.max_num_hypothesis = max_num_hypothesis

        # Dictionary of hypotheses, where each key is a unique hypothesis ID and each value is:
        # (centroid_position, identities_dict, latest_position)
        self.hypotheses = {}

        # A queue to keep track of the latest `n` updated hypotheses
        self.latest_hypotheses = deque()

    def update_hypothesis(self, new_position, new_identity, det):
        """Update the existing hypotheses or create a new one."""

        closest_hypothesis = None
        closest_distance = float("inf")

        # Search for the closest hypothesis within the allowed distance range
        for hyp_id, (centroid, identities, latest_pos, _) in self.hypotheses.items():
            distance = np.linalg.norm(latest_pos - new_position)
            if distance <= self.max_distance and distance < closest_distance:
                closest_hypothesis = hyp_id
                closest_distance = distance

        if closest_hypothesis is not None:
            # Update the existing hypothesis
            centroid, identities, _, det = self.hypotheses[closest_hypothesis]

            # Update centroid (simple average of old centroid and new position)
            updated_centroid = np.array(centroid) * 0.3 + np.array(new_position) * 0.7

            # Update identities count
            identities[new_identity] = identities.get(new_identity, 0) + 1

            # Update the hypothesis with new centroid, identities, and latest position
            self.hypotheses[closest_hypothesis] = (updated_centroid, identities, new_position, det)

            # Move the updated hypothesis to the front of the queue
            self._mark_as_latest(closest_hypothesis)

        else:
            # Create a new hypothesis
            new_hypothesis_id = len(self.hypotheses)
            identities = {new_identity: 1}
            self.hypotheses[new_hypothesis_id] = (new_position, identities, new_position, det)

            # Add the new hypothesis to the latest queue
            self._mark_as_latest(new_hypothesis_id)

        # Ensure we don't exceed the max number of hypotheses
        self._prune_hypotheses()

    def _mark_as_latest(self, hypothesis_id):
        """Mark a hypothesis as recently updated, keeping the queue size within limits."""
        if hypothesis_id in self.latest_hypotheses:
            self.latest_hypotheses.remove(hypothesis_id)
        self.latest_hypotheses.appendleft(hypothesis_id)

    def _prune_hypotheses(self):
        """Remove old hypotheses if the number exceeds `max_num_hypothesis`."""
        while len(self.latest_hypotheses) > self.max_num_hypothesis:
            oldest_hypothesis = self.latest_hypotheses.pop()
            del self.hypotheses[oldest_hypothesis]

    def get_all_hypotheses(self):
        """Get all current hypotheses for output purposes."""
        return self.hypotheses


class ObstaclesManagementServer(Node):

    def __init__(self):
        super().__init__("obstacle_management")

        objects_schema_path = (
            Path(get_package_share_directory("ml_detector"))
            / "configs"
            / "objects_schema.json"
        )
        self.objects_schema = load_schema(objects_schema_path)
        self.declare_parameter("objects_config", "robotx.yaml")
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
        self.name_to_id = {v: k for k, v in self.id_to_name.items()}

        # Parameters: Predefined obstacles and constraints
        self.obstacle_types = {
            "placard": {
                "count": 3,
                "max_distance": 5.0,
                "sub_identities": [
                    "placard_symbol_red",
                    "placard_symbol_green",
                    "placard_symbol_blue",
                ],
            },
            "gate": {
                "count": 2,
                "max_distance": 5.0,
                "sub_identities": ["gate_left", "gate_middle", "gate_right"],
            },
            "buoy": {
                "count": 6,
                "max_distance": 3.0,
                "sub_identities": [
                    "white_cylinder",
                    "red_cylinder",
                    "green_cylinder",
                    "black_cylinder",
                ],
            },
            "sphere": {
                "count": 3,
                "max_distance": 5.0,
                "sub_identities": ["red_sphere", "green_sphere", "blue_sphere"],
            },
            "light_tower": {
                "count": 1,
                "max_distance": 10.0,
                "sub_identities": [
                    "light_tower_panel",
                    "light_tower_panel_red",
                    "light_tower_panel_green",
                    "light_tower_panel_blue",
                    "light_tower_panel_black",
                ],
            },
        }
        self.obstacles_full_map = {}
        for obstacle, config in self.obstacle_types.items():
            for obj in config["sub_identities"]:
                self.obstacles_full_map[obj] = obstacle

        # Hypothesis manager for obstacle types
        self.hypothesis_managers = {
            obstacle_name: HypothesisManager(
                name=obstacle_name,
                max_distance=config["max_distance"],  # can be parameterized per obstacle type
                max_num_hypothesis=config["count"] * 2
            )
            for obstacle_name, config in self.obstacle_types.items()
        }

        # Subscribers and Publishers
        self.detections_sub = self.create_subscription(
            DetectedObject3DArray,
            # "/asv4/vision/detections_2d/projected",
            "/asv4/vision/lidar_small_objects/dets_3d/labelled",
            self.detections_callback,
            10,
        )
        self.detections_pub = self.create_publisher(
            DetectedObject3DArray, "/asv4/robotx/filtered_detections", 10
        )

        self.get_logger().info("Obstacle Management Server with Hypothesis Tracking started.")

    def detections_callback(self, msg):
        # Process each detected object
        for detection in msg.objects:
            obstacle_type = detection.hypothesis.class_id
            obstacle_name = self.id_to_name[obstacle_type]
            if obstacle_name in self.obstacles_full_map:
                # Extract position
                pos = np.array([
                    detection.hypothesis.kinematics.pose_with_covariance.pose.position.x,
                    detection.hypothesis.kinematics.pose_with_covariance.pose.position.y,
                    detection.hypothesis.kinematics.pose_with_covariance.pose.position.z,
                ])
                identity = obstacle_name
                # Update the hypothesis manager
                self.hypothesis_managers[self.obstacles_full_map[identity]].update_hypothesis(pos, identity, detection)

        # Publish filtered detections based on current hypotheses
        filtered_msg = DetectedObject3DArray()
        for obstacle_name, hypothesis_manager in self.hypothesis_managers.items():
            for hyp_id, (centroid, identities, _, det) in hypothesis_manager.get_all_hypotheses().items():
                # Create a DetectedObject3D for each hypothesis
                detection = self.create_detection_msg(centroid, identities, det)
                filtered_msg.objects.append(detection)
        filtered_msg.header = msg.header
        self.detections_pub.publish(filtered_msg)

    def create_detection_msg(self, centroid_position, identities, det):
        """Create a DetectedObject3D message based on the hypothesis's centroid and identity."""
        detection = det

        # Use the most common identity for the class_id
        most_common_identity = max(identities, key=identities.get)
        detection.hypothesis.class_id = self.name_to_id.get(most_common_identity, 0)

        # Set the centroid position

        detection.hypothesis.kinematics.pose_with_covariance.pose.position.x = centroid_position[0]
        detection.hypothesis.kinematics.pose_with_covariance.pose.position.y = centroid_position[1]
        detection.hypothesis.kinematics.pose_with_covariance.pose.position.z = centroid_position[2]
        return detection


def main(args=None):
    rclpy.init(args=args)
    obstacle_management_server = ObstaclesManagementServer()
    rclpy.spin(obstacle_management_server)
    obstacle_management_server.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
