#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from bb_perception_msgs.msg import DetectedObject3DArray
import numpy as np
from collections import deque
from ament_index_python.packages import get_package_share_directory
from pathlib import Path

from ml_detector.schema_validator import get_config, load_schema
from bb_filters.log import RclLogHandler
import logging
from operator import attrgetter
from transforms3d.euler import quat2euler

from pykalman import KalmanFilter
import numpy as np
from collections import deque
import logging
from copy import deepcopy
from geometry_msgs.msg import Point

np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
LOGGER = logging.getLogger("obstacles_management")


class HypothesisManager:
    def __init__(self, name, max_distance, max_num_hypothesis):
        self.name = name
        self.max_distance = max_distance
        self.max_num_hypothesis = max_num_hypothesis
        self.tid_buffer_size = 5

        # Dictionary of hypotheses
        self.hypotheses = {}

        # Queue for latest updated hypotheses
        self.latest_hypotheses = deque()

    def update_hypothesis(self, new_position, new_yaw, new_identity, det, tid):
        """Update the existing hypotheses or create a new one."""
        closest_hypothesis = None
        closest_distance = float("inf")

        # Search for the closest hypothesis within the allowed distance range
        for hyp_id, (
            positions,
            yaw_values,
            identities,
            tids,
            det,
        ) in self.hypotheses.items():
            # Calculate the distance
            distance = np.linalg.norm(positions[-1][:2] - new_position[:2])

            # Check if track ID matches
            if tid in tids:
                distance = 0

            if distance <= self.max_distance and distance < closest_distance:
                closest_hypothesis = hyp_id
                closest_distance = distance

        if closest_hypothesis is not None and closest_distance < self.max_distance:
            # Update the existing hypothesis
            positions, yaw_values, identities, tids, det = self.hypotheses[
                closest_hypothesis
            ]

            # Append the new position and yaw to the rolling window
            positions.append(new_position)
            yaw_values.append(new_yaw)

            # Limit the size of the rolling window
            if len(positions) > self.tid_buffer_size:
                positions.popleft()
            if len(yaw_values) > self.tid_buffer_size:
                yaw_values.popleft()

            # Calculate the median position and yaw
            median_position = np.median(np.array(positions), axis=0)
            median_yaw = np.median(np.array(yaw_values))

            # Update identities count
            identities[new_identity] = identities.get(new_identity, 0) + 1

            if tid not in tids:
                tids.append(tid)

            # Update the hypothesis with new values
            self.hypotheses[closest_hypothesis] = (
                positions,
                yaw_values,
                identities,
                tids,
                det,
            )
            LOGGER.info(
                "obstacle %s updated %s -> %s %s %s %s %s %s %s",
                closest_hypothesis,
                positions[-1],
                new_position,
                new_identity,
                self.name,
                tids,
                identities,
                closest_distance,
                tid,
            )

            # Move the updated hypothesis to the front of the queue
            self._mark_as_latest(closest_hypothesis)

        else:
            # Create a new hypothesis
            new_hypothesis_id = len(self.hypotheses)
            positions = deque([new_position], maxlen=self.tid_buffer_size)
            yaw_values = deque([new_yaw], maxlen=self.tid_buffer_size)
            identities = {new_identity: 1}
            tids = deque([tid], maxlen=self.tid_buffer_size)

            LOGGER.info(
                "obstacle %s created %s %s %s %s %s %s %s",
                new_hypothesis_id,
                new_position,
                new_identity,
                self.name,
                tids,
                identities,
                closest_distance,
                tid,
            )

            # Add the new hypothesis to the hypotheses dictionary
            self.hypotheses[new_hypothesis_id] = (
                positions,
                yaw_values,
                identities,
                tids,
                det,
            )

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
            LOGGER.info("obstacle %s removed", oldest_hypothesis)
            del self.hypotheses[oldest_hypothesis]

    def get_all_hypotheses(self):
        """Get all current hypotheses for output purposes."""
        return self.hypotheses


class ObstaclesManagementServer(Node):

    def __init__(self):
        super().__init__("obstacle_management")
        LOGGER.level = logging.INFO
        LOGGER.propagate = False
        LOGGER.addHandler(RclLogHandler(self.get_logger(), "obstacles_management"))

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
                "max_distance": 3.0,
                "sub_identities": [
                    "placard_symbol_red",
                    "placard_symbol_green",
                    "placard_symbol_blue",
                ],
            },
            "gate_left": {
                "count": 1,
                "max_distance": 5.0,
                "sub_identities": [],
            },
            "gate_middle": {
                "count": 1,
                "max_distance": 5.0,
                "sub_identities": [],
            },
            "gate_right": {
                "count": 1,
                "max_distance": 5.0,
                "sub_identities": [],
            },
            "buoy": {
                "count": 6,
                "max_distance": 3.5,
                "sub_identities": [
                    "white_cylinder",
                    "red_cylinder",
                    "green_cylinder",
                    "black_cylinder",
                ],
            },
            "sphere": {
                "count": 3,
                "max_distance": 3.0,
                "sub_identities": ["red_sphere", "green_sphere", "blue_sphere"],
            },
            "light_tower": {
                "count": 1,
                "max_distance": 5.0,
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
            self.obstacles_full_map[obstacle] = obstacle

        # Hypothesis manager for obstacle types
        self.hypothesis_managers = {
            obstacle_name: HypothesisManager(
                name=obstacle_name,
                max_distance=config[
                    "max_distance"
                ],  # can be parameterized per obstacle type
                max_num_hypothesis=config["count"] * 2,
            )
            for obstacle_name, config in self.obstacle_types.items()
        }

        # Subscribers and Publishers
        self.detection_subs = [
            self.create_subscription(
                DetectedObject3DArray,
                topic,
                self.detections_callback,
                10,
            )
            for topic in [
                "/asv4/vision/detections_2d/projected",
                "/asv4/vision/gate_detections",
                "/asv4/vision/lidar_small_objects/dets_3d/labelled",
            ]
        ]

        self.detections_pub = self.create_publisher(
            DetectedObject3DArray, "/asv4/robotx/filtered_detections", 10
        )

        self.get_logger().info(
            "Obstacle Management Server with Hypothesis Tracking started."
        )

    def detections_callback(self, msg):
        # Process each detected object
        for detection in msg.objects:
            obstacle_type = detection.hypothesis.class_id
            obstacle_name = self.id_to_name[obstacle_type]
            if obstacle_name in self.obstacles_full_map:
                # Extract position
                pos = np.array(
                    [
                        detection.hypothesis.kinematics.pose_with_covariance.pose.position.x,
                        detection.hypothesis.kinematics.pose_with_covariance.pose.position.y,
                        detection.hypothesis.kinematics.pose_with_covariance.pose.position.z,
                    ]
                )
                yaw = quat2euler(
                    attrgetter("w", "x", "y", "z")(
                        detection.hypothesis.kinematics.pose_with_covariance.pose.orientation
                    )
                )[2]

                identity = detection.hypothesis.class_id
                tid = detection.hypothesis.track_id

                parent_obstacle = self.obstacles_full_map[obstacle_name]
                # Update hypothesis manager for the specific obstacle type
                self.hypothesis_managers[parent_obstacle].update_hypothesis(
                    pos, yaw, identity, detection, tid
                )

        # Gather all hypotheses
        filtered_detections = DetectedObject3DArray()
        filtered_detections.header = msg.header
        filtered_detections.objects = []

        for manager in self.hypothesis_managers.values():
            for hyp_id, (
                positions,
                yaw_values,
                identities,
                tids,
                det,
            ) in manager.get_all_hypotheses().items():
                median_position = np.median(np.array(positions), axis=0)
                median_yaw = np.median(np.array(yaw_values))

                best_identity = max(identities, key=identities.get)

                detected_object = (
                    det  # Create a new DetectedObject3D based on median values
                )
                position = Point(x=median_position[0], y=median_position[1], z=median_position[2])
                detected_object.hypothesis.kinematics.pose_with_covariance.pose.position = position
                detected_object.hypothesis.class_id = best_identity
                detected_object.hypothesis.track_id = tids[-1]
                # Set orientation based on the median yaw
                # Set the orientation quaternion based on the yaw
                # Use a utility function to convert yaw to quaternion if needed

                filtered_detections.objects.append(deepcopy(detected_object))

        self.detections_pub.publish(filtered_detections)

    def create_detection_msg(self, centroid_position, identities, det):
        """Create a DetectedObject3D message based on the hypothesis's centroid and identity."""
        detection = det

        # Use the most common identity for the class_id
        most_common_identity = max(identities, key=identities.get)
        detection.hypothesis.class_id = self.name_to_id.get(most_common_identity, 0)

        # Set the centroid position
        detection.hypothesis.kinematics.pose_with_covariance.pose.position.x = float(
            centroid_position[-1][0]
        )
        detection.hypothesis.kinematics.pose_with_covariance.pose.position.y = float(
            centroid_position[-1][1]
        )
        detection.hypothesis.kinematics.pose_with_covariance.pose.position.z = (
            # centroid_position[2]
            0.0
        )
        return detection


def main(args=None):
    rclpy.init(args=args)
    obstacle_management_server = ObstaclesManagementServer()
    rclpy.spin(obstacle_management_server)
    obstacle_management_server.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
