#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from bb_perception_msgs.msg import DetectedObject3DArray
import numpy as np
from collections import deque
from ament_index_python.packages import get_package_share_directory
from pathlib import Path

from ml_detector.schema_validator import get_config, load_schema
from bb_filters.log import RclLogHandler
import logging
from operator import attrgetter
from transforms3d.euler import quat2euler, euler2quat

import numpy as np
from collections import deque
import logging
from copy import deepcopy
from geometry_msgs.msg import Point, Point32, Quaternion, PolygonStamped
from nav_msgs.msg import Odometry
import time
from threading import Lock

np.set_printoptions(formatter={"float": "{:0.3f}".format})
LOGGER = logging.getLogger("obstacles_management")


def throttle_log_info(callback, duration=1):
    last_updated = 0

    def log_info(msg):
        nonlocal last_updated, callback
        if time.time() - last_updated > 1:
            callback(msg)
            last_updated = time.time()

    return log_info


THROTTLE_LOG_INFO = throttle_log_info(LOGGER.info)


class Identities:
    def __init__(self, max_len=100):
        self.identities = {}
        self.max_len = max_len
        self.buffer = deque(maxlen=max_len)

    def update(self, identity):
        if len(self.buffer) == self.max_len:
            removed = self.buffer.popleft()
            self.identities[removed] -= 1
        self.identities[identity] = self.identities.get(identity, 0) + 1
        self.buffer.append(identity)
        return self

    def get_most_common(self):
        return max(self.identities, key=self.identities.get)

    def get_counter(self):
        return self.identities


class HypothesisManager:
    id_to_name = {}
    lock = Lock()

    def __init__(self, name, max_distance, max_num_hypothesis):
        self.name = name
        self.max_distance = max_distance
        self.max_num_hypothesis = max_num_hypothesis
        self.tid_buffer_size = 5

        # Dictionary of hypotheses
        self.hypotheses = {}

        # Queue for latest updated hypotheses
        self.latest_hypotheses = deque()
        self.to_remove = set()

    def update_hypothesis(
        self, new_position, new_yaw, new_identity, det, tid, stamp=None
    ):
        """Update the existing hypotheses or create a new one."""
        closest_hypothesis = None
        closest_distance = float("inf")
        with self.lock:
            # Search for the closest hypothesis within the allowed distance range
            for hyp_id, (
                positions,
                yaw_values,
                identities,
                tids,
                det,
                old_stamp,
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
                positions, yaw_values, identities, tids, det, old_stamp = self.hypotheses[
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
                identities.update(new_identity)

                if tid not in tids:
                    tids.append(tid)

                # Update the hypothesis with new values
                self.hypotheses[closest_hypothesis] = (
                    positions,
                    yaw_values,
                    identities,
                    tids,
                    det,
                    stamp,
                )
                if (
                    len(positions) > 0
                    and np.linalg.norm(positions[-1][:2] - new_position[:2]) > 0.3
                ):
                    LOGGER.info(
                        "obstacle %s updated %s -> %s %s %s %s %s %s %s",
                        closest_hypothesis,
                        positions[-1],
                        new_position,
                        new_identity,
                        self.name,
                        tids,
                        {
                            HypothesisManager.id_to_name.get(k, k): v
                            for k, v in identities.get_counter().items()
                            if v > 0
                        },
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
                identities = Identities().update(new_identity)
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
                    stamp,
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
        _to_remove = set()
        _to_remove |= self.to_remove
        self.to_remove.clear()
        while len(self.latest_hypotheses) > self.max_num_hypothesis:
            oldest_hypothesis = self.latest_hypotheses.pop()
            LOGGER.info("obstacle %s removed, %s %s", oldest_hypothesis, self.max_num_hypothesis,
                        self.latest_hypotheses)
            # del self.hypotheses[oldest_hypothesis]
            _to_remove.add(oldest_hypothesis)
        if len(_to_remove) > 0:
            for hyp_id in _to_remove:
                LOGGER.info("obstacle %s removed %s", hyp_id, _to_remove)
                del self.hypotheses[hyp_id]

    def get_all_hypotheses(self):
        """Get all current hypotheses for output purposes."""
        return self.hypotheses

    def remove_hypothesis(self, hypothesis_id):
        """Remove a hypothesis from the manager."""
        item = self.hypotheses.pop(hypothesis_id)
        if item is None:
            return
        if hypothesis_id in self.latest_hypotheses:
            self.latest_hypotheses.remove(hypothesis_id)


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
        HypothesisManager.id_to_name = self.id_to_name
        self.name_to_id = {v: k for k, v in self.id_to_name.items()}

        self.vehicle_odom = None
        self.odom_sub = self.create_subscription(
            Odometry,
            "/asv4/nav/world",
            self.odom_callback,
            10,
        )

        self.max_range = 50.0
        self.fov = np.deg2rad(270.0)
        self.max_disappear_time = (
            10.0  # detections in perceptive range disappear after 10 seconds
        )
        self.perceptive_range_local = np.array(
            [
                [0.0, 0.0, 1.0],
                *[
                    [
                        self.max_range * np.cos(angle),
                        self.max_range * np.sin(angle),
                        1.0,
                    ]
                    for angle in np.arange(-self.fov / 2, self.fov / 2, np.deg2rad(10))
                ],
                [0.0, 0.0, 1.0],
            ]
        )
        self.perceptive_range_map = None
        self.vehicle_bl = None  # x y dx dy
        self.perceptive_range_pub = self.create_publisher(
            PolygonStamped,
            "/asv4/perceptive_range",
            10,
        )

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
            "gate": {
                "count": 15,
                "max_distance": 5.0,
                "sub_identities": [],
            },
            "buoy": {
                "count": 30,
                "max_distance": 4,
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
        self.latest_stamp = None
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
                # "/asv4/vision/detections_2d/projected",
                "/asv4/vision/red_green_gate_detections",
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
        self.update_perceptive_range_timer = self.create_timer(
            1.0, self.update_perceptive_range
        )

    def odom_callback(self, msg):
        self.vehicle_odom = msg

    def update_perceptive_range(self):
        if self.vehicle_odom is None:
            return
        ps = PolygonStamped()
        ps.header = self.vehicle_odom.header
        self.yaw = quat2euler(
            attrgetter("w", "x", "y", "z")(self.vehicle_odom.pose.pose.orientation)
        )[2]
        map2local = np.array(
            [
                [
                    np.cos(self.yaw),
                    -np.sin(self.yaw),
                    self.vehicle_odom.pose.pose.position.x,
                ],
                [
                    np.sin(self.yaw),
                    np.cos(self.yaw),
                    self.vehicle_odom.pose.pose.position.y,
                ],
                [0, 0, 1],
            ]
        )
        self.perceptive_range_map = np.dot(map2local, self.perceptive_range_local.T).T
        self.vehicle_bl = np.zeros(4)
        self.vehicle_bl[:2] = self.perceptive_range_map[0, :2]
        self.vehicle_bl[2:] = (
            np.dot(map2local, np.array([1, 0, 1]))[:2] - self.vehicle_bl[:2]
        )
        ps.polygon.points = [
            Point32(x=x, y=y, z=0.0) for x, y, z in self.perceptive_range_map
        ]
        self.perceptive_range_pub.publish(ps)

    def det_in_perceptive_range(self, det):
        if self.vehicle_bl is None:
            return False
        det_pos = np.array(
            [
                det.hypothesis.kinematics.pose_with_covariance.pose.position.x,
                det.hypothesis.kinematics.pose_with_covariance.pose.position.y,
                1.0,
            ]
        )
        direction = det_pos[:2] - self.vehicle_bl[:2]
        dist = np.linalg.norm(direction)
        if dist > self.max_range:
            return False
        unit = direction / dist
        if np.dot(unit, self.vehicle_bl[2:] - self.vehicle_bl[:2]) < np.cos(
            self.fov / 2
        ):
            return False
        return True

    def detections_callback(self, msg):
        if self.latest_stamp is None:
            self.latest_stamp = Time.from_msg(msg.header.stamp)
        else:
            self.latest_stamp = max(Time.from_msg(msg.header.stamp), self.latest_stamp)
        debug_str = "-----------------------------------"
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
                    pos, yaw, identity, detection, tid, stamp=msg.header.stamp
                )

        # Gather all hypotheses
        filtered_detections = DetectedObject3DArray()
        filtered_detections.header = msg.header
        filtered_detections.header.stamp = self.get_clock().now().to_msg()
        filtered_detections.objects = []
        for manager in self.hypothesis_managers.values():
            debug_str += f"\n{manager.name}: {len(manager.hypotheses)}/{manager.max_num_hypothesis}\n"
            to_remove = set()
            for hyp_id, (
                positions,
                yaw_values,
                identities,
                tids,
                det,
                old_stamp,
            ) in manager.get_all_hypotheses().items():
                median_position = np.median(np.array(positions), axis=0)
                median_yaw = np.median(np.array(yaw_values))

                best_identity = identities.get_most_common()

                detected_object = (
                    det  # Create a new DetectedObject3D based on median values
                )
                position = Point(
                    x=median_position[0], y=median_position[1], z=median_position[2]
                )
                detected_object.hypothesis.kinematics.pose_with_covariance.pose.position = (
                    position
                )
                detected_object.hypothesis.class_id = best_identity
                detected_object.hypothesis.track_id = tids[-1]
                # Set orientation based on the median yaw
                # Set the orientation quaternion based on the yaw
                quat = euler2quat(0, 0, median_yaw)
                detected_object.hypothesis.kinematics.pose_with_covariance.pose.orientation = Quaternion(
                    w=quat[0], x=quat[1], y=quat[2], z=quat[3]
                )

                _last_updated = self.latest_stamp - Time.from_msg(old_stamp)
                _in_perceptive_range = self.det_in_perceptive_range(det)
                if (
                    _in_perceptive_range
                    and _last_updated.nanoseconds / 1e9 > self.max_disappear_time
                ):
                    # remove the detection if it is in perceptive range and has not been updated for a long time
                    to_remove.add(hyp_id)
                elif (
                    not _in_perceptive_range
                    and _last_updated.nanoseconds / 1e9 > self.max_disappear_time * 3
                    and sum(identities.get_counter().values()) < 0.5 * manager.max_num_hypothesis
                ):
                    # remove the detection if it is not in perceptive range and has not been updated for a long time and few detections
                    to_remove.add(hyp_id)
                filtered_detections.objects.append(deepcopy(detected_object))
                identity_str = {
                    HypothesisManager.id_to_name.get(k, k): v
                    for k, v in identities.get_counter().items()
                    if v > 0
                }
                debug_str += f"{hyp_id}: {median_position} {best_identity} {tids} {identity_str} {_last_updated.nanoseconds/1e9} {self.latest_stamp} {_in_perceptive_range}\n"
            for hyp_id in to_remove:
                manager.to_remove.add(hyp_id)
        THROTTLE_LOG_INFO(debug_str)
        self.detections_pub.publish(filtered_detections)

    def create_detection_msg(self, centroid_position, identities, det):
        """Create a DetectedObject3D message based on the hypothesis's centroid and identity."""
        detection = det

        # Use the most common identity for the class_id
        most_common_identity = identities.get_most_common()
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
