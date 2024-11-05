#!/usr/bin/env python3
"""
Light tower detection Node for ROS2

This ROS2 node detects color sequence based on detections 2d from camera.

The node performs the following tasks:
1. Subscribes to a `DetectedObject2DArray` topic to receive panel color data.

# TODO fill this up.

Dependencies:
- rclpy for ROS2 node functionality.
"""
from collections import Counter, deque
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from bb_perception_msgs.msg import (
    DetectedObject2DArray,
    DetectedObject3DArray,
    DetectorSource,
)
from bb_robotx_msgs.msg import LightSequence
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from ml_detector.schema_validator import get_config, load_schema
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from shapely.geometry import Point, Polygon
from sklearn.cluster import DBSCAN
from std_msgs.msg import Bool
from tf2_ros import TransformBroadcaster
from transforms3d.euler import euler2quat
from bb_robotx_msgs.srv import ConfigureLightSequenceTask


class LightTowerDetection(Node):
    COLORS_LIST = ["black", "red", "blue", "green"]
    colors = {colour: i for i, colour in enumerate(COLORS_LIST)}
    objects_schema_path = (
        Path(get_package_share_directory("ml_detector"))
        / "configs"
        / "objects_schema.json"
    )
    objects_schema = load_schema(objects_schema_path)
    objects_config = get_config(
        Path(get_package_share_directory("ml_detector"))
        / "configs"
        / "objects"
        / "robotx.yaml",
        objects_schema,
    )
    ID_TO_NAME = {obj["label"]: obj["name"] for obj in objects_config["objects"]}
    NAME_TO_ID = {v: k for k, v in ID_TO_NAME.items()}
    BLUE_PANEL_ID = NAME_TO_ID["light_tower_panel_blue"]
    GREEN_PANEL_ID = NAME_TO_ID["light_tower_panel_green"]
    RED_PANEL_ID = NAME_TO_ID["light_tower_panel_red"]
    BLACK_PANEL_ID = NAME_TO_ID["light_tower_panel_black"]
    LIGHT_TOWER_ID = NAME_TO_ID["light_tower"]

    ID_TO_COLOR = {
        BLACK_PANEL_ID: 0,
        RED_PANEL_ID: 1,
        BLUE_PANEL_ID: 2,
        GREEN_PANEL_ID: 3,
    }
    COLOR_ENUMS = [
        LightSequence.UNKNOWN,
        LightSequence.RED,
        LightSequence.BLUE,
        LightSequence.GREEN,
    ]
    COLOR_ENUM_TO_COLOR = {
        LightSequence.UNKNOWN: (0, 0, 0),
        LightSequence.RED: (0, 0, 255),
        LightSequence.BLUE: (255, 0, 0),
        LightSequence.GREEN: (0, 255, 0),
    }

    def __init__(self):
        super().__init__("light_tower_detection")
        self.running = False
        self.bridge = CvBridge()
        self.image = None
        self.declare_parameter("debug", True)
        self.declare_parameter("namespace", "asv4")
        self.namespace = (
            self.get_parameter("namespace").get_parameter_value().string_value
        )
        self.declare_parameter("use_time_colour_map", True)
        self.use_time_colour_map = (
            self.get_parameter("use_time_colour_map").get_parameter_value().bool_value
        )
        self.declare_parameter("panel_in_tower_check", True)
        self.panel_in_tower_check = (
            self.get_parameter("panel_in_tower_check").get_parameter_value().bool_value
        )
        self.time_colour_map_granularity = 8  # 0.25 seconds
        self.debug = self.get_parameter("debug").get_parameter_value().bool_value
        self.detector_source = DetectorSource(
            sensor_name="light_tower_detector",
            frame_id=f"{self.namespace}/base_link",
            category=DetectorSource.LIDAR,
        )

        # Begin: Light tower pose estimation variables
        self.light_tower_known_height = 2  # meters
        self.light_tower_positions = deque(maxlen=150)
        self.last_pose_estimate = None
        # Noise model for measurements
        self.tf_broadcaster = TransformBroadcaster(self)
        self.latest_stamp = self.get_clock().now()
        # End: Light tower pose estimation variables
        self.vehicle_position = (0, 0)

        self.degree = 1  # only update transition up to degree time steps in the past.
        # e.g. if degree is 2, increments state n-1 -> n by 2 and state n-2 -> n by 1
        self.latest_colors = deque(maxlen=self.degree)
        self.scaling_factor = (
            self.degree * (self.degree + 1) / 2
        )  # scale transition matrix by factor
        self.best_sequence_threshold = 3

        self.reset()

        self.debug_pub = self.create_publisher(
            Image, f"/{self.namespace}/robotx/light_tower/debug", 10
        )  # 5x4 image
        self.tcm_debug_pub = self.create_publisher(
            Image, f"/{self.namespace}/robotx/light_tower/tcm_debug", 10
        )  # 4x4 image
        self.light_sequence_pub = self.create_publisher(
            LightSequence, "/robotx24/light_sequence_raw", 1
        )
        self.light_tower_valid_pose_pub = self.create_publisher(
            Bool, "/robotx24/light_tower/valid_pose", 1
        )  # publish if light tower pose is valid
        self.subscription_2d = self.create_subscription(
            DetectedObject2DArray,
            (
                f"/{self.namespace}/vision/detections_2d"
                # f"/{self.namespace}/vision/detections_2d/fixed"
                # if self.namespace == "asv4"
                # else f"/{self.namespace}/vision/detections_2d"
                # "/asv4/vision/detections_2d_relabelled"
            ),
            self.detected_objects_2d_callback,
            10,
        )
        self.subscription_3d = self.create_subscription(
            DetectedObject3DArray,
            # f"/{self.namespace}/vision/detections_2d/projected/filtered",
            f"/{self.namespace}/robotx/filtered_detections",
            self.detected_objects_3d_callback,
            10,
        )
        self.odom_sub = self.create_subscription(
            Odometry, f"/{self.namespace}/nav/world", self.odom_callback, 1
        )
        self.create_timer(1, self.publish_sequence)
        self.update_timer = self.create_timer(0.5, self.update_pose_estimate)
        self.config_service = self.create_service(
            ConfigureLightSequenceTask, "/robotx24/configure_light_sequence_task", self.configure_task
        )

    def initialize_debug_img(self):
        img = np.zeros((6, 5, 3)).astype(np.uint8)
        img[0, 2] = img[2, 0] = np.array([0, 0, 255])
        img[0, 3] = img[3, 0] = np.array([255, 0, 0])
        img[0, 4] = img[4, 0] = np.array([0, 255, 0])
        return img

    def initialize_tcm_debug_img(self):
        img = np.zeros((7, 4, 3)).astype(np.uint8)
        img[0, 1] = np.array([0, 0, 255])
        img[0, 2] = np.array([255, 0, 0])
        img[0, 3] = np.array([0, 255, 0])
        return img

    def reset(self):
        self.transition_matrix = np.zeros((4, 4))
        self.debug_img = self.initialize_debug_img()
        self.tcm_debug_img = self.initialize_tcm_debug_img()
        self.latest_colors.clear()
        self.best_sequence = None
        self.best_sequence_count = 0
        self.best_sequence_tcm = None
        self.best_sequence_count_tcm = 0
        self.best_sequence_count_tcm_last_increment = 0
        self.time_colour_map = np.zeros(
            (5 * self.time_colour_map_granularity, 4)
        )  # rows: 0 1 2 3 4, cols: black red blue green
        self.tcm_fixed = None
        self.light_tower_positions.clear()

    def configure_task(self,
                       request: ConfigureLightSequenceTask.Request,
                       response: ConfigureLightSequenceTask.Response):
        if not request.active:
            self.get_logger().info("configure_task inactive")
            self.running = False
            response.success = True
            return response
        if self.running:
            # self.get_logger().info("configure_task already running, ignoring")
            response.success = True
            return response
        if not self.running:
            self.get_logger().info("configure_task starting, resetting")
            self.use_time_colour_map = request.use_tcm
            self.time_colour_map_granularity = request.tcm_granularity
            self.panel_in_tower_check = request.filter_panel_in_tower
            self.best_sequence_threshold = request.best_sequence_threshold
            self.light_tower_known_height = request.light_tower_known_height

            self.reset()
            self.running = True
        response.success = True
        return response

    def publish_tf_transform(self, centroid):
        """Publish the latest estimated object pose as a TF transform."""
        t = TransformStamped()
        # Fill in header information
        new_stamp = self.get_clock().now()
        if new_stamp <= self.latest_stamp:
            self.get_logger().warn("New stamp is not greater than latest stamp")
            return
        self.latest_stamp = new_stamp
        t.header.stamp = new_stamp.to_msg()
        t.header.frame_id = "map"  # Global frame
        # Set the name of the object (e.g., "tracked_object")
        t.child_frame_id = "light_tower"
        # Set the position (translation)
        t.transform.translation.x = centroid[0]
        t.transform.translation.y = centroid[1]
        t.transform.translation.z = 0.0

        # compute orientation based on heading relative to vehicle
        yaw = np.arctan2(
            centroid[1] - self.vehicle_position[1],
            centroid[0] - self.vehicle_position[0],
        )
        quat = euler2quat(0, 0, yaw)
        t.transform.rotation.w = quat[0]
        t.transform.rotation.x = quat[1]
        t.transform.rotation.y = quat[2]
        t.transform.rotation.z = quat[3]

        # Broadcast the transform
        self.tf_broadcaster.sendTransform(t)

    def get_seq_from_time_colour_map(self):
        best_colours = np.argmax(self.time_colour_map, axis=1)
        if np.all(best_colours == 0):
            self.get_logger().info("get_seq_from_time_colour_map all black")
            return

        # check that there are 2 best colours that are 0 and their indices are adjacent or at the start and end
        first_non_black_idx = np.where(best_colours != 0)[0][0]
        last_black_idx = np.where(best_colours == 0)[0][-1]
        if last_black_idx < 5 * self.time_colour_map_granularity - 1:
            first_non_black_idx = last_black_idx + 1
        sorted_colours = (
            np.roll(self.time_colour_map, -first_non_black_idx, axis=0)
            .reshape(5, self.time_colour_map_granularity, 4)
            .sum(axis=1)
        )
        self.tcm_fixed = sorted_colours
        new_best_colours = np.argmax(sorted_colours, axis=1)
        print(new_best_colours, sorted_colours)
        if any(np.sum(sorted_colours, axis=1) == 0):
            self.get_logger().info("get_seq_from_time_colour_map some empty")
            return
        if any(col == 0 for col in new_best_colours[:3]):
            self.get_logger().info(f"get_seq_from_time_colour_map some colour black")
            return
        if any(col != 0 for col in new_best_colours[3:]):
            self.get_logger().info(f"get_seq_from_time_colour_map last 2 not black")
            return
        if (
            new_best_colours[0] == new_best_colours[1]
            or new_best_colours[1] == new_best_colours[2]
        ):
            self.get_logger().info(
                "get_seq_from_time_colour_map consecutive same colour invalid"
            )
            return
        # check that best colours are at least 0.5 of everything
        if np.any(
            np.sort(sorted_colours, axis=1)[:, -1] / np.sum(sorted_colours, axis=1)
            < 0.5
        ):
            return
        return LightSequence(
            first=self.COLOR_ENUMS[new_best_colours[0]],
            second=self.COLOR_ENUMS[new_best_colours[1]],
            third=self.COLOR_ENUMS[new_best_colours[2]],
        )

    def check_condition_tcm(self):
        if self.best_sequence_count_tcm >= self.best_sequence_threshold:
            return True
        new_sequence = self.get_seq_from_time_colour_map()
        if new_sequence is None:
            self.best_sequence_count_tcm -= 1
            return False
        if new_sequence == self.best_sequence_tcm:
            current_time = Time.from_msg(self.get_clock().now().to_msg())
            if (
                current_time.nanoseconds - self.best_sequence_count_tcm_last_increment
                >= 5e9
            ):
                self.get_logger().info("check_condition_tcm incrementing")
                self.best_sequence_count_tcm_last_increment = current_time.nanoseconds
                self.best_sequence_count_tcm += 1
            self.best_sequence_count_tcm += 1
        else:
            self.best_sequence_count_tcm = 1
            self.best_sequence_count_tcm_last_increment = Time.from_msg(
                self.get_clock().now().to_msg()
            ).nanoseconds
        self.best_sequence_tcm = new_sequence
        return True

    def check_condition(self):
        if np.sum(self.transition_matrix >= 2) >= 3:
            c1 = np.argmax(self.transition_matrix[0, 1:]) + 1
            c2 = np.argmax(self.transition_matrix[c1, 1:]) + 1
            c3 = np.argmax(self.transition_matrix[c2, 1:]) + 1
            c31 = np.argmax(self.transition_matrix[1:, 0]) + 1
            c21 = np.argmax(self.transition_matrix[1:, c31]) + 1
            c11 = np.argmax(self.transition_matrix[1:, c21]) + 1
            if any(
                (
                    self.transition_matrix[0, c1] == 0,
                    self.transition_matrix[c1, c2] == 0,
                    self.transition_matrix[c2, c3] == 0,
                    self.transition_matrix[c3, 0] == 0,
                )
            ):
                return False
            if not (c1 == c11 and c2 == c21 and c3 == c31):
                self.get_logger().warn(f"{c1} {c2} {c3} != {c11} {c21} {c31}, waiting")
                self.best_sequence_count = 0
                return False
            if c1 != c2 != c3:
                new_sequence = LightSequence(
                    first=self.COLOR_ENUMS[c1],
                    second=self.COLOR_ENUMS[c2],
                    third=self.COLOR_ENUMS[c3],
                )
                if new_sequence == self.best_sequence:
                    self.best_sequence_count += 1
                else:
                    self.best_sequence_count = 1
                self.best_sequence = new_sequence
                return True
            if c1 != c2:
                # This checks for a sequence where first and third color are the same
                new_sequence = LightSequence(
                    first=self.COLOR_ENUMS[c1],
                    second=self.COLOR_ENUMS[c2],
                    third=self.COLOR_ENUMS[c1],
                )
                if new_sequence == self.best_sequence:
                    self.best_sequence_count += 1
                else:
                    self.best_sequence_count = 1
                self.best_sequence = new_sequence
                return True

            # no consecutive same colour
            # if c1 == c2 and c1 != c3:
            #     # This checks for a sequence where first and second are the same
            #     new_sequence = LightSequence(
            #         first=self.COLOR_ENUMS[c1],  # Green (or any color that c1 maps to)
            #         second=self.COLOR_ENUMS[c2], # Green (same as first)
            #         third=self.COLOR_ENUMS[c3],  # Red (or any color that c3 maps to)
            #     )
            #     if new_sequence == self.best_sequence:
            #         self.best_sequence_count += 1
            #     else:
            #     self.best_sequence_count = 1
            #     self.best_sequence = new_sequence
            #     return True
            # elif c1 == c11 and c2 == c21 and c3 == c11 and c1 != c2:
            #     # This checks for a sequence where first and third color are the same
            #     new_sequence = LightSequence(
            #         first=self.COLOR_ENUMS[c1],
            #         second=self.COLOR_ENUMS[c2],
            #         third=self.COLOR_ENUMS[c1],
            #     )
            #     if new_sequence == self.best_sequence:
            #         self.best_sequence_count += 1
            #     else:
            #         self.best_sequence_count = 1
            #     self.best_sequence = new_sequence
            #     return True

    def publish_sequence(self):
        if not self.running:
            return
        if self.debug and self.transition_matrix.max() > 0:
            self.debug_img[1:5, 1:5, :] = np.repeat(
                self.transition_matrix.reshape((4, 4, 1))
                / self.transition_matrix.max()
                * 255,
                3,
                axis=2,
            )
            self.debug_img = self.debug_img.astype(np.uint8)
            debug_img = self.bridge.cv2_to_imgmsg(self.debug_img, encoding="bgr8")
            self.debug_pub.publish(debug_img)
        if self.use_time_colour_map and self.tcm_fixed is not None:
            # given array of shape (5 * time_colour_map_granularity, 4), determine the maximum and rescale entire array to 0-255, then convert to uint8
            tcm_debug_arr = np.repeat(
                self.tcm_fixed.reshape((5, 4, 1)) / self.tcm_fixed.max() * 255,
                3,
                axis=2,
            ).astype(np.uint8)
            self.tcm_debug_img[1:6, :] = tcm_debug_arr
            tcm_debug_img_msg = self.bridge.cv2_to_imgmsg(
                self.tcm_debug_img, encoding="bgr8"
            )
            self.tcm_debug_pub.publish(tcm_debug_img_msg)
        # if self.best_sequence_count < self.best_sequence_threshold:
        self.check_condition()
        # if self.best_sequence_count_tcm < self.best_sequence_threshold:
        self.check_condition_tcm()
        # store best sequence and don't recompute subsequently (no point since probably reported alr)
        if self.best_sequence_count >= self.best_sequence_threshold:
            self.get_logger().info(
                f"Best sequence: {self.best_sequence} detected {self.best_sequence_count} times"
            )
            if not self.use_time_colour_map:
                self.light_sequence_pub.publish(self.best_sequence)
            self.debug_img[5, 1] = self.COLOR_ENUM_TO_COLOR[self.best_sequence.first]
            self.debug_img[5, 2] = self.COLOR_ENUM_TO_COLOR[self.best_sequence.second]
            self.debug_img[5, 3] = self.COLOR_ENUM_TO_COLOR[self.best_sequence.third]
        if self.best_sequence_count_tcm >= self.best_sequence_threshold:
            self.get_logger().info(
                f"Best sequence from time colour map: {self.best_sequence_tcm}"
            )
            if self.use_time_colour_map:
                self.light_sequence_pub.publish(self.best_sequence_tcm)
            self.tcm_debug_img[6, 1] = self.COLOR_ENUM_TO_COLOR[
                self.best_sequence_tcm.first
            ]
            self.tcm_debug_img[6, 2] = self.COLOR_ENUM_TO_COLOR[
                self.best_sequence_tcm.second
            ]
            self.tcm_debug_img[6, 3] = self.COLOR_ENUM_TO_COLOR[
                self.best_sequence_tcm.third
            ]
        return
        # compute best sequence

    def detected_objects_2d_callback(self, msg):
        if not self.running:
            return
        if len(msg.objects) == 0:
            return

        tower_dets = [
            det for det in msg.objects if det.hypothesis.class_id == self.LIGHT_TOWER_ID
        ]
        # look for placards in objects list
        panel_dets = [
            det
            for det in msg.objects
            if det.hypothesis.class_id
            in [
                self.BLUE_PANEL_ID,
                self.GREEN_PANEL_ID,
                self.RED_PANEL_ID,
                self.BLACK_PANEL_ID,
            ]
        ]

        if self.panel_in_tower_check:
            light_tower_polys = [
                Polygon(np.array(det.contour).reshape(-1, 2)) for det in tower_dets
            ]
            panel_dets = [
                det
                for det in panel_dets
                if any(
                    poly.contains(
                        Point(
                            det.centre_x,
                            det.centre_y,
                        )
                    )
                    for poly in light_tower_polys
                )
            ]

        if len(panel_dets) == 0:
            self.get_logger().info("No panel detected.")
            return
        else:
            best_placard = max(panel_dets, key=lambda det: det.hypothesis.probability)
            color = self.ID_TO_COLOR[best_placard.hypothesis.class_id]
            self.get_logger().info("Detected panel: " + self.COLORS_LIST[color])
        if len(self.latest_colors) >= self.degree:
            for i, prev_color in enumerate(self.latest_colors):
                if prev_color != color:
                    self.transition_matrix[prev_color][color] += (
                        (i + 1)
                        / self.scaling_factor
                        * best_placard.hypothesis.probability
                    )
        self.latest_colors.append(color)
        self.time_colour_map[
            int(
                (
                    Time.from_msg(msg.header.stamp).nanoseconds
                    / 1e9
                    * self.time_colour_map_granularity
                )
                % (5 * self.time_colour_map_granularity)
            ),
            color,
        ] += best_placard.hypothesis.probability

    def detected_objects_3d_callback(self, msg):
        if not self.running:
            self.get_logger().info("light tower detector Not running", throttle_duration_sec=2)
            return
        else:
            self.get_logger().info("light tower detector running", throttle_duration_sec=2)
        towers = [
            det for det in msg.objects if det.hypothesis.class_id == self.LIGHT_TOWER_ID
        ]
        if len(towers) == 0:
            self.get_logger().info("No light tower detected.")
            return
        best_tower = max(towers, key=lambda det: det.hypothesis.probability)
        self.light_tower_positions.append(
            (
                best_tower.hypothesis.kinematics.pose_with_covariance.pose.position.x,
                best_tower.hypothesis.kinematics.pose_with_covariance.pose.position.y,
            ),
        )
        return

    def update_pose_estimate(self):
        """
        Update the pose estimate based on clustered light tower positions.

        Parameters:
        - msg: Message containing light tower positions (assumed to be a list of [x, y, z] coordinates).
        """
        if not self.running:
            self.get_logger().info("light tower detector Not running", throttle_duration_sec=2)
            return
        # Extract light tower positions from the msg
        light_tower_positions = np.array(
            self.light_tower_positions
        )  # Ensure this extracts correctly
        if len(self.light_tower_positions) < 3:
            self.get_logger().info(
                f"Insufficient light tower positions to compute centroid. {self.light_tower_positions}"
            )
            return
        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=0.5, min_samples=2)  # Adjust eps and min_samples as needed
        labels = dbscan.fit_predict(light_tower_positions)

        # Count the occurrences of each cluster (ignore outliers)
        label_counts = Counter(labels)

        # Ignore the noise label (-1) and find the largest cluster label
        if -1 in label_counts:
            del label_counts[-1]  # Remove outliers from the count

        if not label_counts:
            self.get_logger().info("No clusters found. Unable to compute centroid.")
            return None  # Return or handle the case of no valid positions

        # Find the label of the largest cluster
        largest_cluster_label = max(label_counts, key=label_counts.get)

        # Get the indices of the largest cluster
        largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
        largest_cluster_positions = light_tower_positions[largest_cluster_indices]

        # Calculate the centroid of the largest cluster
        centroid = np.mean(largest_cluster_positions, axis=0)

        # Update the pose estimate
        self.get_logger().info(
            f"Updated pose estimate (centroid of largest cluster): {centroid}"
        )

        # If needed, store or use the centroid for further processing
        # For example, you might want to store it in an instance variable
        self.last_pose_estimate = centroid  # or another suitable attribute

        if len(largest_cluster_positions) > 3:
            self.light_tower_valid_pose_pub.publish(Bool(data=True))
        else:
            self.light_tower_valid_pose_pub.publish(Bool(data=False))

        # publish tf
        self.publish_tf_transform(centroid)

    def odom_callback(self, msg):
        self.vehicle_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)


def main(args=None):
    rclpy.init(args=args)
    light_tower_detection = LightTowerDetection()
    rclpy.spin(light_tower_detection)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
