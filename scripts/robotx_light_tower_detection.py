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
from pathlib import Path
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from bb_perception_msgs.msg import (
    DetectedObject2DArray,
    DetectorSource,
)
from bb_robotx_msgs.msg import LightSequence
from cv_bridge import CvBridge
from ml_detector.schema_validator import get_config, load_schema
from rclpy.node import Node
from sensor_msgs.msg import Image
from collections import deque


class LightTowerDetection(Node):
    def __init__(self):
        super().__init__("light_tower_detection")
        self.buoys = {}
        self.bridge = CvBridge()
        self.image = None
        self.declare_parameter("debug", True)
        self.debug = self.get_parameter("debug").get_parameter_value().bool_value
        self.is_ned = False
        self.header = None
        self.detector_source = DetectorSource(
            sensor_name="light_tower_detector",
            frame_id="asv4/base_link",
            category=DetectorSource.LIDAR,
        )

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
        self.blue_panel_id = self.name_to_id["light_tower_panel_blue"]
        self.green_panel_id = self.name_to_id["light_tower_panel_green"]
        self.red_panel_id = self.name_to_id["light_tower_panel_red"]
        self.black_panel_id = self.name_to_id["light_tower_panel_black"]

        self.degree = 10  # only update transition up to degree time steps in the past.
        # e.g. if degree is 2, increments state n-1 -> n by 2 and state n-2 -> n by 1
        self.colors = {
            "black": 0,
            "red": 1,
            "blue": 2,
            "green": 3,
        }
        self.id_to_color = {
            self.black_panel_id: 0,
            self.red_panel_id: 1,
            self.blue_panel_id: 2,
            self.green_panel_id: 3
        }
        self.colors_enums = [
            LightSequence.UNKNOWN,
            LightSequence.RED,
            LightSequence.BLUE,
            LightSequence.GREEN,
        ]
        self.transition_matrix = np.zeros((4, 4))
        self.debug_img = np.zeros((5, 5, 3))
        self.debug_img[0, 1] = self.debug_img[1, 0] = np.array([0, 0, 255])
        self.debug_img[0, 2] = self.debug_img[2, 0] = np.array([255, 0, 0])
        self.debug_img[0, 3] = self.debug_img[3, 0] = np.array([0, 255, 0])
        self.scaling_factor = (
            self.degree * (self.degree + 1) / 2
        )  # scale transition matrix by factor
        self.latest_colors = deque(maxlen=self.degree)
        self.best_sequence = None

        self.debug_pub = self.create_publisher(
            Image, "/asv4/robotx/light_tower/debug", 10
        )  # 4x4 image
        self.light_sequence_pub = self.create_publisher(
            LightSequence, "/robotx24/light_sequence", 1
        )
        # self.subscription = self.create_subscription(
        #     DetectedObject3DArray,
        #     "/asv4/vision/lidar_small_objects/dets_3d/labelled",
        #     self.detected_objects_3d_callback,
        #     10,
        # )
        self.subscription_2d = self.create_subscription(
            DetectedObject2DArray,
            "/asv4/vision/detections_2d",
            self.detected_objects_2d_callback,
            10,
        )
        self.create_timer(1, self.publish_sequence)

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
            if c1 == c11 and c2 == c21 and c3 == c31 and c1 != c2 and c1 != c3 and c2 != c3:
                self.best_sequence = LightSequence(
                    first=self.colors_enums[c1],
                    second=self.colors_enums[c2],
                    third=self.colors_enums[c3],
                )
                return True
            else:
                self.get_logger().warn(f"{c1} {c2} {c3} != {c11} {c21} {c31}, waiting")
                return False

    def publish_sequence(self):
        if self.debug and self.transition_matrix.max() > 0:
            self.debug_img[1:, 1:, 0] = self.transition_matrix / self.transition_matrix.max() * 255
            debug_img = self.bridge.cv2_to_imgmsg(self.debug_img)
            self.debug_pub.publish(debug_img)
        if self.best_sequence is not None:
            self.light_sequence_pub.publish(self.best_sequence)
            return
        # compute best sequence
        if not self.check_condition():
            return
        assert self.best_sequence is not None
        self.light_sequence_pub.publish(self.best_sequence)

    def detected_objects_2d_callback(self, msg):
        if len(msg.objects) == 0:
            return
        # look for placards in objects list
        placard_dets = [
            det
            for det in msg.objects
            if det.hypothesis.class_id
            in [
                self.blue_panel_id,
                self.green_panel_id,
                self.red_panel_id,
                self.black_panel_id,
            ]
        ]
        if len(placard_dets) == 0:
            return
        best_placard = max(placard_dets, key=lambda det: det.hypothesis.probability)
        color = self.id_to_color[best_placard.hypothesis.class_id]
        if len(self.latest_colors) >= self.degree:
            for i, prev_color in enumerate(self.latest_colors):
                if prev_color != color:
                    print(f"transition {prev_color} -> {color}")
                    self.transition_matrix[prev_color][color] += (i + 1) / self.scaling_factor
        self.latest_colors.append(color)
        self.get_logger().info(f"{self.transition_matrix}")

    # def calculate_placard_pose(self, cluster):
    #     # cluster the cluster into 2 clusters based on the x,y positions of the buoys
    #     if len(cluster) < 2:
    #         return None, None, None
    #     km = KMeans(n_clusters=2)
    #     positions = np.array([[t[1][0], t[1][1]] for t in cluster])
    #     green_red_clusters = km.fit_predict(positions)
    #     cluster_centers = km.cluster_centers_
    #     green_identities = [0, 0]  # green_red_cluster_0, green_red_cluster_1
    #     for i, c in enumerate(green_red_clusters):
    #         probabilities = cluster[i][1][2]
    #         if c == 0:
    #             green_identities[0] += probabilities[1] - probabilities[0]
    #         else:
    #             green_identities[1] += probabilities[1] - probabilities[0]
    #     if green_identities[0] > green_identities[1]:
    #         green_buoy_cluster = 0
    #     else:
    #         green_buoy_cluster = 1
    #     green_buoy_pose = cluster_centers[green_buoy_cluster]
    #     red_buoy_pose = cluster_centers[1 - green_buoy_cluster]
    #     gate_position = [
    #         (green_buoy_pose[0] + red_buoy_pose[0]) / 2,
    #         (green_buoy_pose[1] + red_buoy_pose[1]) / 2,
    #     ]
    #     gate_orientation = np.arctan2(
    #         green_buoy_pose[1] - red_buoy_pose[1], green_buoy_pose[0] - red_buoy_pose[0]
    #     )
    #     gate_width = np.linalg.norm(np.array(green_buoy_pose) - np.array(red_buoy_pose))
    #     if self.is_ned:
    #         gate_orientation -= np.pi / 2
    #     else:
    #         gate_orientation += np.pi / 2
    #     return gate_position, gate_orientation, gate_width


def main(args=None):
    rclpy.init(args=args)
    light_tower_detection = LightTowerDetection()
    rclpy.spin(light_tower_detection)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
