#!/usr/bin/env python3
import rclpy
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from bb_perception_msgs.msg import (
    DetectedObject2DArray,
    DetectedObject2D,
)
from std_msgs.msg import String
from ml_detector.schema_validator import get_config, load_schema
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
from copy import deepcopy

import numpy


def patch_asscalar(a):
    return a.item()


setattr(numpy, "asscalar", patch_asscalar)


class RelabelNode(Node):
    def __init__(self):
        super().__init__("relabel_node")
        self.subscription = self.create_subscription(
            DetectedObject2DArray,
            "/asv4/vision/detections_2d_base_link",
            self.detected_objects_2d_callback,
            10,
        )
        self.publisher = self.create_publisher(
            DetectedObject2DArray,
            "/asv4/vision/detections_2d_base_link_relabelled",  # Topic to publish relabeled objects
            10,
        )
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
        self.ID_TO_NAME = {obj["label"]: obj["name"] for obj in objects_config["objects"]}
        self.NAME_TO_ID = {v: k for k, v in self.ID_TO_NAME.items()}
        self.BLUE_PANEL_ID = self.NAME_TO_ID["light_tower_panel_blue"]
        self.GREEN_PANEL_ID = self.NAME_TO_ID["light_tower_panel_green"]
        self.RED_PANEL_ID = self.NAME_TO_ID["light_tower_panel_red"]
        self.BLACK_PANEL_ID = self.NAME_TO_ID["light_tower_panel_black"]
        self.reference_colors = {
            self.RED_PANEL_ID: LabColor(lab_l=(102 / 255) * 100, lab_a=64, lab_b=53),
            self.GREEN_PANEL_ID: LabColor(lab_l=(173 / 255) * 100, lab_a=-70, lab_b=67),
            self.BLUE_PANEL_ID: LabColor(lab_l=(58 / 255) * 100, lab_a=63, lab_b=-86),
            self.BLACK_PANEL_ID: LabColor(lab_l=0.0, lab_a=0.0, lab_b=0.0),
        }

    def detected_objects_2d_callback(self, msg):
        relabeled_detections = DetectedObject2DArray()
        relabeled_detections.header = msg.header

        for detection in msg.objects:
            if detection.hypothesis.class_id not in self.reference_colors:
                relabeled_detections.objects.append(detection)
                continue

            if len(detection.middle_lab) != 3:
                continue

            relabeled_detection = deepcopy(detection)

            opencv_l, opencv_a, opencv_b = detection.middle_lab
            lab_color = LabColor(
                lab_l=(opencv_l / 255) * 100, lab_a=opencv_a - 128, lab_b=opencv_b - 128
            )

            min_diff = float("inf")
            best_match = None
            for label, ref_color in self.reference_colors.items():
                diff = delta_e_cie2000(lab_color, ref_color)
                print(label, diff)
                if diff < min_diff:
                    min_diff = diff
                    best_match = label
            print(self.ID_TO_NAME[relabeled_detection.hypothesis.class_id],
                  self.ID_TO_NAME[best_match])

            relabeled_detection.hypothesis.class_id = best_match
            relabeled_detections.objects.append(relabeled_detection)

        # Publish the relabeled detections
        self.publisher.publish(relabeled_detections)
        self.get_logger().info(f"Published relabeled objects with LAB color matching")


def main(args=None):
    rclpy.init(args=args)
    node = RelabelNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
