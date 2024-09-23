#!/usr/bin/env python3

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
import tf2_ros
from ament_index_python.packages import get_package_share_directory
from bb_perception_msgs.msg import DetectedObject2D, DetectedObject2DArray
from geometry_msgs.msg import Point, Vector3
from ml_detector.helpers.log import RclLogHandler
from ml_detector.schema_validator import get_config, load_schema
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


class DetectedObject2DArrayVisNode(Node):
    def __init__(self):
        super().__init__("detected_object_2d_visualization_node")
        self.declare_parameter(
            "input_detections_topics",
            [
                "/asv4/vision/detections_2d_left",
                "/asv4/vision/detections_2d_right",
                "/asv4/vision/detections_2d_front",
            ],
        )
        self.declare_parameter(
            "camera_info_topics",
            [
                "/asv4/left_cam/camera_info",
                "/asv4/right_cam/camera_info",
                "/asv4/front_cam/camera_info",
            ],
        )
        self.declare_parameter(
            "output_markers_topic", "/asv4/vision/detections_2d/marker"
        )
        self.declare_parameter("objects_config", "robotx.yaml")

        self.input_detections_topics = (
            self.get_parameter("input_detections_topics")
            .get_parameter_value()
            .string_array_value
        )
        self.camera_info_topics = (
            self.get_parameter("camera_info_topics")
            .get_parameter_value()
            .string_array_value
        )
        self.output_markers_topic = (
            self.get_parameter("output_markers_topic")
            .get_parameter_value()
            .string_value
        )
        self.logger = logging.getLogger("detected_object_2d_visualization")
        self.logger.level = logging.INFO
        self.logger.propagate = False
        self.logger.addHandler(
            RclLogHandler(self.get_logger(), "detected_object_2d_visualization")
        )
        self.logger.info("Initializing 2D Det Vis Node...")

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

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.publisher = self.create_publisher(
            MarkerArray, self.output_markers_topic, 10
        )

        self.camera_info_subscribers = [
            self.create_subscription(
                CameraInfo,
                topic,
                self.camera_info_callback,
                1
            )
            for topic in self.camera_info_topics
        ]

        self.camera_info_dict: Dict[str, CameraInfo] = {}

        self.detection_subscribers = [
            self.create_subscription(
                DetectedObject2DArray,
                topic,
                self.callback,
                1
            ) for topic in self.input_detections_topics
        ]

    def camera_info_callback(self, camera_info: CameraInfo):
        self.camera_info_dict[camera_info.header.frame_id] = camera_info

    def get_color(self, i):
        # random color
        r = max(10.0, ((hash(i) >> 16) % 256) / 255.0)
        g = max(10.0, ((hash(i) >> 8) % 256) / 255.0)
        b = max(10.0, (hash(i) % 256) / 255.0)
        return ColorRGBA(r=float(r), g=float(g), b=float(b), a=1.0)

    def callback(self, detection_msg):
        markers = MarkerArray()
        i = -1
        objects: List[DetectedObject2D] = detection_msg.objects

        for detection in objects:
            i += 1
            class_name = self.id_to_name[detection.hypothesis.class_id]
            marker = Marker()
            marker.header.frame_id = detection_msg.header.frame_id
            marker.header.stamp = detection_msg.header.stamp
            marker.ns = class_name
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.lifetime = Duration(sec=1)

            camera_info = self.camera_info_dict.get(detection_msg.sensor.frame_id)
            if not camera_info:
                self.logger.warning(
                    f"No CameraInfo found for frame {detection_msg.sensor.frame_id}"
                )
                continue

            # Compute the ray based on camera intrinsics and detection center
            ray_start = Point(
                x=detection_msg.sensor_pose.position.x,
                y=detection_msg.sensor_pose.position.y,
                z=detection_msg.sensor_pose.position.z,
            )
            print(class_name)
            ray_ends = self.calculate_rays(
                camera_info,
                detection.centre_x,
                detection.centre_y,
                detection.bbox_width,
                detection.bbox_height,
                detection_msg.sensor_pose,
            )
            if ray_ends is None:
                continue

            marker.points = [
                ray_start,
                ray_ends[0],
                ray_ends[1],
                ray_start,
                ray_ends[3],
                ray_ends[2],
                ray_start,
                ray_ends[0],
                ray_ends[1],
                ray_ends[3],
                ray_ends[2],
                ray_ends[0],
            ]
            marker.scale = Vector3(x=0.05, y=0.05, z=0.05)  # Line thickness
            marker.color = self.get_color(class_name)
            marker.ns = f"{detection_msg.sensor.sensor_name}/{class_name}"
            markers.markers.append(marker)

        self.publisher.publish(markers)

    def calculate_rays(
        self, camera_info: CameraInfo, u: int, v: int, w: int, h: int, sensor_pose
    ):
        # Extract the camera intrinsics
        fx = camera_info.p[0]
        fy = camera_info.p[5]
        cx = camera_info.p[2]
        cy = camera_info.p[6]

        # Calculate the four corners of the bounding box
        bbox_corners = [
            (u - w // 2, v + h // 2),  # Bottom-left corner
            (u + w // 2, v + h // 2),  # Bottom-right corner
            (u - w // 2, v - h // 2),  # Top-left corner
            (u + w // 2, v - h // 2),  # Top-right corner
        ]

        rays_end_points = []
        ts = []

        for i, (corner_u, corner_v) in enumerate(bbox_corners):
            # Normalized image coordinates for each corner
            x_norm = (corner_u - cx) / fx
            y_norm = (corner_v - cy) / fy

            # Assuming the ray is projected out of the camera at a distance of 1 unit in the z direction
            ray_dir_camera = np.array([x_norm, y_norm, 1.0])

            # Rotate the ray direction to align with the sensor_pose
            # Assuming sensor_pose is a Pose message
            q = sensor_pose.orientation
            rotation_matrix = self.quaternion_to_rotation_matrix(q)
            ray_dir_world = rotation_matrix @ ray_dir_camera

            camera_forward = rotation_matrix @ np.array([0, 0, 1])
            camera_forward[2] = 0

            # Calculate the intersection with the ground plane (z = 0)
            t = -sensor_pose.position.z / ray_dir_world[2]
            ts.append(t)
            # if i == 0:
            #     # check if direction is same as camera forward direction
            #     if np.dot(ray_dir_world, camera_forward) < 0:
            #         return None
            if len(ts) > 1:
                t = ts[i - 2]
            ray_end = Point(
                x=sensor_pose.position.x + ray_dir_world[0] * t,
                y=sensor_pose.position.y + ray_dir_world[1] * t,
                z=sensor_pose.position.z + ray_dir_world[2] * t,
                # z=0.0  # Since we are intersecting with the ground plane
            )

            rays_end_points.append(ray_end)

        return rays_end_points

    @staticmethod
    def quaternion_to_rotation_matrix(q):
        # Convert quaternion to rotation matrix
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        return np.array(
            [
                [
                    1 - 2 * qy * qy - 2 * qz * qz,
                    2 * qx * qy - 2 * qz * qw,
                    2 * qx * qz + 2 * qy * qw,
                ],
                [
                    2 * qx * qy + 2 * qz * qw,
                    1 - 2 * qx * qx - 2 * qz * qz,
                    2 * qy * qz - 2 * qx * qw,
                ],
                [
                    2 * qx * qz - 2 * qy * qw,
                    2 * qy * qz + 2 * qx * qw,
                    1 - 2 * qx * qx - 2 * qy * qy,
                ],
            ]
        )


def main(args=None):
    rclpy.init(args=args)
    node = DetectedObject2DArrayVisNode()
    rclpy.spin(node)
    # node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
