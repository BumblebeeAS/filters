#!/usr/bin/env python3

import logging
from pathlib import Path
from typing import Dict, List
from operator import attrgetter
import numpy as np
import rclpy
import tf2_ros
from ament_index_python.packages import get_package_share_directory
from bb_perception_msgs.msg import (
    DetectedObject2D,
    DetectedObject2DArray,
    DetectedObject3D,
    DetectedObject3DArray,
)
from geometry_msgs.msg import Point, Pose
from message_filters import Subscriber, TimeSynchronizer
from ml_detector.helpers.log import RclLogHandler
from ml_detector.schema_validator import get_config, load_schema
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo
from transforms3d.quaternions import quat2mat


class DetectedObject2DProjection(Node):
    def __init__(self):
        super().__init__("detected_object_2d_projection_node")
        self.declare_parameter("input_detections_topics", ["/sim/detections_2d"])
        self.declare_parameter(
            "camera_info_topics",
            [
                "/camera_info",
            ],
        )
        self.declare_parameter(
            "output_detections_topic", "/uav2/projected_3d"
        )
        self.declare_parameter("objects_config", "drone.yaml")
        self.declare_parameter("publish_tf", False)

        self.input_detections_topics = (
            self.get_parameter("input_detections_topics")
            .get_parameter_value()
            .string_array_value
        )
        self.declare_parameter("inflate_height", 0.0)
        self.inflate_height = (
            self.get_parameter("inflate_height").get_parameter_value().double_value
        )
        self.declare_parameter("ground_z", -0.2)
        self.ground_z = (
            self.get_parameter("ground_z").get_parameter_value().double_value
        )
        self.declare_parameter("dist_limit", 60.0)
        self.dist_limit = (
            self.get_parameter("dist_limit").get_parameter_value().double_value
        )
        self.camera_info_topics = (
            self.get_parameter("camera_info_topics")
            .get_parameter_value()
            .string_array_value
        )
        self.output_detections_topic = (
            self.get_parameter("output_detections_topic")
            .get_parameter_value()
            .string_value
        )
        self.publish_tf = (
            self.get_parameter("publish_tf").get_parameter_value().bool_value
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
        self.estimated_object_bottom = {
            obj["label"]: obj.get("z", 0.0) for obj in self.objects_config["objects"]
        }
        self.estimated_sizes = {}
        for obj in self.objects_config["objects"]:
            if "shape" in obj and "dimensions" in obj["shape"]:
                self.estimated_sizes[obj["label"]] = (
                    max(
                        obj["shape"]["dimensions"]["x"], obj["shape"]["dimensions"]["y"]
                    ),
                    obj["shape"]["dimensions"]["z"],
                )

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.publisher = self.create_publisher(
            DetectedObject3DArray, self.output_detections_topic, 10
        )

        self.tf_broadcast = tf2_ros.TransformBroadcaster(self)

        self.camera_info_subscribers = [
            Subscriber(self, CameraInfo, topic, qos_profile=qos)
            for topic in self.camera_info_topics
        ]

        self.camera_info_dict: Dict[str, CameraInfo] = {}
        for subscriber in self.camera_info_subscribers:
            subscriber.registerCallback(self.camera_info_callback)

        self.detection_subscribers = [
            Subscriber(self, DetectedObject2DArray, topic, qos_profile=qos)
            for topic in self.input_detections_topics
        ]
        self.time_sync = TimeSynchronizer([*self.detection_subscribers], 10)
        self.time_sync.registerCallback(self.callback)

    def camera_info_callback(self, camera_info: CameraInfo):
        self.camera_info_dict[camera_info.header.frame_id] = camera_info

    def callback(self, *detection_msgs):
        detected_objects_3d_array = DetectedObject3DArray()
        detected_objects_3d_array.header.stamp = detection_msgs[0].header.stamp
        detected_objects_3d_array.header.frame_id = detection_msgs[0].header.frame_id

        for detection_msg in detection_msgs:
            objects: List[DetectedObject2D] = detection_msg.objects
            for detection in objects:
                detected_object_3d = DetectedObject3D()
                detected_object_3d.hypothesis.classes = detection.hypothesis.classes
                detected_object_3d.hypothesis.class_id = detection.hypothesis.class_id
                detected_object_3d.hypothesis.probability = (
                    detection.hypothesis.probability
                )
                detected_object_3d.hypothesis.track_id = detection.hypothesis.track_id
                detected_object_3d.hypothesis.mode = detection.hypothesis.mode

                camera_info = self.camera_info_dict.get(detection_msg.sensor.frame_id)
                if not camera_info:
                    self.logger.warning(
                        f"No CameraInfo found for frame {detection_msg.sensor.frame_id}"
                    )
                    continue

                # Compute the ray end points based on camera intrinsics and detection center
                object_bottom_z = self.estimated_object_bottom.get(
                    detection.hypothesis.class_id, 0.0
                )

                estimated_size = self.estimated_sizes.get(
                    detection.hypothesis.class_id, (0.0, 0.0)
                )
                # get max estimate dist if dimensions provided
                estimated_dists_from_dimension = []
                if estimated_size[0] > 0:
                    estimated_dists_from_dimension.append(
                        estimated_size[0] * 2 / detection.bbox_width * camera_info.p[0]
                    )
                if estimated_size[1] > 0:
                    estimated_dists_from_dimension.append(
                        estimated_size[1] * 2 / detection.bbox_height * camera_info.p[5]
                    )

                ray_ends = self.calculate_rays(
                    camera_info,
                    detection.centre_x,
                    detection.centre_y,
                    detection.bbox_width,
                    detection.bbox_height * (1.0 + self.inflate_height),
                    detection_msg.sensor_pose,
                    object_bottom_z,
                    np.mean(estimated_dists_from_dimension)
                )
                if len(ray_ends) == 0:
                    continue

                # Populate the DetectedObject3D with the calculated rays
                detected_object_3d.hypothesis.shape.dimensions.x = np.linalg.norm(
                    np.array([ray_ends[1].x, ray_ends[1].y, ray_ends[1].z])
                    - np.array([ray_ends[0].x, ray_ends[0].y, ray_ends[0].z])
                )
                detected_object_3d.hypothesis.shape.dimensions.y = np.linalg.norm(
                    np.array([ray_ends[2].x, ray_ends[2].y, ray_ends[2].z])
                    - np.array([ray_ends[0].x, ray_ends[0].y, ray_ends[0].z])
                )
                detected_object_3d.hypothesis.shape.dimensions.z = (
                    1.0
                )
                estimated_pose = Pose()
                for i in range(4):
                    estimated_pose.position.x += ray_ends[i].x
                    estimated_pose.position.y += ray_ends[i].y
                    estimated_pose.position.z += ray_ends[i].z
                estimated_pose.position.x /= 4
                estimated_pose.position.y /= 4
                estimated_pose.position.z /= 4
                detected_object_3d.hypothesis.kinematics.pose_with_covariance.pose = (
                    estimated_pose
                )
                detected_object_3d.hypothesis.kinematics.header.frame_id = (
                    detection_msg.header.frame_id
                )
                detected_object_3d.hypothesis.kinematics.header.stamp = (
                    detection_msg.header.stamp
                )
                detected_objects_3d_array.objects.append(detected_object_3d)
        self.publisher.publish(detected_objects_3d_array)

    def calculate_rays(
        self,
        camera_info: CameraInfo,
        u: int,
        v: int,
        w: int,
        h: int,
        sensor_pose,
        object_bottom_z,
        max_estimate_dist
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
        self.logger.warning(f"Bounding Box corners {bbox_corners}")

        rays_end_points = []
        ts = []

        for i, (corner_u, corner_v) in enumerate(bbox_corners):
            # Normalized image coordinates for each corner
            x_norm = (corner_u - cx) / fx
            y_norm = (corner_v - cy) / fy


            # Assuming the ray is projected out of the camera at a distance of 1 unit in the z direction
            ray_dir_camera = np.array([x_norm, y_norm, 1.0])
            self.logger.warning(f"ray_dir_camera {ray_dir_camera}")
            # Rotate the ray direction to align with the sensor_pose
            # Assuming sensor_pose is a Pose message
            q = sensor_pose.orientation
            rotation_matrix = quat2mat(attrgetter("w", "x", "y", "z")(q))
            ray_dir_world = rotation_matrix @ ray_dir_camera


            t = (
                -sensor_pose.position.z
            )
            self.logger.warning(f"t value {t}")

            # t = min(t, max_estimate_dist) # doesn't work well
            ts.append(t)
            ray_end = Point(
                x=sensor_pose.position.x + ray_dir_world[0] * t,
                y=sensor_pose.position.y + ray_dir_world[1] * t,
                z=sensor_pose.position.z + ray_dir_world[2] * t,
                # z=0.0  # Since we are intersecting with the ground plane
            )

            rays_end_points.append(ray_end)
        return rays_end_points


def main(args=None):
    rclpy.init(args=args)
    node = DetectedObject2DProjection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
