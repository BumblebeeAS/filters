#!/usr/bin/env python3
from collections import defaultdict
from pathlib import Path
from operator import attrgetter
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from bb_perception_msgs.msg import (
    DetectedObject3D,
    DetectedObject3DArray,
    ObjectHypothesis,
)
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3, Quaternion
from ml_detector.schema_validator import get_config, load_schema

# from motrackers import SORT
from bb_filters.sort_3d import SORT3D
from rclpy.node import Node
from transforms3d.euler import quat2euler, euler2quat


class TrackerFilter(Node):
    def __init__(self):
        super().__init__("motracker_iou_tracker_node")
        self.declare_parameter(
            "dets_3d_topic", "/asv4/vision/lidar_small_objects/dets_3d"
        )
        self.declare_parameter(
            "filtered_topic", "/asv4/vision/lidar_small_objects/dets_3d/filtered"
        )
        self.declare_parameter("max_lost", 5)
        self.declare_parameter("dist_threshold", 1.5)
        self.max_lost = (
            self.get_parameter("max_lost").get_parameter_value().integer_value
        )
        self.dist_threshold = (
            self.get_parameter("dist_threshold").get_parameter_value().double_value
        )
        self.declare_parameter("objects_config", "robotx.yaml")
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
        self.name_to_id = {v: k for k, v in self.id_to_name.items()}

        self.dets_3d_topic = (
            self.get_parameter("dets_3d_topic").get_parameter_value().string_value
        )
        self.filtered_topic = (
            self.get_parameter("filtered_topic").get_parameter_value().string_value
        )
        self.sensor_pose = None
        self.odom_sub = self.create_subscription(
            Odometry, "/asv4/nav/world", self.odom_callback, 10
        )
        self.tracked_objects_pub = self.create_publisher(
            DetectedObject3DArray, self.filtered_topic, 10
        )

        self.buffer = self.dist_threshold
        self.tracker = SORT3D(
            max_lost=self.max_lost,
            # min_detection_confidence=0.2,
            # max_detection_confidence=1.0,
            # iou_threshold=0.001,
            dist_threshold=self.dist_threshold,
            # tracker_output_format="visdrone_challenge",
        )
        self.latest_header = None
        self.min_age = 5
        self.track_counts = defaultdict(lambda: np.zeros(6))
        self.detection_sub = self.create_subscription(
            DetectedObject3DArray,
            self.dets_3d_topic,
            self.detection_callback,
            10,
        )
        self.obj_heights = defaultdict(lambda: (0, 0))
        self.obj_z = defaultdict(lambda: (0, 0))
        self.latest_header = None

    def odom_callback(self, msg: Odometry):
        self.sensor_pose = msg.pose.pose

    def detection_callback(self, msg: DetectedObject3DArray):
        self.latest_header = msg.header
        bboxes, confidences, ids = [], [], []

        for det in msg.objects:
            if det.hypothesis.class_id not in self.id_to_name:
                continue

            pose = det.hypothesis.kinematics.pose_with_covariance.pose
            if self.sensor_pose is not None:
                distance = np.sqrt(
                    (pose.position.x - self.sensor_pose.position.x) ** 2
                    + (pose.position.y - self.sensor_pose.position.y) ** 2
                    + (pose.position.z - self.sensor_pose.position.z) ** 2
                )

                # Filter out objects beyond 50 meters
                if distance > 50.0:
                    continue

            x = pose.position.x - det.hypothesis.shape.dimensions.x / 2
            y = pose.position.y - det.hypothesis.shape.dimensions.y / 2
            z = pose.position.z - det.hypothesis.shape.dimensions.z / 2
            dx = det.hypothesis.shape.dimensions.x + self.buffer
            dy = det.hypothesis.shape.dimensions.y + self.buffer
            dz = det.hypothesis.shape.dimensions.z + self.buffer
            yaw = quat2euler(attrgetter("w", "x", "y", "z")(pose.orientation))[2]

            bboxes.append([x, y, z, dx, dy, dz, yaw])
            prob = det.hypothesis.probability
            if prob == 0:
                prob = 0.6
            confidences.append(prob)
            ids.append(det.hypothesis.class_id)
            (h, ct) = self.obj_heights[det.hypothesis.class_id]
            self.obj_heights[det.hypothesis.class_id] = (
                h + det.hypothesis.shape.dimensions.z,
                ct + 1,
            )
            (h, ct) = self.obj_z[det.hypothesis.class_id]
            self.obj_z[det.hypothesis.class_id] = (
                h + det.hypothesis.kinematics.pose_with_covariance.pose.position.z,
                ct + 1,
            )
        if len(msg.objects) > 0:
            self.latest_header = msg.objects[0].hypothesis.kinematics.header

        # Update tracker with filtered bboxes
        tracked_objects = self.tracker.update(
            np.array(bboxes), np.array(confidences), np.array(ids)
        )

        # Publish tracked objects (remaining code as is)

        output = DetectedObject3DArray()
        output.header = msg.header

        for track in tracked_objects:
            (
                frame,
                tid,
                bb_x,
                bb_y,
                bb_z,
                bb_dx,
                bb_dy,
                bb_dz,
                bb_yaw,
                confidence,
                class_id,
                trunc,
                occ,
            ) = track
            tracked_obj_msg = DetectedObject3D()
            tracked_obj_msg.hypothesis.track_id = tid
            print(class_id)
            tracked_obj_msg.hypothesis.class_id = int(class_id)
            tracked_obj_msg.hypothesis.kinematics.header = self.latest_header

            width, length, height = (
                max(bb_dx - self.buffer, 0.2),
                max(bb_dy - self.buffer, 0.2),
                max(bb_dz - self.buffer, 0.2),
            )
            tracked_obj_msg.hypothesis.kinematics.pose_with_covariance.pose.position.x = (
                bb_x + width / 2
            )
            tracked_obj_msg.hypothesis.kinematics.pose_with_covariance.pose.position.y = (
                bb_y + length / 2
            )
            tracked_obj_msg.hypothesis.kinematics.pose_with_covariance.pose.position.z = (
                bb_z + height / 2
            )
            tracked_obj_msg.hypothesis.mode = ObjectHypothesis.MODE_TRACKED
            tracked_obj_msg.hypothesis.probability = confidence
            tracked_obj_msg.hypothesis.shape.category = 0
            tracked_obj_msg.hypothesis.shape.dimensions = Vector3(
                x=width, y=length, z=height
            )
            # z = self.obj_z[class_id]
            # tracked_obj_msg.hypothesis.kinematics.pose_with_covariance.pose.position.z = (
            #     np.sign(z[0]) * h[0] / h[1] / 2
            # )
            q = euler2quat(0, 0, bb_yaw)
            # self.get_logger().info(f"post yaw: {np.rad2deg(bb_yaw)}")
            tracked_obj_msg.hypothesis.kinematics.pose_with_covariance.pose.orientation = Quaternion(
                **dict(zip("wxyz", q))
            )
            output.objects.append(tracked_obj_msg)
        output.header = msg.header
        output.name = msg.name
        output.source = msg.source
        output.sensor_pose = msg.sensor_pose
        self.tracked_objects_pub.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = TrackerFilter()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
