#!/usr/bin/env python3
from collections import defaultdict
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from bb_perception_msgs.msg import DetectedObject3D, DetectedObject3DArray, ObjectHypothesis
from geometry_msgs.msg import Vector3
from ml_detector.schema_validator import get_config, load_schema
from motrackers import SORT
from rclpy.node import Node


class TrackerFilter(Node):
    def __init__(self):
        super().__init__("motracker_iou_tracker_node")
        self.declare_parameter(
            "dets_3d_topic", "/asv4/vision/lidar_small_objects/dets_3d"
        )
        self.declare_parameter(
            "filtered_topic", "/asv4/vision/lidar_small_objects/dets_3d/filtered"
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
        self.tracked_objects_pub = self.create_publisher(
            DetectedObject3DArray, self.filtered_topic, 10
        )

        self.buffer = 1.5
        self.tracker = SORT(
            max_lost=5,
            # min_detection_confidence=0.2,
            # max_detection_confidence=1.0,
            iou_threshold=0.001,
            tracker_output_format="visdrone_challenge",
        )
        self.latest_header = None
        self.min_age = 3
        self.track_counts = defaultdict(lambda: np.zeros(6))
        self.detection_sub = self.create_subscription(
            DetectedObject3DArray,
            self.dets_3d_topic,
            self.detection_callback,
            10,
        )
        self.obj_heights = defaultdict(
            lambda: (0, 0)
        )
        self.obj_z = defaultdict(
            lambda: (0, 0)
        )

    def detection_callback(self, msg: DetectedObject3DArray):
        self.latest_header = msg.header
        bboxes, confidences, ids = [], [], []

        for det in msg.objects:
            if det.hypothesis.class_id not in self.id_to_name:
                continue

            x = det.hypothesis.kinematics.pose_with_covariance.pose.position.x
            y = det.hypothesis.kinematics.pose_with_covariance.pose.position.y
            dx = det.hypothesis.shape.dimensions.x + self.buffer
            dy = det.hypothesis.shape.dimensions.y + self.buffer
            bboxes.append([x - dx / y, y - dy / y, dx, dy])
            prob = det.hypothesis.probability
            if prob == 0:
                prob = 0.6
            confidences.append(prob)
            ids.append(det.hypothesis.class_id)
            (h, ct) = self.obj_heights[det.hypothesis.class_id]
            self.obj_heights[det.hypothesis.class_id] = (
                h + det.hypothesis.shape.dimensions.z,
                ct + 1)
            (h, ct) = self.obj_z[det.hypothesis.class_id]
            self.obj_z[det.hypothesis.class_id] = (
                h + det.hypothesis.kinematics.pose_with_covariance.pose.position.z,
                ct + 1)

        tracked_objects = self.tracker.update(
            np.array(bboxes), np.array(confidences), np.array(ids)
        )

        # Publish tracked objects
        output = DetectedObject3DArray()
        output.header = msg.header

        for track in tracked_objects:
            (
                frame,
                tid,
                bb_left,
                bb_top,
                bb_width,
                bb_height,
                confidence,
                class_id,
                trunc,
                occ,
            ) = track
            tracked_obj_msg = DetectedObject3D()
            tracked_obj_msg.hypothesis.track_id = tid
            print(class_id)
            tracked_obj_msg.hypothesis.class_id = int(class_id)
            tracked_obj_msg.hypothesis.kinematics.header = msg.objects[0].hypothesis.kinematics.header

            width, height = max(bb_width - self.buffer, 0.2), max(bb_height - self.buffer, 0.2)
            tracked_obj_msg.hypothesis.kinematics.pose_with_covariance.pose.position.x = (
                bb_left + width / 2
            )
            tracked_obj_msg.hypothesis.kinematics.pose_with_covariance.pose.position.y = (
                bb_top + height / 2
            )
            tracked_obj_msg.hypothesis.mode = ObjectHypothesis.MODE_TRACKED
            tracked_obj_msg.hypothesis.probability = confidence
            tracked_obj_msg.hypothesis.shape.category = 0
            h = self.obj_heights[class_id]
            tracked_obj_msg.hypothesis.shape.dimensions = Vector3(
                x=width,
                y=height,
                z=h[0] / h[1]
            )
            z = self.obj_z[class_id]
            tracked_obj_msg.hypothesis.kinematics.pose_with_covariance.pose.position.z = (
               np.sign(z[0]) * h[0] / h[1] / 2
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
