#!/usr/bin/env python3
"""
Labelling 3d tracks with 2d sensors
"""
import numpy as np
from collections import Counter, defaultdict, deque
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from sensor_msgs.msg import CameraInfo
from bb_perception_msgs.msg import (
    DetectedObject2DArray,
    DetectedObject3DArray,
    DetectedObject3D,
    DetectedObject2D,
    ObjectClassification,
)
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import do_transform_pose
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from std_msgs.msg import Header
from typing import List, Tuple
from shapely.geometry import box
from scipy.optimize import linear_sum_assignment
from copy import deepcopy

np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


class DetectedObject3DLabelingNode(Node):
    def __init__(self):
        super().__init__("detected_object_3d_labelling_node")

        # Declare parameters
        self.declare_parameter("detection_2d_topic", "/asv4/vision/detections_2d")
        self.declare_parameter(
            "detection_3d_topic", "/asv4/vision/lidar_small_objects/dets_3d/filtered"
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
            "output_labeled_topic", "/asv4/vision/lidar_small_objects/dets_3d/labelled"
        )

        # Get parameters
        self.detection_2d_topic = (
            self.get_parameter("detection_2d_topic").get_parameter_value().string_value
        )
        self.detection_3d_topic = (
            self.get_parameter("detection_3d_topic").get_parameter_value().string_value
        )
        self.camera_info_topics = (
            self.get_parameter("camera_info_topics")
            .get_parameter_value()
            .string_array_value
        )
        self.output_labeled_topic = (
            self.get_parameter("output_labeled_topic")
            .get_parameter_value()
            .string_value
        )

        # Set up subscribers
        self.camera_info_dict = {}
        for topic in self.camera_info_topics:
            self.create_subscription(CameraInfo, topic, self.camera_info_callback, 10)

        self.create_subscription(
            DetectedObject2DArray,
            self.detection_2d_topic,
            self.detection_2d_callback,
            10,
        )
        self.create_subscription(
            DetectedObject3DArray,
            self.detection_3d_topic,
            self.detection_3d_callback,
            10,
        )

        self.publisher = self.create_publisher(
            DetectedObject3DArray, self.output_labeled_topic, 10
        )

        # self.inflate_width = 1.5
        self.inflate_width = 0.3
        self.inflate_height = 0.6
        self.bbox_inflate_factor = 1.5
        self.latest_3d_detections = (None, None)
        self.timeout_duration = Duration(seconds=5.0)
        self.track_identities = defaultdict(Counter)

        self.tf_buffer = Buffer(Duration(seconds=15), self)
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

    def camera_info_callback(self, camera_info: CameraInfo):
        self.camera_info_dict[camera_info.header.frame_id] = camera_info

    def detection_3d_callback(self, detection_3d_msg: DetectedObject3DArray):
        self.latest_3d_detections = (
            Time.from_msg(detection_3d_msg.header.stamp),
            detection_3d_msg,
        )

    def detection_2d_callback(self, detection_2d_msg: DetectedObject2DArray):
        # expire 3d detections
        if self.latest_3d_detections[0] is not None and (
            Time.from_msg(detection_2d_msg.header.stamp) - self.latest_3d_detections[0]
            > self.timeout_duration
        ):
            self.latest_3d_detections = (None, None)
        if self.latest_3d_detections[0] is None:
            return
        next_dets = self.latest_3d_detections[1]
        camera_info = self.camera_info_dict.get(detection_2d_msg.sensor.frame_id)
        if not camera_info:
            self.get_logger().warn(
                f"No CameraInfo for frame {detection_2d_msg.sensor.frame_id}"
            )
            return

        labeled_3d_objects = DetectedObject3DArray()
        labeled_3d_objects.header = next_dets.header
        labeled_3d_objects.name = next_dets.name
        labeled_3d_objects.source = next_dets.source
        labeled_3d_objects.sensor_pose = next_dets.sensor_pose

        cost_matrix = (
            np.ones((len(next_dets.objects), len(detection_2d_msg.objects))) * 1e9
        )
        # self.get_logger().info(f"Num objects: {len(next_dets.objects)} {len(detection_2d_msg.objects)}")
        for i, obj_3d in enumerate(next_dets.objects):
            projected_2d_points, dist = self.project_3d_to_2d(
                camera_info,
                detection_2d_msg.header,
                detection_2d_msg.sensor_pose,
                obj_3d,
            )

            if not projected_2d_points or dist < 0:
                continue

            # best_overlap = 0
            # best_class_id = -1

            for j, det_2d in enumerate(detection_2d_msg.objects):
                det_bbox = self.get_bbox_from_2d_detection(det_2d)
                proj_bbox = self.get_bbox_from_2d_points(projected_2d_points)
                overlap = self.compute_overlap(det_bbox, proj_bbox)
                cost_matrix[i][j] = (
                    1 / (overlap + 1e-9) * dist
                )  # prioritize nearer objects

                # if overlap > best_overlap:
                #     best_overlap = overlap
                #     best_class_id = det_2d.hypothesis.class_id

            # if best_overlap > 0.02:
            #     self.track_identities[obj_3d.hypothesis.track_id][best_class_id] += 1
        if min(cost_matrix.shape) == 0:
            return
        if cost_matrix.min() > 1 / (0.01 + 1e-9):
            return
        assignments = linear_sum_assignment(cost_matrix)

        for row, col in zip(assignments[0], assignments[1]):
            track_id = next_dets.objects[row].hypothesis.track_id
            class_id = detection_2d_msg.objects[col].hypothesis.class_id
            self.track_identities[track_id][class_id] += 1

        # Label 3D objects based on the highest count
        for obj_3d in next_dets.objects:
            obj = deepcopy(obj_3d)
            track_counts = self.track_identities[obj.hypothesis.track_id]
            if track_counts:
                most_common_class, count = track_counts.most_common(1)[0]
                second_most_common = (
                    sorted(track_counts.values(), reverse=True)[1]
                    if len(track_counts) > 1
                    else 0
                )
                if count > 3 and count > 1.2 * second_most_common:
                    obj.hypothesis.class_id = most_common_class
                else:
                    self.get_logger().info(
                        f"top label not good enough {count > 3} {count} {second_most_common}",
                        throttle_duration_sec=2.0,
                    )
                    obj.hypothesis.class_id = 0

                total_counts = sum(track_counts.values())
                class_map = defaultdict(float)
                for k, v in track_counts.items():
                    class_map[k] += v / total_counts
                total_prob_sum = sum(class_map.values())
                for k, v in class_map.items():
                    obj.hypothesis.classes.append(
                        ObjectClassification(class_id=k, score=v / total_prob_sum)
                    )
            labeled_3d_objects.objects.append(obj)
        labeled_3d_objects.header.stamp = detection_2d_msg.header.stamp
        self.publisher.publish(labeled_3d_objects)

    def get_bbox_from_2d_detection(self, detection: DetectedObject2D) -> box:
        x = detection.centre_x
        y = detection.centre_y
        width = detection.bbox_width * self.bbox_inflate_factor
        height = detection.bbox_height * self.bbox_inflate_factor
        return box(x, y, x + width, y + height)

    def get_bbox_from_2d_points(self, points: List[Tuple[float, float]]) -> box:
        xs, ys = zip(*points)
        return box(min(xs), min(ys), max(xs), max(ys))

    def project_3d_to_2d(
        self,
        camera_info: CameraInfo,
        header: Header,
        sensor_pose: Pose,  # sensor pose is the camera's position and orientation in the world frame
        obj_3d: DetectedObject3D,
    ):
        # Extract the camera intrinsics
        fx = camera_info.p[0]
        fy = camera_info.p[5]
        cx = camera_info.p[2]
        cy = camera_info.p[6]

        # Extract the 3D object's centroid position
        # x_3d = obj_3d.hypothesis.kinematics.pose_with_covariance.pose.position.x
        # y_3d = obj_3d.hypothesis.kinematics.pose_with_covariance.pose.position.y
        # z_3d = obj_3d.hypothesis.kinematics.pose_with_covariance.pose.position.z
        transformed_pose = None
        try:
            # Lookup the transform from the 3D object's frame to the sensor (camera) frame
            transform = self.tf_buffer.lookup_transform(
                camera_info.header.frame_id,
                obj_3d.hypothesis.kinematics.header.frame_id,
                rclpy.time.Time(),
                Duration(seconds=0.1),
            )
            # self.get_logger().info(f"Transform lookup success: {transform} {camera_info.header.frame_id} {obj_3d.hypothesis.kinematics.header.frame_id}")

            # Create a PoseStamped object for transformation
            pose_stamped = PoseStamped()
            pose_stamped.header = obj_3d.hypothesis.kinematics.header
            pose_stamped.pose = obj_3d.hypothesis.kinematics.pose_with_covariance.pose

            transformed_pose = do_transform_pose(pose_stamped.pose, transform)

        except Exception as e:
            self.get_logger().warn(f"Transform lookup failed: {e}")
            return [], -1

        # Check if the object is in front of the camera
        if transformed_pose.position.z <= 0:
            return [], -1

        # self.get_logger().info(f"pt: {point_in_sensor_frame} {header.frame_id}")

        # Project the 3D point to 2D
        u = (transformed_pose.position.x * fx / transformed_pose.position.z) + cx
        v = (transformed_pose.position.y * fy / transformed_pose.position.z) + cy

        # Calculate the 2D bounding box size based on the distance from the camera
        bbox_width = (
            (
                max(
                    obj_3d.hypothesis.shape.dimensions.x,
                    obj_3d.hypothesis.shape.dimensions.y,
                )
                + self.inflate_width
            )
            / transformed_pose.position.z
            * fx
        )
        bbox_height = (
            (obj_3d.hypothesis.shape.dimensions.z + self.inflate_height * 2)
            / transformed_pose.position.z
            * fy
        )
        bbox_width = max(bbox_width, 5)
        bbox_height = max(bbox_height, 5)

        # Define the corners of the 2D bounding box
        bbox_corners = [
            (u - bbox_width / 2, v - bbox_height / 2),  # Bottom-left corner
            (u + bbox_width / 2, v - bbox_height / 2),  # Bottom-right corner
            (u + bbox_width / 2, v + bbox_height / 2),  # Top-right corner
            (u - bbox_width / 2, v + bbox_height / 2),  # Top-left corner
        ]

        return bbox_corners, np.linalg.norm(
            [transformed_pose.position.x, transformed_pose.position.y]
        )

    def compute_overlap(self, bbox1: box, bbox2: box) -> float:
        intersection = bbox1.intersection(bbox2).area
        union = bbox1.union(bbox2).area
        return intersection / union if union > 0 else 0


def main(args=None):
    rclpy.init(args=args)
    node = DetectedObject3DLabelingNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
