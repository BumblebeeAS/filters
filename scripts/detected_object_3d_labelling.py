#!/usr/bin/env python3
"""
Labelling 3d tracks with 2d sensors
"""
import numpy as np
from collections import Counter, defaultdict
import rclpy
from rclpy.node import Node
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

        self.inflate_width = 1.5
        self.latest_3d_detections = None
        self.track_identities = defaultdict(Counter)

        self.tf_buffer = Buffer(Duration(seconds=15), self)
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

    def camera_info_callback(self, camera_info: CameraInfo):
        self.camera_info_dict[camera_info.header.frame_id] = camera_info

    def detection_3d_callback(self, detection_3d_msg: DetectedObject3DArray):
        self.latest_3d_detections = detection_3d_msg

    def detection_2d_callback(self, detection_2d_msg: DetectedObject2DArray):
        if self.latest_3d_detections is None:
            return

        camera_info = self.camera_info_dict.get(detection_2d_msg.sensor.frame_id)
        if not camera_info:
            self.get_logger().warn(
                f"No CameraInfo for frame {detection_2d_msg.sensor.frame_id}"
            )
            return

        labeled_3d_objects = DetectedObject3DArray()
        labeled_3d_objects.header = self.latest_3d_detections.header
        labeled_3d_objects.name = self.latest_3d_detections.name
        labeled_3d_objects.source = self.latest_3d_detections.source
        labeled_3d_objects.sensor_pose = self.latest_3d_detections.sensor_pose

        cost_matrix = np.ones((len(self.latest_3d_detections.objects), len(detection_2d_msg.objects))) * 1e9
        for i, obj_3d in enumerate(self.latest_3d_detections.objects):
            projected_2d_points = self.project_3d_to_2d(
                camera_info,
                detection_2d_msg.header,
                detection_2d_msg.sensor_pose,
                obj_3d,
            )

            if not projected_2d_points:
                continue

            # best_overlap = 0
            # best_class_id = -1

            for j, det_2d in enumerate(detection_2d_msg.objects):
                det_bbox = self.get_bbox_from_2d_detection(det_2d)
                proj_bbox = self.get_bbox_from_2d_points(projected_2d_points)
                overlap = self.compute_overlap(det_bbox, proj_bbox)
                cost_matrix[i][j] = 1/(overlap+1e-9)

                # if overlap > best_overlap:
                #     best_overlap = overlap
                #     best_class_id = det_2d.hypothesis.class_id

            # if best_overlap > 0.02:
            #     self.track_identities[obj_3d.hypothesis.track_id][best_class_id] += 1
        if min(cost_matrix.shape) == 0:
            return
        if cost_matrix.min() > 1/(0.02 + 1e-9):
            return
        assignments = linear_sum_assignment(cost_matrix)

        for row, col in zip(assignments[0], assignments[1]):
            track_id = self.latest_3d_detections.objects[row].hypothesis.track_id
            class_id = detection_2d_msg.objects[col].hypothesis.class_id
            self.track_identities[track_id][class_id] += 1

        # Label 3D objects based on the highest count
        for obj_3d in self.latest_3d_detections.objects:
            track_counts = self.track_identities[obj_3d.hypothesis.track_id]
            if track_counts:
                most_common_class, count = track_counts.most_common(1)[0]
                second_most_common = (
                    sorted(track_counts.values(), reverse=True)[1]
                    if len(track_counts) > 1
                    else 0
                )
                if count > 3 and count > 1.5 * second_most_common:
                    obj_3d.hypothesis.class_id = most_common_class
                else:
                    # self.get_logger().info(f"top label not good enough {count > 4} {count} {second_most_common}")
                    obj_3d.hypothesis.class_id = 0

                total_counts = sum(track_counts.values())
                for k, v in track_counts.items():
                    obj_3d.hypothesis.classes.append(
                        ObjectClassification(class_id=k, score=v / total_counts)
                    )

            labeled_3d_objects.objects.append(obj_3d)

        self.publisher.publish(labeled_3d_objects)

    def get_bbox_from_2d_detection(self, detection: DetectedObject2D) -> box:
        x = detection.centre_x
        y = detection.centre_y
        width = detection.bbox_width
        height = detection.bbox_height
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
                Duration(seconds=0.1)
            )

            # Create a PoseStamped object for transformation
            pose_stamped = PoseStamped()
            pose_stamped.header = obj_3d.hypothesis.kinematics.header
            pose_stamped.pose = obj_3d.hypothesis.kinematics.pose_with_covariance.pose

            transformed_pose = do_transform_pose(pose_stamped.pose, transform)

        except Exception as e:
            self.get_logger().warn(f"Transform lookup failed: {e}")
            return []


        # Check if the object is in front of the camera
        if transformed_pose.position.z <= 0:
            return []

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
            (obj_3d.hypothesis.shape.dimensions.z + self.inflate_width)
            / transformed_pose.position.z
            * fy
        )

        # Define the corners of the 2D bounding box
        bbox_corners = [
            (u - bbox_width / 2, v - bbox_height / 2),  # Bottom-left corner
            (u + bbox_width / 2, v - bbox_height / 2),  # Bottom-right corner
            (u + bbox_width / 2, v + bbox_height / 2),  # Top-right corner
            (u - bbox_width / 2, v + bbox_height / 2),  # Top-left corner
        ]

        return bbox_corners



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
