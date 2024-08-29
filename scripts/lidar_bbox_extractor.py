#!/usr/bin/env python3
import os
import cv2
import rclpy
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from bb_perception_msgs.msg import DetectedObject3DArray
import transforms3d as t3d
import time
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import TransformStamped


class BBoxExtractor(Node):
    def __init__(self):
        super().__init__("bbox_extractor")

        # Initialize parameters
        self.declare_parameter("camera_info_topic", "/asv4/left_cam/camera_info")
        self.declare_parameter("camera_image_topic", "/asv4/left_cam/image_rect_color")
        self.declare_parameter(
            "detected_objects_topic",
            "/asv4/vision/lidar_small_objects/dets_3d/filtered",
        )
        self.declare_parameter("output_dir", "output")
        self.declare_parameter("save_interval", 0.5)  # default save interval

        # Initialize subscriptions
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.get_parameter("camera_info_topic").value,
            self.camera_info_callback,
            10,
        )
        self.image_sub = self.create_subscription(
            Image,
            self.get_parameter("camera_image_topic").value,
            self.image_callback,
            10,
        )
        self.detected_objects_sub = self.create_subscription(
            DetectedObject3DArray,
            self.get_parameter("detected_objects_topic").value,
            self.detected_objects_callback,
            10,
        )

        # Initialize camera parameters and cv_bridge
        self.camera_matrix = None
        self.dist_coeffs = None
        self.bridge = CvBridge()

        # Output directory and save interval
        self.output_dir = self.get_parameter("output_dir").value
        os.makedirs(self.output_dir, exist_ok=True)
        self.save_interval = self.get_parameter("save_interval").value
        self.last_save_time = 0.0

        self.current_image = None
        self.current_image_header = None

        # Initialize TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def camera_info_callback(self, msg):
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        self.current_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.current_image_header = msg.header

    def detected_objects_callback(self, msg):
        current_time = time.time()
        if self.current_image is None or self.camera_matrix is None:
            self.get_logger().error("Camera info or image not received yet")
            return

        if current_time - self.last_save_time >= self.save_interval:
            self.last_save_time = current_time
            for obj in msg.objects:
                bbox = self.process_object(obj, msg.header.frame_id)
                if bbox is not None:
                    self.save_bbox_crop(bbox, obj.id)
                else:
                    self.get_logger().error("Error processing object")

    def process_object(self, obj, detected_objects_frame_id):
        try:
            # Look up transform from the detected object's frame to the camera frame
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                self.current_image_header.frame_id,
                detected_objects_frame_id,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.5),
            )

            # Extract rotation and translation from the transform
            trans = transform.transform.translation
            rot = transform.transform.rotation

            # Convert quaternion to rotation matrix
            R_sensor_to_camera = t3d.quaternions.quat2mat([rot.w, rot.x, rot.y, rot.z])

            # Combine rotation and translation into transformation matrix
            T_sensor_to_camera = t3d.affines.compose(
                [trans.x, trans.y, trans.z], R_sensor_to_camera, np.ones(3)
            )

            # Transform object pose to the camera frame
            object_pose = obj.hypothesis.kinematics.pose_with_covariance.pose
            object_translation = [
                object_pose.position.x,
                object_pose.position.y,
                object_pose.position.z,
            ]

            object_rotation = [
                object_pose.orientation.w,
                object_pose.orientation.x,
                object_pose.orientation.y,
                object_pose.orientation.z,
            ]

            # Convert quaternion to rotation matrix
            R_object = t3d.quaternions.quat2mat(object_rotation)

            # Combine rotation and translation into transformation matrix
            T_object = t3d.affines.compose(object_translation, R_object, np.ones(3))

            # Transform the object pose to the camera frame
            T_object_in_camera = np.dot(T_sensor_to_camera, T_object)

            # Extract the transformed translation (3D point)
            point_3d_in_camera = T_object_in_camera[:3, 3]

            # Check if the object is in front of the camera
            if point_3d_in_camera[2] <= 0:
                return None

            # Project 3D point to 2D using camera intrinsics
            point_2d, _ = cv2.projectPoints(
                point_3d_in_camera.reshape(-1, 3),
                (0, 0, 0),
                (0, 0, 0),
                self.camera_matrix,
                self.dist_coeffs,
            )

            # Check if the projected point is within the image bounds (FOV)
            x, y = point_2d[0][0]
            if not (
                0 <= x < self.current_image.shape[1]
                and 0 <= y < self.current_image.shape[0]
            ):
                return None

            return point_2d[0][0] if point_2d is not None else None

        except Exception as e:
            self.get_logger().error(f"Error processing object: {str(e)}")
            return None

    def save_bbox_crop(self, bbox, obj_id):
        x, y = int(bbox[0]), int(bbox[1])
        # Define crop size, e.g., 50x50 pixels around the point
        crop_size = 50
        x1, y1 = max(0, x - crop_size), max(0, y - crop_size)
        x2, y2 = min(self.current_image.shape[1], x + crop_size), min(
            self.current_image.shape[0], y + crop_size
        )

        # Crop and save the image
        cropped_image = self.current_image[y1:y2, x1:x2]
        output_path = os.path.join(
            self.output_dir, f"{obj_id}_{self.current_image_header.stamp.sec}.png"
        )
        if cropped_image.size == 0:
            self.get_logger().error(f"Empty cropped image {x1} {y1} {x2} {y2}")
            return
        cv2.imwrite(output_path, cropped_image)
        self.get_logger().info(f"Saved cropped image: {output_path}")


def main(args=None):
    rclpy.init(args=args)
    node = BBoxExtractor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
