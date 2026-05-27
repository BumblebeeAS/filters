#!/usr/bin/env python3
import os
import cv2
import rclpy
import numpy as np
from rclpy.duration import Duration
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from bb_perception_msgs.msg import DetectedObject3DArray
import transforms3d as t3d
import time
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import TransformStamped
import hashlib
import csv
import message_filters


class BBoxExtractor(Node):
    def __init__(self):
        super().__init__("bbox_extractor")

        # Initialize parameters
        self.declare_parameter("camera_info_topic", "/asv4/left_cam/camera_info")
        self.declare_parameter("camera_frame_override", "left_cam")
        self.declare_parameter("camera_image_topic", "/asv4/left_cam/image_rect_color")
        self.declare_parameter(
            "detected_objects_topic",
            "/asv4/vision/lidar_small_objects/dets_3d/filtered",
        )
        self.declare_parameter("output_dir", "~/auto_det/output")
        self.declare_parameter("save_interval", 0.5)  # default save interval

        # Initialize subscriptions
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.get_parameter("camera_info_topic").value,
            self.camera_info_callback,
            10,
        )
        image_sub = message_filters.Subscriber(
            self, Image, self.get_parameter("camera_image_topic").value
        )
        objects_sub = message_filters.Subscriber(
            self,
            DetectedObject3DArray,
            self.get_parameter("detected_objects_topic").value,
        )

        # Synchronize image and detected object messages
        ts = message_filters.ApproximateTimeSynchronizer(
            [image_sub, objects_sub], 10, slop=0.1
        )
        ts.registerCallback(self.sync_callback)

        # Initialize camera parameters and cv_bridge
        self.camera_matrix = None
        self.dist_coeffs = None
        self.bridge = CvBridge()

        # Output directory and save interval
        self.output_dir = self.get_parameter("output_dir").value
        os.makedirs(self.output_dir, exist_ok=True)

        self.images_path = os.path.join(self.output_dir, "images")
        self.crops_path = os.path.join(self.output_dir, "crops")
        os.makedirs(self.images_path, exist_ok=True)
        os.makedirs(self.crops_path, exist_ok=True)
        self.csv_path = os.path.join(self.output_dir, "data.csv")
        self.clusters_csv_path = os.path.join(self.output_dir, "clusters.csv")

        self.save_interval = self.get_parameter("save_interval").value
        self.last_save_time = 0.0

        self.current_image = None
        self.current_image_header = None

        # Initialize TF buffer and listener
        self.tf_buffer = Buffer(Duration(seconds=10))
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def camera_info_callback(self, msg):
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)

    def sync_callback(self, image_msg, msg):
        self.current_image = self.bridge.imgmsg_to_cv2(
            image_msg, desired_encoding="bgr8"
        )
        self.current_image_header = image_msg.header
        current_time = (
            image_msg.header.stamp.sec + image_msg.header.stamp.nanosec * 1e-9
        )
        if self.camera_matrix is None:
            self.get_logger().error("Camera info not received yet")
            return

        if current_time - self.last_save_time < self.save_interval:
            return

        self.last_save_time = current_time
        if len(msg.objects) == 0:
            self.get_logger().info("No objects detected")
            return
        records = []
        current_time = time.time()
        image = self.current_image
        IMAGE_UUID = hashlib.md5(cv2.imencode(".jpg", image)[1].tobytes()).hexdigest()
        width, height = image.shape[1], image.shape[0]
        records.append(
            {
                "stamp": current_time,
                "uuid": IMAGE_UUID,
                "width": width,
                "height": height,
            }
        )

        for i, obj in enumerate(msg.objects):
            bbox = self.process_object(obj, obj.hypothesis.kinematics.header.frame_id)
            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                cx, cy, w, h = x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2, x2 - x1, y2 - y1

                os.makedirs(
                    os.path.join(self.crops_path, f"cluster_{obj.hypothesis.track_id}"),
                    exist_ok=True,
                )
                self.save_bbox_crop(
                    bbox, f"cluster_{obj.hypothesis.track_id}/{IMAGE_UUID}_{i}"
                )
                with open(
                    self.clusters_csv_path, "a", newline="", encoding="utf-8"
                ) as clusters_file:
                    clusters_writer = csv.writer(clusters_file, delimiter="\t")
                    clusters_writer.writerow(
                        [IMAGE_UUID, cx, cy, w, h, obj.hypothesis.track_id, i]
                    )

        cv2.imwrite(os.path.join(self.images_path, f"{IMAGE_UUID}.jpg"), image)

        with open(self.csv_path, mode="a", newline="") as data_file:
            fieldnames = ["stamp", "uuid", "width", "height", "dataset_creation_date"]
            writer = csv.DictWriter(data_file, fieldnames=fieldnames)
            writer.writerows(records)

    def process_object(self, obj, detected_objects_frame_id):
        try:
            # Look up transform from the detected object's frame to the camera frame
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                self.current_image_header.frame_id,
                detected_objects_frame_id,
                rclpy.time.Time.from_msg(obj.hypothesis.kinematics.header.stamp),
                # rclpy.duration.Duration(seconds=1.0),
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
            if point_2d is None:
                return None
            x, y = point_2d[0][0]
            if not (
                0 <= x < self.current_image.shape[1]
                and 0 <= y < self.current_image.shape[0]
            ):
                return None

            est_width = (
                obj.hypothesis.shape.dimensions.z
                * self.camera_matrix[0][0]
                / point_3d_in_camera[2]
            )
            est_height = (
                max(                    
                    obj.hypothesis.shape.dimensions.y,
                    obj.hypothesis.shape.dimensions.x,
                )
                * self.camera_matrix[1][1]
                / point_3d_in_camera[2]
            )
            est_width = max(5, est_width)
            est_height = max(5, est_height)
            return [
                x - est_width / 2 * 2.0,
                y - est_height / 2 * 2.0,
                x + est_width / 2 * 2.0,
                y + est_height / 2 * 2.0,
            ]

        except Exception as e:
            self.get_logger().error(f"Error processing object: {str(e)}")
            return None

    def save_bbox_crop(self, bbox, obj_id):
        x1, y1, x2, y2 = map(int, bbox)
        # Define crop size, e.g., 50x50 pixels around the point
        # crop_size = 50
        # x1, y1 = max(0, x - crop_size), max(0, y - crop_size)
        # x2, y2 = min(self.current_image.shape[1], x + crop_size), min(
        #     self.current_image.shape[0], y + crop_size
        # )

        # Crop and save the image
        cropped_image = self.current_image[y1:y2, x1:x2]
        output_path = os.path.join(self.crops_path, f"{obj_id}.jpg")
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
