#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import Quaternion
# from zed_interfaces.msg import ConfidenceMap, Detection2DArray
import cv2
from bb_perception_msgs.msg import DetectedObject2DArray, DetectedObject3DArray, DetectedObject3D, Shape
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge
import numpy as np
from transforms3d.quaternions import quat2mat
from transforms3d.euler import euler2quat
from operator import attrgetter
from bb_filters.min_bounding_rect import minimum_bounding_rectangle
from sklearn.cluster import DBSCAN

class ZEDTrackerNode(Node):
    def __init__(self):
        super().__init__('zed_tracker_node')
        self.depth_sub = Subscriber(self, Image, '/asv4/zed2i/zed_node/depth/depth_registered')
        self.confidence_sub = Subscriber(self, Image, '/asv4/zed2i/zed_node/confidence/confidence_map')
        self.image_sub = Subscriber(self, CompressedImage, '/asv4/zed2i/zed_node/left/image_rect_color/compressed')
        self.detection_sub = Subscriber(self, DetectedObject2DArray, '/asv4/vision/detections_2d_zed_left')
        self.detection_pub = self.create_publisher(DetectedObject3DArray, '/asv4/vision/detections_3d_zed_left', 10)
        self.debug_img_pub = self.create_publisher(Image, '/asv4/vision/debug_img_zed_tracker', 10)
        self.camera_info_sub = Subscriber(self, CameraInfo, '/asv4/zed2i/zed_node/left/camera_info')
        self.confidence_threshold = 0.5  # Initial confidence threshold
        self.declare_parameter('confidence_threshold', self.confidence_threshold)
        self.declare_parameter('debug_confidence', False)
        self.declare_parameter('debug_depth', False)
        self.declare_parameter('skip_pixels', 15)
        self.declare_parameter('mode_2d', False)

        self.dbscan = DBSCAN(eps=0.1, min_samples=5)
        
        self.skip_pixels = self.get_parameter('skip_pixels').value  # Skip every n pixels
        self.mode_2d = self.get_parameter('mode_2d').value
        self.add_on_set_parameters_callback(self.on_set_parameters)
        self.bridge = CvBridge()
        self.camera_info = None
        self.time_sync = ApproximateTimeSynchronizer(
            [   
                self.depth_sub,
                self.confidence_sub,
                self.image_sub,
                self.detection_sub,
                self.camera_info_sub
            ], 20, 0.01
        )
        self.time_sync.registerCallback(self.callback)



    def on_set_parameters(self, params):
        for param in params:
            if param.name == 'confidence_threshold':
                self.confidence_threshold = param.value.double_value
                self.get_logger().info(f"New confidence threshold: {self.confidence_threshold}")
            elif param.name == 'debug_confidence':
                self.debug_confidence = param.value.bool_value
                self.get_logger().info(f"Debug confidence: {self.debug_confidence}")
            elif param.name == 'debug_depth':
                self.debug_depth = param.value.bool_value
                self.get_logger().info(f"Debug depth: {self.debug_depth}")

    def get_color(self, i):
        return (i * 5 % 255, i * 10 % 255, i * 20 % 255)

    @staticmethod
    def convert_pose_to_matrix(pose):
        T = np.eye(4)
        T[0:3, 0:3] = quat2mat(attrgetter("w","x","y","z")(pose.orientation))
        T[0:3, 3] = attrgetter("x", "y", "z")(pose.position)
        return T

    def callback(self, depth, confidence, image, detections: DetectedObject2DArray, camera_info: CameraInfo):
        self.get_logger().info(f"{depth.header.stamp}")
        depth_img = self.bridge.imgmsg_to_cv2(depth)
        confidence_img = self.bridge.imgmsg_to_cv2(confidence)
        image_img = self.bridge.compressed_imgmsg_to_cv2(image)
        confidence_mask = confidence_img > self.confidence_threshold
        depth_img = depth_img * confidence_mask
        print(depth_img.shape)
        print(np.nanmax(depth_img), np.nanmin(depth_img))
        detection_array = detections.objects
        masked_depth = None

        combined_mask = np.zeros_like(depth_img).astype(np.uint8)
        camera_pose = detections.sensor_pose

        T_cam = ZEDTrackerNode.convert_pose_to_matrix(camera_pose)
        K_cam = np.array(camera_info.k).reshape(3, 3)

        detections_3d = DetectedObject3DArray()

        debug_img = np.zeros_like(depth_img).astype(np.uint8)
        for i, detection in enumerate(detection_array):
            
            poly = np.array(detection.contour).reshape(-1, 2).astype(np.int32)
            mask = np.zeros_like(depth_img).astype(np.uint8)
            cv2.fillPoly(mask, pts=[poly], color=(1))
            mask = mask.astype(np.uint16)
            # masked_depth = cv2.bitwise_and(image_img, image_img, mask=mask.astype(np.uint8))

            # masked_depth = np.multiply(depth_img, mask)
            # masked_depth[masked_depth == 0] = np.nan
            combined_mask = np.maximum(combined_mask, mask)
            # draw contours of color get_color(i) on debug_img
            print(i)
            cv2.drawContours(debug_img, [poly], -1, self.get_color(i), 2)
            cx, cy, w, h = detection.centre_x, detection.centre_y, detection.bbox_width, detection.bbox_height
            x1, x2, y1, y2 = map(int, (cx - w/2, cx + w/2, cy - h/2, cy + h/2))


            # 3dpoints = []

            projected_points = []
            for y in range(y1, y2, self.skip_pixels):
                for x in range(x1, x2, self.skip_pixels):
                    if combined_mask[y, x] > 0:
                        depth = depth_img[y, x]
                        # print(depth)
                        if not np.isnan(depth):
                            point_2d = np.array([x, y, 1])
                            point_3d = np.linalg.inv(K_cam) @ point_2d * depth
                            point_3d_homogeneous = np.append(point_3d, 1)
                            # print(point_3d_homogeneous)
                            point_3d_camera_frame = T_cam @ point_3d_homogeneous
                            projected_points.append(point_3d_camera_frame[:3])

            projected_points = np.array(projected_points)
            # perform dbscan cluster to remove noise
            if len(projected_points) < 5:
                continue
            clustered = self.dbscan.fit_predict(projected_points)
            projected_points = projected_points[clustered == 0]
            if len(projected_points) < 5:
                continue
            try:
                rect = minimum_bounding_rectangle(projected_points[:,:2])
            except:
                continue
            width = np.linalg.norm(rect[0] - rect[1])
            length = np.linalg.norm(rect[1] - rect[2])
            height = np.max(projected_points[:, 2]) - np.min(projected_points[:, 2])            

            det_3d = DetectedObject3D()
            det_3d.hypothesis = detection.hypothesis

            x_3d, y_3d, z_3d = np.mean(projected_points, axis=0)
            yaw = np.arctan2(rect[1][1] - rect[0][1], rect[1][0] - rect[0][0])
            setattr(
                det_3d.hypothesis.kinematics.pose_with_covariance.pose,
                "orientation",
                Quaternion(
                    **dict(
                        zip(
                            ["w", "x", "y", "z"],
                            euler2quat(0, 0, yaw)
                        )
                    )
                ))
            det_3d.hypothesis.kinematics.header = detections.header
            det_3d.hypothesis.kinematics.pose_with_covariance.pose.position.x = x_3d
            det_3d.hypothesis.kinematics.pose_with_covariance.pose.position.y = y_3d
            if not self.mode_2d:
                det_3d.hypothesis.kinematics.pose_with_covariance.pose.position.z = z_3d
            det_3d.hypothesis.shape.dimensions.x = width
            det_3d.hypothesis.shape.dimensions.y = length
            det_3d.hypothesis.shape.dimensions.z = height
            det_3d.hypothesis.kinematics.yaw_ambiguity_deg = 180.0
            detections_3d.objects.append(det_3d)
        
        
        detections_3d.header = detections.header
        detections_3d.source = detections.sensor
        detections_3d.sensor_pose = detections.sensor_pose



            # get depth values from mask
            # estimate 3d position given Pose from detections.sensor_pose
        
        masked_rgb = cv2.bitwise_and(image_img, image_img, mask=combined_mask.astype(np.uint8))
        # print(type(masked_depth))
        img = self.bridge.cv2_to_imgmsg(masked_rgb, encoding='bgr8')
        img.header.stamp = image.header.stamp
        self.debug_img_pub.publish(img)
        self.detection_pub.publish(detections_3d)
        # print(len(detection_array))

        
        

def main(args=None):
    rclpy.init(args=args)
    zed_tracker_node = ZEDTrackerNode()
    rclpy.spin(zed_tracker_node)
    zed_tracker_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()