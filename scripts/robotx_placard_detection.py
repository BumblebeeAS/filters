#!/usr/bin/env python3

"""
Placard Detection Node using CV Approach

TODO: 
(1) Publish Pose of detected placards
(2) Allows for other colours to be detected

Publications:
- `/asv4/robotx/placard/debug` (sensor_msgs/CompressedImage): Debug image showing placard positions.
- [not implemented] `/asv4/vision/placard_detections` (bb_perception_msgs/DetectedObject3DArray): Detected placards.

Subscriptions:
- `/asv4/front_cam/image_rect_color` (sensor_msgs/Image): Front Camera Rectified Image Topic.

"""
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import TransformStamped, PoseStamped, Vector3
import cv2
import numpy as np
import tf2_ros
from transforms3d.euler import euler2mat, euler2quat
from cv_bridge import CvBridge

# Define the HSV threshold for red color, can add bounds for other colours
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([40, 255, 200])
lower_red2 = np.array([170, 120, 50])
upper_red2 = np.array([180, 255, 120])

class PlacardPoseNode(Node):
    def __init__(self):
        super().__init__('placard_pose_node')
        self.image_subscription = self.create_subscription(
            CompressedImage,
            '/asv4/left_cam/image_rect_color/compressed',
            self.image_callback,
            10
        )
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            '/asv4/left_cam/camera_info',
            self.camera_info_callback,
            10
        )
        self.placard_pose_pub = self.create_publisher(PoseStamped, '/placard_pose_cv', 10) # not implemented
        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=20.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.placard_detect_pub = self.create_publisher(CompressedImage, '/asv4/robotx/placard/debug/compressed', 10)
        self.camera_info_received = False

    def camera_info_callback(self, msg):
        if (self.camera_info_received):
            return

        self.camera_info_received = True
        # capture projection matrix and other camera info
        self.projection_matrix = np.array(msg.p).reshape(3, 4)
        self.camera_matrix = np.array(msg.k).reshape(3, 3)


    def image_callback(self, msg):
        if self.camera_info_received:
            # Process the image using the camera info
            image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            self.compute_pose(image)
            pass
        else:
            self.get_logger().warn('Camera info not received yet')
        
        
    def compute_pose(self, image):
        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Mask the red regions in the image
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Find contours of the red square
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



        # extract 4 point simplified contour
        try:
            # Find the largest contour among the red contours
            contour = max(red_contours, key=cv2.contourArea)
            
            epsilon = 0.1 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # Check if the contour is a quadrilateral
            if (len(approx) == 4 and cv2.contourArea(contour) > 100):
                # order points by angle relative to centroid
                centroid = np.mean(approx, axis=0)
                # angle between line from centroid to point and vector (1, 0)
                angles = np.arctan2(approx[:, 0, 1] - centroid[0, 1], approx[:, 0, 0] - centroid[0, 0]) % (2 * np.pi)
                approx = approx[np.argsort(angles), :, :]
                br, bl, tl, tr = approx
                left_height = np.linalg.norm(tl - bl)
                right_height = np.linalg.norm(tr - br)

                # Draw the center of the placard
                cv2.circle(image, (centroid[0].astype(int)), 5, (255, 255, 255), -1)

                # Obtain the distance between the camera and placard
                placard_size_real = 0.59  # Real-world placard size in m

                # dist_coeffs = np.array(calibration_data['distortion_coefficients']['data']).reshape(4, 1)
                dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
                
                fx = self.camera_matrix[0, 0]
                fy = self.camera_matrix[1, 1]  # Focal length in pixels

                left_dist = (placard_size_real * fy) / left_height
                right_dist = (placard_size_real * fy) / right_height
                distance_v1 = (left_dist + right_dist) / 2
                # get yaw of placard relative to camera by considering distance difference from camera plane
                try:
                    yaw_v1 = np.arcsin((left_dist - right_dist) / placard_size_real)
                except Exception as e:
                    self.get_logger().error('Error computing yaw: %s' % str(e))
                    return

                average_y_midpoint = centroid[0, 1]
                z_shift_v1 = distance_v1
                y_shift_v1 = (average_y_midpoint - self.camera_matrix[1, 2]) / fy * distance_v1
                x_shift_v1 = (centroid[0, 0] - self.camera_matrix[0, 2]) / fx * distance_v1

                rot = euler2mat(0, yaw_v1, 0, 'szyx')
                trans = np.array([x_shift_v1, y_shift_v1, z_shift_v1])
                cambody2placard = np.eye(4)
                cambody2placard[:3, :3] = rot
                cambody2placard[:3, 3] = trans
                axis_length = 0.3  # Define the axis length
                axis_points_3d = np.float32([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length], [0, 0, 0]])
                # axis_points_3d_cam = np.dot(cambody2placard[:3, :3], axis_points_3d.T).T + world2cambody[:3, 3]
                axis_points_2d, _ = cv2.projectPoints(axis_points_3d, cambody2placard[:3, :3], cambody2placard[:3, 3], self.camera_matrix, dist_coeffs)
                axis_points_2d = np.int32(axis_points_2d).reshape(-1, 2)
                image = cv2.line(image, tuple(axis_points_2d[3].ravel()), tuple(axis_points_2d[2].ravel()), (255,0,0), 3) #b
                image = cv2.line(image, tuple(axis_points_2d[3].ravel()), tuple(axis_points_2d[1].ravel()), (0,255,0), 3) #g
                image = cv2.line(image, tuple(axis_points_2d[3].ravel()), tuple(axis_points_2d[0].ravel()), (0,0,255), 3) #r
                corner_points = np.float32([[-placard_size_real / 2, -placard_size_real / 2, 0],
                                            [placard_size_real / 2, -placard_size_real / 2, 0],
                                            [placard_size_real / 2, placard_size_real / 2, 0],
                                            [-placard_size_real / 2, placard_size_real / 2, 0]])
                corner_points_2d, _ = cv2.projectPoints(corner_points, cambody2placard[:3, :3], cambody2placard[:3, 3], self.camera_matrix, dist_coeffs)
                corner_points_2d = np.int32(corner_points_2d).reshape(-1, 2)
                image = cv2.polylines(image, [corner_points_2d], True, (255, 0, 0), 2)
                big_hole_center_offset = np.array([-0.15, -0.75, 0.0])
                big_hole_width = 0.5
                small_hole_center_offset = np.array([0.3, -0.6, 0.0])
                small_hole_width = 0.25
                for (offset, width) in [(big_hole_center_offset, big_hole_width),
                                        (small_hole_center_offset, small_hole_width)]:
                    hole_points = np.float32([[-width / 2, -width / 2, 0],
                                            [width / 2, -width / 2, 0],
                                            [width / 2, width / 2, 0],
                                            [-width / 2, width / 2, 0]]) + offset
                    hole_points_2d, _ = cv2.projectPoints(hole_points, cambody2placard[:3, :3], cambody2placard[:3, 3], self.camera_matrix, dist_coeffs)
                    hole_points_2d = np.int32(hole_points_2d).reshape(-1, 2)
                    image = cv2.polylines(image, [hole_points_2d], True, (0, 255, 0), 2)

                # Display the distance and z-axis shift
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, f"X: {x_shift_v1:.2f}m, Y: {y_shift_v1:.2f}m, Z: {z_shift_v1:.2f}m", (10, 70), font, 1.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(image, f'Yaw: {np.degrees(yaw_v1):.2f} degrees', (10, 100), font, 1.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(image, f'left_height: {left_height:.2f}, right_height: {right_height:.2f}', (10, 130), font, 1.5, (0, 255, 0), 1, cv2.LINE_AA)
                
                self.get_logger().info('Placard detected')
                self.get_logger().info(f"X: {x_shift_v1:.2f}m, Y: {y_shift_v1:.2f}m, Z: {z_shift_v1:.2f}m")
                self.get_logger().info(f'Yaw: {np.degrees(yaw_v1):.2f} degrees')
                self.get_logger().info(f'left_height: {left_height:.2f}, right_height: {right_height:.2f}')
                self.get_logger().info(f"left_dist: {left_dist:.2f}, right_dist: {right_dist:.2f}, distance: {distance_v1:.2f}")

                # Publish the image with the detected placard
                image_msg = self.bridge.cv2_to_compressed_imgmsg(image)

                self.placard_detect_pub.publish(image_msg)
                # Obtain pose of the placard relative to the camera and publish it
                # try: 
                #     self.get_logger().info('Publishing placard pose')
                #     placard_pose = PoseStamped()
                #     placard_pose.header.stamp = image_msg.header.stamp
                #     placard_pose.header.frame_id = "asv4/left_cam"
                #     placard_pose.pose.position.x = float(x_shift_v1)
                #     placard_pose.pose.position.y = float(y_shift_v1)
                #     placard_pose.pose.position.z = float(z_shift_v1)
                #     placard_pose.pose.orientation.x = quat[0]
                #     placard_pose.pose.orientation.y = quat[1]
                #     placard_pose.pose.orientation.z = quat[2]
                #     placard_pose.pose.orientation.w = quat[3]
                #     self.placard_pose_pub.publish(placard_pose)

                #     world_pose = self.tf_buffer.transform(
                #         placard_pose, "map_ned", Duration(seconds=0.05)
                #     )
                    
                #     t = TransformStamped()
                #     t.header = world_pose.header
                #     t.child_frame_id = "placard"
                #     t.transform.translation.x = float(world_pose.pose.position.x)
                #     t.transform.translation.y = float(world_pose.pose.position.y)
                #     t.transform.translation.z = 0.0
                #     t.transform.rotation = world_pose.pose.orientation

                #     self.tf_broadcaster.sendTransform(t)
                
                # except Exception as e:
                #     self.get_logger().error('Error publishing placard pose: %s' % str(e))


        except Exception as e:
            self.get_logger().error('Error processing placard: %s' % str(e))
            return


def main(args=None):
    rclpy.init(args=args)
    placard_pose_node = PlacardPoseNode()
    rclpy.spin(placard_pose_node)
    placard_pose_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()