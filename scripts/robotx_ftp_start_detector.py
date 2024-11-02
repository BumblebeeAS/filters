#!/usr/bin/env python3
"""
RobotX FTP Start Detection Node for ROS2

This ROS2 node subscribes to a light sequence message topic and publishes a pose and orientation to the `/robotx24/ftp_start` topic.

Subscriptions:
- `/robotx24/light_sequence` (bb_robotx_msgs/msg/LightSequence): Input light sequence message.

Publications:
- `/robotx24/ftp_start` (geometry_msgs/PoseStamped): Output pose and orientation message.
- `/robotx24/ftp_end` (geometry_msgs/PoseStamped): Output pose and orientation message for the end location.
"""

import rclpy
from rclpy.node import Node
from bb_robotx_msgs.msg import LightSequence
from geometry_msgs.msg import PoseStamped
from transforms3d.euler import euler2quat
import math

# Global variables for start poses x,y,z,yaw
RED_START = "map;190;205;0;180"
GREEN_START = "map;92;160;0;0"
BLUE_START = None
# One of the start locations must be None

def create_pose_stamped_from_string(pose_str):
    parts = pose_str.split(';')
    pose_msg = PoseStamped()
    pose_msg.header.frame_id = parts[0]
    pose_msg.pose.position.x = float(parts[1])
    pose_msg.pose.position.y = float(parts[2])
    pose_msg.pose.position.z = float(parts[3])
    
    # Convert yaw to quaternion
    yaw = math.radians(float(parts[4]))
    q = euler2quat(0, 0, yaw)
    pose_msg.pose.orientation.w = q[0]
    pose_msg.pose.orientation.x = q[1]
    pose_msg.pose.orientation.y = q[2]
    pose_msg.pose.orientation.z = q[3]
    
    return pose_msg

class FTPStartDetectionNode(Node):
    def __init__(self):
        super().__init__('ftp_start_detection_node')
        
        # Create a subscriber to the light sequence topic
        self.subscription = self.create_subscription(
            LightSequence,
            '/robotx24/light_sequence',
            self.listener_callback,
            10
        )
        
        # Create publishers to the /robotx24/ftp_start and /robotx24/ftp_end topics
        self.start_publisher = self.create_publisher(PoseStamped, '/robotx24/ftp_start', 10)
        self.end_publisher = self.create_publisher(PoseStamped, '/robotx24/ftp_end', 10)
        
    def listener_callback(self, msg):
        # Process the incoming light sequence message
        first_light = msg.first
        self.get_logger().info(f'Received first light: {first_light}')
        
        # Determine the start pose based on the first light in the sequence
        if first_light == LightSequence.BLUE:
            start_pose_str = BLUE_START
        elif first_light == LightSequence.RED:
            start_pose_str = RED_START
        elif first_light == LightSequence.GREEN:
            start_pose_str = GREEN_START
        else:
            self.get_logger().warn('Unknown first light')
            return
        
        # Determine the end pose based on the remaining non-None start locations
        if start_pose_str == BLUE_START:
            end_pose_str = RED_START if RED_START is not None else GREEN_START
        elif start_pose_str == RED_START:
            end_pose_str = BLUE_START if BLUE_START is not None else GREEN_START
        elif start_pose_str == GREEN_START:
            end_pose_str = BLUE_START if BLUE_START is not None else RED_START
        
        # Create PoseStamped messages for start and end poses
        start_pose_msg = create_pose_stamped_from_string(start_pose_str)
        end_pose_msg = create_pose_stamped_from_string(end_pose_str)
        
        # Set the timestamp
        current_time = self.get_clock().now().to_msg()
        start_pose_msg.header.stamp = current_time
        end_pose_msg.header.stamp = current_time
        
        # Publish the PoseStamped messages
        self.start_publisher.publish(start_pose_msg)
        self.end_publisher.publish(end_pose_msg)
        self.get_logger().info('Published start and end poses to /robotx24/ftp_start and /robotx24/ftp_end')

def main(args=None):
    rclpy.init(args=args)
    node = FTPStartDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()