#!/usr/bin/env python3.10
"""
RobotX FTP Start Detection Node for ROS2

This ROS2 node subscribes to a light sequence message topic and publishes a pose and orientation to the `/robotx24/ftp_start` topic.

Subscriptions:
- `/robotx24/light_sequence` (bb_robotx_msgs/msg/LightSequence): Input light sequence message.

Publications:
- `/robotx24/ftp_start` (geometry_msgs/PoseStamped): Output pose and orientation message.
"""

import rclpy
from rclpy.node import Node
from bb_robotx_msgs.msg import LightSequence
from geometry_msgs.msg import PoseStamped
from transforms3d.euler import euler2quat
import math

# Global variables for start poses x,y,z,yaw
BLUE_START = "map;92;160;0;0"
RED_START = "map;190;205;0;180"
GREEN_START = "map;92;160;0;0"

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
    pose_msg.pose.orientation.x = q[0]
    pose_msg.pose.orientation.y = q[1]
    pose_msg.pose.orientation.z = q[2]
    pose_msg.pose.orientation.w = q[3]
    
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
        
        # Create a publisher to the /robotx24/ftp_start topic
        self.publisher = self.create_publisher(PoseStamped, '/robotx24/ftp_start', 10)
        
    def listener_callback(self, msg):
        # Process the incoming light sequence message
        light_sequence = (msg.first, msg.second, msg.third)
        self.get_logger().info(f'Received light sequence: {light_sequence}')
        
        # Determine the start pose based on the light sequence
        if light_sequence == (LightSequence.BLUE, LightSequence.RED, LightSequence.GREEN):
            pose_msg = create_pose_stamped_from_string(BLUE_START)
        elif light_sequence == (LightSequence.RED, LightSequence.GREEN, LightSequence.BLUE):
            pose_msg = create_pose_stamped_from_string(RED_START)
        elif light_sequence == (LightSequence.GREEN, LightSequence.BLUE, LightSequence.RED):
            pose_msg = create_pose_stamped_from_string(GREEN_START)
        else:
            self.get_logger().warn('Unknown light sequence')
            return
        
        # Set the timestamp
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        
        # Publish the PoseStamped message
        self.publisher.publish(pose_msg)
        self.get_logger().info('Published pose and orientation to /robotx24/ftp_start')

def main(args=None):
    rclpy.init(args=args)
    node = FTPStartDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()