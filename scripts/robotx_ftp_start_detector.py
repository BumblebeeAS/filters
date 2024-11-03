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
from std_msgs.msg import Bool
from transforms3d.euler import euler2quat
import math
import yaml
from ament_index_python.packages import get_package_share_directory
import os

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
        
        # Load the YAML file
        package_share_directory = get_package_share_directory('software-asv-3')
        yaml_file_path = os.path.join(package_share_directory, 'estimates', 'current_course.yaml')
        
        with open(yaml_file_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Parse the channel start and end colors
        self.channel_start_color = config['channel_start_color']
        self.channel_end_color = config['channel_end_color']

        # Log colours
        self.get_logger().info(f'Channel start color: {self.channel_start_color}')
        self.get_logger().info(f'Channel end color: {self.channel_end_color}')
        
        # Assign the values based on the channel start color
        channel_entrance = config['estimates']['channel_entrance'] # red on left
        channel_exit = config['estimates']['channel_exit']

        self.start_pose_str = f"map;{channel_entrance['x']};{channel_entrance['y']};0;{channel_entrance['yaw']}"
        self.end_pose_str = f"map;{channel_exit['x']};{channel_exit['y']};0;{channel_exit['yaw']}"
        
        # Create a subscriber to the light sequence topic
        self.subscription = self.create_subscription(
            LightSequence,
            '/robotx24/light_sequence',
            self.listener_callback,
        )
        
        # Create publishers for start and end poses
        self.start_publisher = self.create_publisher(PoseStamped, '/robotx24/ftp_start', 10)
        self.end_publisher = self.create_publisher(PoseStamped, '/robotx24/ftp_end', 10)

        # Create publisher for boolean of is_red_on_left
        self.is_reversed_publisher = self.create_publisher(Bool, '/robotx24/ftp_is_reversed', 10)

    def listener_callback(self, msg):
        # Process the incoming light sequence message
        first_light = msg.first
        self.get_logger().info(f'Received first light: {first_light}')
        
        is_red_on_left = first_light == self.channel_start_color # is_forward
        is_reversed = not is_red_on_left

        self.get_logger().info(f'Is reversed (green on left)): {is_reversed}')

        # Get PoseStamped messages for start and end poses
        start_pose_msg = create_pose_stamped_from_string(self.start_pose_str)
        end_pose_msg = create_pose_stamped_from_string(self.end_pose_str)

        if (is_reversed):
            # Swap the start and end poses
            start_pose_msg, end_pose_msg = end_pose_msg, start_pose_msg
        
        
        # Set the timestamp
        current_time = self.get_clock().now().to_msg()
        start_pose_msg.header.stamp = current_time
        end_pose_msg.header.stamp = current_time
        
        # Publish the PoseStamped messages
        self.start_publisher.publish(start_pose_msg)
        self.end_publisher.publish(end_pose_msg)
        self.get_logger().info('Published start and end poses to /robotx24/ftp_start and /robotx24/ftp_end')

        # Publish the boolean of is_red_on_left
        self.is_red_on_left_publisher.publish(Bool(data=is_reversed))
        self.get_logger().info('Published is_reversed to /robotx24/ftp_is_reversed')

def main(args=None):
    rclpy.init(args=args)
    node = FTPStartDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()