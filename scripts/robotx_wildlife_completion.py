#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from bb_robotx_msgs.srv import ComputeCompletion  # replace with your package name
import numpy as np


class PoseCompletionService(Node):
    def __init__(self):
        super().__init__("pose_completion_service")
        self.pose_sequence = []
        self.tolerance = self.declare_parameter("tolerance", 1.0).get_parameter_value().double_value
        self.completed_poses = set()
        self.active = False

        # Initialize service
        self.service = self.create_service(
            ComputeCompletion, "/robotx24/compute_completion", self.handle_compute_completion)

        # Subscription to the Odometry topic for current position
        self.subscription = self.create_subscription(
            Odometry, "/asv4/nav/world", self.odom_callback, 10)
        self.get_logger().info("Pose completion service is ready.")

    def handle_compute_completion(self, req, res):
        # Load or reset poses
        self.pose_sequence = req.poses
        self.active = req.active
        self.completed_poses = set()
        self.get_logger().info(f"Received {len(self.pose_sequence)} poses for completion calculation.")
        res.success = True
        return res

    def odom_callback(self, msg):
    # Get the current position from the Odometry message
        if self.active:
            current_position = np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ])
            self.get_logger().info(f"Current position: x={current_position[0]}, y={current_position[1]}, z={current_position[2]}")

            # Check completion starting from the beginning to maintain continuity
            for idx, pose_stamped in enumerate(self.pose_sequence):
                if idx in self.completed_poses:
                    continue  # Skip already completed poses

                # Access the nested pose position properly
                pose_position = np.array([
                    pose_stamped.pose.position.x,
                    pose_stamped.pose.position.y,
                    pose_stamped.pose.position.z
                ])


                distance = np.linalg.norm(current_position - pose_position)
                
                # Log the next target position before checking distance
                self.get_logger().info(f"Next target position: x={pose_position[0]}, y={pose_position[1]}, z={pose_position[2]}, distance={distance}")

                # If within tolerance, mark this and all previous poses as completed
                if distance < self.tolerance:
                    self.completed_poses.update(range(0, idx + 1))
                    break  # Stop checking further poses after finding the nearest incomplete one

            # Calculate percentage completion
            completion_percentage = (len(self.completed_poses) / len(self.pose_sequence)) * 100.0 if len(self.pose_sequence) > 0 else 0.0
            self.get_logger().info(f"Current completion: {completion_percentage:.2f}%")




def main(args=None):
    rclpy.init(args=args)
    service = PoseCompletionService()
    rclpy.spin(service)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
