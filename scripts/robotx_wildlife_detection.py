#!/usr/bin/env python3
"""
Wildlife Detection Node for ROS2
"""

from collections import defaultdict
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import time
from rclpy.node import Node
import rclpy
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from bb_perception_msgs.msg import (
    DetectedObject3D,
    DetectedObject3DArray,
    DetectorSource,
    ObjectHypothesis,
)
from bb_robotx_msgs.srv import ConfigureWildlifeTask
from ml_detector.schema_validator import get_config, load_schema
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import numpy as np
from bb_robotx_msgs.msg import WildlifePoses


class EncirclementTask(Node):
    def __init__(self):
        super().__init__('encirclement_task')
        objects_schema_path = (
            Path(get_package_share_directory("ml_detector"))
            / "configs"
            / "objects_schema.json"
        )
        self.objects_schema = load_schema(objects_schema_path)
        self.declare_parameter("objects_config", "robotx.yaml")
        self.objects_config = get_config(
            Path(get_package_share_directory("ml_detector"))
            / "configs"
            / "objects"
            / self.get_parameter("objects_config").get_parameter_value().string_value,
            self.objects_schema,
        )
        self.id_to_name = {
            obj["label"]: obj["name"] for obj in self.objects_config["objects"]
        }
        self.name_to_id = {v: k for k, v in self.id_to_name.items()}
        self.green_buoy_id = self.name_to_id["green_cylinder"]
        self.red_buoy_id = self.name_to_id["red_cylinder"]
        self.black_bouy_id = self.name_to_id["black_cylinder"]
        self.green_sphere_id = self.name_to_id["green_sphere"]
        self.red_sphere_id = self.name_to_id["red_sphere"]
        self.blue_sphere_id = self.name_to_id["blue_sphere"]
        self.unknown_id = self.name_to_id["unknown"]
        self.running = False
        self.subscription = self.create_subscription(
            DetectedObject3DArray,
            # "/asv4/vision/lidar_small_objects/dets_3d/labelled",
            # "/asv4/vision/detections_2d/projected/filtered",
            "/asv4/vision/detections_2d/projected",
            # "/asv4/vision/detections_2d/fixed",
            self.detected_objects_callback,
            1,
        )
        # Rest of your initialization code ...

        self.buoy_pose_history = defaultdict(list)  # History of buoy poses
        self.all_buoy_poses = defaultdict(list)  # All buoy poses
        self.tracking_duration = 10.0  # Track for 10 seconds
        self.tolerance = 2.0  # Tolerance for clustering poses (meters)

        self.tf_publisher = self.create_publisher(
            PoseStamped, "/wildlife_buoy_pose", 1)
        self.start_time = time.time()
        self.latest = None
        self.wildlife_type = None
        self.correct_ids = []
        self.gate_task_config_service = self.create_service(
            ConfigureWildlifeTask,
            "/robotx24/configure_wildlife_task",
            self.configure_wildlife_task_callback,
        )

        self.all_wildlife_publisher = self.create_publisher(
            WildlifePoses, "/all_wildlife_tf", 1
        )

        self.curr_pose = None

        


        # Subscription to the Odometry topic for current position
        self.curr_position = self.create_subscription(
            Odometry, "/asv4/nav/world", self.odom_callback, 10)
        
    def publish_closest_wildlife_poses(self):
        # Create WildlifePoses message
        wildlife_poses_msg = WildlifePoses()

        # Check current position
        if self.curr_pose is None:
            self.get_logger().warn("Current position not yet available.")
            return

        # Closest poses for each wildlife type
        closest_poses = {'python': None, 'iguana': None, 'manatee': None}
        min_distances = {'python': float('inf'), 'iguana': float('inf'), 'manatee': float('inf')}

        for class_id, poses in self.all_buoy_poses.items():
            for pose in poses:
                # Convert pose to numpy array
                pose_np = np.array(pose)
                distance = np.linalg.norm(pose_np - self.curr_pose)

                # Check if it's the closest for the wildlife type
                print("class id", class_id,"check: ", pose, distance)
                if class_id in [self.red_sphere_id, self.red_buoy_id] and distance < min_distances['python']:
                    print("python","check: ", pose, distance)
                    min_distances['python'] = distance
                    closest_poses['python'] = pose
                elif class_id in [self.blue_sphere_id, self.black_bouy_id] and distance < min_distances['manatee']:
                    print("manatee","check: ", pose, distance)
                    min_distances['manatee'] = distance
                    closest_poses['manatee'] = pose
                elif class_id in [self.green_sphere_id, self.green_buoy_id] and distance < min_distances['iguana']:
                    print("iguana","checkt: ", pose, distance)   
                    min_distances['iguana'] = distance
                    closest_poses['iguana'] = pose

        # Assign closest poses to message if found
        for wildlife_type, pose in closest_poses.items():
            if pose is not None:
                pose_stamped = PoseStamped()
                pose_stamped.header.stamp = self.get_clock().now().to_msg()
                pose_stamped.header.frame_id = "map"
                pose_stamped.pose.position.x = pose[1]
                pose_stamped.pose.position.y = pose[0]
                pose_stamped.pose.position.z = 0.0
                if wildlife_type == 'python':
                    wildlife_poses_msg.python = pose_stamped
                elif wildlife_type == 'manatee':
                    wildlife_poses_msg.manatee = pose_stamped
                elif wildlife_type == 'iguana':
                    wildlife_poses_msg.iguana = pose_stamped

        # Publish message
        self.all_wildlife_publisher.publish(wildlife_poses_msg)
        self.get_logger().info("Published closest wildlife poses for each type.")
    
    def odom_callback(self, msg):
        if not self.running:
            return 
        # Update current position based on Odometry message
        self.curr_pose = np.array([
            msg.pose.pose.position.y,
            msg.pose.pose.position.x,
        ])
        # self.get_logger().info(f"Current position updated to: x={self.curr_pose[0]}, y={self.curr_pose[1]}")

    def configure_wildlife_task_callback(
        self, req: ConfigureWildlifeTask.Request, res: ConfigureWildlifeTask.Response
    ):
        if not req.active:
            self.running = False
            self.buoy_pose_history = defaultdict(list)
            self.all_buoy_poses = defaultdict(list)
            self.tracking_duration = 10.0
            self.tolerance = 2.0
            self.wildlife_type = req.wildlife_type
            res.success = True
            print("Wildlife task deactivated.")
            return res
        self.running = True
        res.success = True
        print(f"Wildlife task configured with type: {req.wildlife_type}")
        self.wildlife_type = req.wildlife_type
        if req.wildlife_type == "python":
            self.correct_ids = [self.red_sphere_id, self.red_buoy_id]
        elif req.wildlife_type == "manatee":
            self.correct_ids = [self.blue_sphere_id, self.black_bouy_id]
        elif req.wildlife_type == "iguana":
            self.correct_ids = [self.green_sphere_id, self.green_buoy_id]

        return res

    def detected_objects_callback(self, msg):
        if not self.running:
            return  # Skip processing if the service call deactivated the task

        current_time = time.time()

        # Clear history if 10 seconds have passed
        if current_time - self.start_time > self.tracking_duration:
            self.determine_most_likely_pose()
            self.publish_closest_wildlife_poses()
            self.buoy_pose_history.clear()
            self.start_time = current_time

        if self.latest is not None:
            # Publish the most likely transform
            self.tf_publisher.publish(self.latest)

            # Logging the published pose
            self.get_logger().info(
                f"Published most likely transform for track_id {1}: {self.latest}"
            )
        for det in msg.objects:

            if det.hypothesis.class_id in [self.red_sphere_id, self.red_buoy_id, self.green_sphere_id, self.green_buoy_id, self.blue_sphere_id, self.black_bouy_id]:
                self.all_buoy_poses[det.hypothesis.class_id].append(
                    (det.hypothesis.kinematics.pose_with_covariance.pose.position.x, det.hypothesis.kinematics.pose_with_covariance.pose.position.y)
                )

            if det.hypothesis.class_id not in self.correct_ids:
                continue

            pose = det.hypothesis.kinematics.pose_with_covariance.pose
            track_id = det.hypothesis.track_id

            # Track the position (x, y) of the buoy
            self.buoy_pose_history[track_id].append(
                (pose.position.x, pose.position.y))
            
            print(det.hypothesis.class_id, "wildlife", self.wildlife_type, "detected at", pose.position.x, pose.position.y)

        
        # Publish closest wildlife poses
        # self.publish_closest_wildlife_poses()


    def cluster_poses(self, poses):
        """Clusters poses that are within a given tolerance and returns them with counts."""
        clusters = []
        for pose in poses:
            found_cluster = False
            for cluster in clusters:
                if self.is_within_tolerance(cluster[0], pose):
                    cluster[1] += 1
                    found_cluster = True
                    break
            if not found_cluster:
                # Initialize new cluster with count 1
                clusters.append([pose, 1])
        return clusters

    def is_within_tolerance(self, pose1, pose2):
        """Checks if two poses are within a tolerance range."""
        return (
            abs(pose1[0] - pose2[0]) <= self.tolerance and
            abs(pose1[1] - pose2[1]) <= self.tolerance
        )

    def create_pose_message(self, track_id, pose):
        """Creates a PoseStamped message for the buoy's most likely pose."""
        pose_msg = PoseStamped()
        pose_msg.header = Header()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"  # Replace with appropriate frame of reference

        # Set position from the pose array (assuming pose[0] is y, pose[1] is x)
        pose_msg.pose.position.x = pose[1]
        pose_msg.pose.position.y = pose[0]
        pose_msg.pose.position.z = 0.0  # Assuming a 2D plane (z=0)

        # Identity orientation (no rotation)
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = 0.0
        pose_msg.pose.orientation.w = 1.0

        return pose_msg

    def determine_most_likely_pose(self):
        geometry_msg = PoseStamped()
        # To store (track_id, most_likely_pose, distance)
        all_most_likely_poses = []

        if self.curr_pose is None:
            self.get_logger().warn("Current position not yet available.")
            return

        # Iterate over all tracked buoy poses
        for track_id, poses in self.buoy_pose_history.items():
            # Cluster the poses for the current buoy
            clustered_poses = self.cluster_poses(poses)
            # Find the closest pose to the current position
            closest_pose = min(clustered_poses, key=lambda cluster: np.linalg.norm(np.array(cluster[0]) - self.curr_pose))
            most_likely_pose = closest_pose[0]  # Get the pose with the lowest distance
            count = closest_pose[1]  # Count remains as is

            # Store the most likely pose with its count
            all_most_likely_poses.append((track_id, most_likely_pose, count))

        # Find the most likely pose across all buoys (highest count)
        if all_most_likely_poses:
            track_id, most_likely_pose, _ = max(all_most_likely_poses, key=lambda x: x[2])

            # Create a transform message for the most likely pose
            pose = self.create_pose_message(track_id, most_likely_pose)
            self.latest = pose

        else:
            self.get_logger().info("No buoys detected in the last 10 seconds.")


def main(args=None):
    rclpy.init(args=args)
    wildlife_detection = EncirclementTask()
    rclpy.spin(wildlife_detection)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
