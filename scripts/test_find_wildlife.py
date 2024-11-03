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
from ml_detector.schema_validator import get_config, load_schema
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped


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
        self.black_sphere = self.name_to_id["black_sphere"]
        self.unknown_id = self.name_to_id["unknown"]
        self.gate_id = self.name_to_id["gate"]
        self.subscription = self.create_subscription(
            DetectedObject3DArray,
            # "/asv4/vision/lidar_small_objects/dets_3d/labelled",
            # "/asv4/vision/detections_2d/projected/filtered",
            "/asv4/vision/detections_2d/projected",
            self.detected_objects_callback,
            1,
        )
        # Rest of your initialization code ...

        self.buoy_pose_history = defaultdict(list)  # History of buoy poses
        self.tracking_duration = 10.0  # Track for 10 seconds
        self.tolerance = 2.0  # Tolerance for clustering poses (meters)

        self.tf_publisher = self.create_publisher(
            TFMessage, "/wildlife_buoy_tf", 1)
        self.start_time = time.time()
        self.latest = None

    def detected_objects_callback(self, msg):
        current_time = time.time()

        # Clear history if 10 seconds have passed
        if current_time - self.start_time > self.tracking_duration:
            self.determine_most_likely_pose()
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
            print(det.hypothesis.class_id)
            # is_red_buoy = det.hypothesis.class_id == self.red_buoy_id
            # if not is_red_buoy:
            #     print("not red")
            #     continue
            # print("red")

            # is_black_sphere = det.hypothesis.class_id == self.black_sphere
            # if not is_black_sphere:
            #     print("not black")
            #     continue
            # print("black")

            is_green_buoy = det.hypothesis.class_id == self.green_buoy_id
            if not is_green_buoy:
                continue

            pose = det.hypothesis.kinematics.pose_with_covariance.pose
            track_id = det.hypothesis.track_id

            # Track the position (x, y) of the buoy
            self.buoy_pose_history[track_id].append(
                (pose.position.x, pose.position.y))

    def determine_most_likely_pose(self):
        tf_message = TFMessage()
        # To store (track_id, most_likely_pose, count)
        all_most_likely_poses = []

        # Iterate over all tracked buoy poses
        for track_id, poses in self.buoy_pose_history.items():
            # Cluster the poses for the current buoy
            clustered_poses = self.cluster_poses(poses)
            # Find the most likely pose for the current buoy
            most_likely_pose, count = max(clustered_poses, key=lambda x: x[1])
            # Store the most likely pose with its count
            all_most_likely_poses.append((track_id, most_likely_pose, count))

        # Find the most likely pose across all buoys (highest count)
        if all_most_likely_poses:
            track_id, most_likely_pose, _ = max(
                all_most_likely_poses, key=lambda x: x[2])

            # Create a transform message for the most likely pose
            transform = self.create_transform_message(
                track_id, most_likely_pose)
            tf_message.transforms.append(transform)

            self.latest = tf_message

        else:
            print("No buoys detected in the last 10 seconds.")

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

    def create_transform_message(self, track_id, pose):
        """Creates a TransformStamped message for the buoy's most likely pose."""
        transform = TransformStamped()
        transform.header = Header()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "map"  # Replace with appropriate frame of reference
        # Unique frame for each buoy
        transform.child_frame_id = f"buoy_{track_id}"

        transform.transform.translation.x = pose[1]
        transform.transform.translation.y = pose[0]
        transform.transform.translation.z = 0.0  # Assuming 2D plane (z=0)

        # Identity rotation (no rotation)
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0
        transform.transform.rotation.w = 1.0

        return transform


def main(args=None):
    rclpy.init(args=args)
    prequali_detection = EncirclementTask()
    rclpy.spin(prequali_detection)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
