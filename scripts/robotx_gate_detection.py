#!/usr/bin/env python3
"""
Red/Green Gate Detection Node for ROS2

This ROS2 node detects gates based on 3D buoy data from a LiDAR sensor. It uses clustering to group buoys into gates and calculates the position, orientation, and width of each gate. The node also publishes debug images showing the detected buoys and gates.

The node performs the following tasks:
1. Subscribes to a `DetectedObject3DArray` topic to receive buoy detection data.
2. Clusters buoys using Agglomerative Clustering to group them into potential gates.
3. Calculates the position, orientation, and width of each gate based on clustered buoy positions.
4. Publishes detected gates as `DetectedObject3DArray` messages.
5. Optionally publishes debug images showing buoy positions, gate positions, and orientations.

Parameters:
- `objects_config` (str): Path to the configuration file for object schemas.

Publications:
- `/asv4/robotx/gates/debug` (sensor_msgs/Image): Debug image showing buoy positions and gates.
- `/asv4/vision/gate_detections` (bb_perception_msgs/DetectedObject3DArray): Detected gates.

Subscriptions:
- `/asv4/vision/lidar_small_objects/dets_3d/labelled` (bb_perception_msgs/DetectedObject3DArray): Input buoy detections.

Dependencies:
- OpenCV for image processing and visualization.
- scikit-learn for clustering.
- transforms3d for Euler angle to quaternion conversion.
- rclpy for ROS2 node functionality.
"""
import colorsys
import random
from pathlib import Path

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from bb_perception_msgs.msg import (
    DetectedObject3D,
    DetectedObject3DArray,
    DetectorSource,
    ObjectHypothesis,
)
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, Quaternion
from ml_detector.schema_validator import get_config, load_schema
from rclpy.node import Node
from sensor_msgs.msg import Image
from sklearn.cluster import AgglomerativeClustering, KMeans
from transforms3d.euler import euler2quat


class GateDetection(Node):
    def __init__(self):
        super().__init__("gate_detection")
        self.buoys = {}
        self.bridge = CvBridge()
        self.image = None
        self.declare_parameter("debug", True)
        self.debug = self.get_parameter("debug").get_parameter_value().bool_value
        self.past_buoy_ids = set()
        self.is_ned = False
        self.header = None
        self.detector_source = DetectorSource(
            sensor_name="gate_detector",
            frame_id="asv4/base_link",
            category=DetectorSource.LIDAR,
        )

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
        self.gate_id = self.name_to_id["gate"]

        self.buoy_id_to_gate_id = {}
        self.latest_gate_id = -1
        self.gate_poses = {}
        self.debug_image_scale = 10
        self.debug_pub = self.create_publisher(Image, "/asv4/robotx/gates/debug", 10)
        self.gate_detections_pub = self.create_publisher(
            DetectedObject3DArray, "/asv4/vision/gate_detections", 10
        )
        self.subscription = self.create_subscription(
            DetectedObject3DArray,
            "/asv4/vision/lidar_small_objects/dets_3d/labelled",
            self.detected_objects_callback,
            10,
        )
        self.create_timer(0.1, self.show_buoys)

    def get_new_gate_id(self):
        self.latest_gate_id += 1
        return self.latest_gate_id

    def get_color(self, track_id):
        random.seed(0)
        num_colors = 3  # Adjust based on the expected number of unique track IDs
        hue = (track_id % num_colors) / num_colors  # Ensures hue is between 0 and 1

        # Convert from HSV to RGB
        rgb_float = colorsys.hsv_to_rgb(
            hue, 1.0, 1.0
        )  # Full saturation and value for bright colors
        rgb = tuple(int(255 * x) for x in rgb_float)

        return rgb

    def detected_objects_callback(self, msg):
        self.buoys = {}
        if len(msg.objects) != 0:
            self.is_ned = msg.objects[0].hypothesis.kinematics.header.frame_id.endswith(
                "ned"
            )
            self.header = msg.objects[0].hypothesis.kinematics.header
        for det in msg.objects:
            is_green_red_buoy = (
                det.hypothesis.class_id == self.red_buoy_id
                or det.hypothesis.class_id == self.green_buoy_id
            )
            if not is_green_red_buoy:
                continue
            pose = det.hypothesis.kinematics.pose_with_covariance.pose
            self.buoys[det.hypothesis.track_id] = [
                pose.position.x,
                pose.position.y,
                [0, 0],  # red, green
            ]
            for class_ in det.hypothesis.classes:
                if class_.class_id == self.red_buoy_id:  # red
                    self.buoys[det.hypothesis.track_id][2][0] += class_.score
                elif class_.class_id == self.green_buoy_id:  # green
                    self.buoys[det.hypothesis.track_id][2][1] += class_.score

    def calculate_gate_pose(self, cluster):
        # cluster the cluster into 2 clusters based on the x,y positions of the buoys
        if len(cluster) < 2:
            return None, None, None
        km = KMeans(n_clusters=2)
        positions = np.array([[t[1][0], t[1][1]] for t in cluster])
        green_red_clusters = km.fit_predict(positions)
        cluster_centers = km.cluster_centers_
        green_identities = [0, 0]  # green_red_cluster_0, green_red_cluster_1
        for i, c in enumerate(green_red_clusters):
            probabilities = cluster[i][1][2]
            if c == 0:
                green_identities[0] += probabilities[1] - probabilities[0]
            else:
                green_identities[1] += probabilities[1] - probabilities[0]
        if green_identities[0] > green_identities[1]:
            green_buoy_cluster = 0
        else:
            green_buoy_cluster = 1
        green_buoy_pose = cluster_centers[green_buoy_cluster]
        red_buoy_pose = cluster_centers[1 - green_buoy_cluster]
        gate_position = [
            (green_buoy_pose[0] + red_buoy_pose[0]) / 2,
            (green_buoy_pose[1] + red_buoy_pose[1]) / 2,
        ]
        gate_orientation = np.arctan2(
            green_buoy_pose[1] - red_buoy_pose[1], green_buoy_pose[0] - red_buoy_pose[0]
        )
        gate_width = np.linalg.norm(np.array(green_buoy_pose) - np.array(red_buoy_pose))
        if self.is_ned:
            gate_orientation -= np.pi / 2
        else:
            gate_orientation += np.pi / 2
        return gate_position, gate_orientation, gate_width

    def show_buoys(self):
        if not self.buoys:
            return

        # Extract buoy positions for clustering
        positions = np.array([[buoy[0], buoy[1]] for buoy in self.buoys.values()])

        # # Apply DBSCAN clustering
        # dbscan = DBSCAN(
        #     eps=15, min_samples=2
        # )  # Adjust eps based on required distance threshold e.g. 10m between robotx buoys
        # cluster_labels = dbscan.fit_predict(positions)
        hierarchical_clusterer = AgglomerativeClustering(
            n_clusters=None,  # Set to None to allow distance-based threshold
            distance_threshold=15,  # Similar to 'eps', defines max distance for clusters
            linkage="ward",  # Linkage method; 'ward', 'complete', 'average', or 'single'
        )
        cluster_labels = hierarchical_clusterer.fit_predict(positions)
        # for each cluster, check if any of the buoy track ids are in past_buoy_ids
        # if so, add self.buoy_id_to_gate_id[track_id] = that gate id for all buoys in the cluster
        # if not, add a new gate id to self.buoy_id_to_gate_id[track_id] for all buoys in the cluster
        # new gate id is just incremented from the previous gate id with self.get_new_gate_id()

        clusters = {}
        for i, cluster_label in enumerate(cluster_labels):
            if cluster_label not in clusters:
                clusters[cluster_label] = []
            clusters[cluster_label].append(i)

        gate_detections = DetectedObject3DArray()
        gate_detections.header = self.header
        gate_detections.name = "gate_detector"
        gate_detections.source = self.detector_source

        for cluster in clusters.values():
            for i in cluster:
                track_id = list(self.buoys.keys())[i]
                if track_id in self.past_buoy_ids:
                    gate_id = self.buoy_id_to_gate_id[track_id]
                else:
                    gate_id = self.get_new_gate_id()
                    self.buoy_id_to_gate_id[track_id] = gate_id
                self.past_buoy_ids.add(track_id)

            # logic for calculating gate pose given cluster of buoy poses
            cluster_tids = [list(self.buoys.keys())[i] for i in cluster]
            cluster_details = [(tid, self.buoys[tid]) for tid in cluster_tids]
            gate_position, yaw, width = self.calculate_gate_pose(cluster_details)
            if gate_position is None:
                continue
            gate_detection = DetectedObject3D()
            gate_detection.hypothesis.mode = ObjectHypothesis.MODE_TRACKED
            gate_detection.hypothesis.track_id = self.buoy_id_to_gate_id[
                cluster_tids[0]
            ]
            gate_detection.hypothesis.class_id = self.name_to_id[
                "gate"
            ]  # gate class id
            gate_detection.hypothesis.kinematics.header = self.header
            gate_pose = Pose()
            gate_pose.position.x = gate_position[0]
            gate_pose.position.y = gate_position[1]
            gate_pose.position.z = 0.0

            quat = euler2quat(0, 0, yaw)
            gate_pose.orientation = Quaternion(
                w=quat[0], x=quat[1], y=quat[2], z=quat[3]
            )
            gate_detection.hypothesis.kinematics.pose_with_covariance.pose = gate_pose
            gate_detection.hypothesis.shape.dimensions.x = 0.5
            gate_detection.hypothesis.shape.dimensions.y = width
            gate_detection.hypothesis.shape.dimensions.z = 1.0

            gate_detections.objects.append(gate_detection)
        self.gate_detections_pub.publish(gate_detections)
        if self.debug:
            min_x = min(buoy[0] for buoy in self.buoys.values())
            max_x = max(buoy[0] for buoy in self.buoys.values())
            min_y = min(buoy[1] for buoy in self.buoys.values())
            max_y = max(buoy[1] for buoy in self.buoys.values())
            extension = max(0, 50 - max_x + min_x, 50 - max_y + min_y)
            min_x -= int(extension / 2)
            max_x += int(extension / 2)
            min_y -= int(extension / 2)
            max_y += int(extension / 2)

            min_x *= self.debug_image_scale
            max_x *= self.debug_image_scale
            min_y *= self.debug_image_scale
            max_y *= self.debug_image_scale

            image = 255 * np.ones(
                (
                    int(max_y - min_y + 1),
                    int(max_x - min_x + 1),
                    3,
                ),
                np.uint8,
            )

            for i, (track_id, buoy) in enumerate(self.buoys.items()):
                x, y, [red, green] = buoy
                x_scaled = int(x * self.debug_image_scale - min_x)
                y_scaled = int(y * self.debug_image_scale - min_y)

                # Assign color based on cluster label
                cluster_label = cluster_labels[i]
                color = self.get_color(cluster_label)

                # Draw a triangle for green buoys and a circle for red buoys
                if green > red:
                    # Draw triangle for green buoy
                    points = np.array(
                        [
                            [x_scaled, y_scaled - 15],
                            [x_scaled - 15, y_scaled + 15],
                            [x_scaled + 15, y_scaled + 15],
                        ],
                        np.int32,
                    )
                    cv2.polylines(
                        image, [points], isClosed=True, color=color, thickness=2
                    )
                else:
                    # Draw circle for red buoy
                    cv2.circle(image, (x_scaled, y_scaled), 10, color, -1)

                # Display track_id and probabilities as text
                cv2.putText(
                    image,
                    f"{track_id}",
                    (x_scaled, y_scaled - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )

            # for each gate detection, draw arrow, direction based on yaw
            for gate_detection in gate_detections.objects:
                gate_pose = (
                    gate_detection.hypothesis.kinematics.pose_with_covariance.pose
                )
                x = int((gate_pose.position.x) * self.debug_image_scale - min_x)
                y = int((gate_pose.position.y) * self.debug_image_scale - min_y)
                yaw = np.arctan2(gate_pose.orientation.y, gate_pose.orientation.x)
                if self.is_ned:
                    yaw -= np.pi / 2
                else:
                    yaw = -yaw
                cv2.line(
                    image,
                    (x, y),
                    (
                        int(x + 50 * np.cos(yaw)),
                        int(y + 50 * np.sin(yaw)),
                    ),
                    (0, 100, 0),
                    5,
                )
            # add stamp as text at bottom corner
            cv2.putText(
                image,
                f"Stamp: {self.header.stamp.sec}.{self.header.stamp.nanosec}",
                (10, int(max_y - min_y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))


def main(args=None):
    rclpy.init(args=args)
    buoy_detection = GateDetection()
    rclpy.spin(buoy_detection)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
