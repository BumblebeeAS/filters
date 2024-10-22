#!/usr/bin/env python3
"""
Red/Green Gate Detection Node for ROS2

This ROS2 node detects gates based on 3D buoy data from a LiDAR sensor. It uses clustering to group buoys into gates and calculates the position, orientation, and width of each gate. The node also publishes debug images showing the detected buoys and gates.

The node performs the following tasks:
1. Subscribes to a `DetectedObject3DArray` topic to receive buoy detection data.
2. Clusters buoys using Agglomerative Clustering to group them into potential gates. Scales buoy positions in certain direction to improve clustering based on prior knowledge of rough direction of gates
TODO: expose the setting of direction as service.
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
from operator import attrgetter

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
from geometry_msgs.msg import Pose, Quaternion, TransformStamped
from nav_msgs.msg import Odometry
from ml_detector.schema_validator import get_config, load_schema
from rclpy.node import Node
from sensor_msgs.msg import Image
from sklearn.cluster import AgglomerativeClustering, KMeans
from transforms3d.euler import euler2quat, quat2euler, quat2mat
from tf2_ros.transform_broadcaster import TransformBroadcaster
from scipy.spatial.distance import pdist, squareform


class RedGreenGateDetection(Node):
    def __init__(self):
        super().__init__("red_green_gate_detection")
        self.buoys = {}
        self.bridge = CvBridge()
        self.image = None
        self.declare_parameter("debug", True)
        self.debug = self.get_parameter("debug").get_parameter_value().bool_value
        self.past_buoy_ids = set()
        self.is_ned = False
        self.header = None
        self.detector_source = DetectorSource(
            sensor_name="red_green_gate_detector",
            frame_id="map",
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
        self.unknown_id = self.name_to_id["unknown"]
        self.gate_id = self.name_to_id["gate"]

        # scale all points in direction by factor to account for case where distance from start->end gate < width of gate
        self.use_heading = True
        self.heading_direction = np.deg2rad(
            180
        )  # degrees enu # for nbpark # point west
        # calculate 2x2 matrix to transform all x y coordinates to stretch coordinates in direction by 2x
        c, s = np.cos(self.heading_direction), np.sin(self.heading_direction)
        R = np.array([(c, -s), (s, c)])
        self.clustering_T = R @ np.array([[2, 0], [0, 1]]) @ R.T
        self.inv_clustering_T = np.linalg.inv(self.clustering_T)
        self.forward_direction = R @ np.array([1, 0])
        self.vehicle_forward_direction = np.array([1, 0])
        self.buoy_id_to_gate_id = {}
        self.latest_gate_id = -1
        self.gate_poses = {}
        self.debug_image_scale = 10
        self.debug_pub = self.create_publisher(
            Image, "/asv4/robotx/red_green_gates/debug", 10
        )
        self.gate_detections_pub = self.create_publisher(
            DetectedObject3DArray, "/asv4/vision/red_green_gate_detections", 10
        )

        self.tf_broadcaster = TransformBroadcaster(self)
        self.subscription = self.create_subscription(
            DetectedObject3DArray,
            # "/asv4/vision/lidar_small_objects/dets_3d/labelled",
            # "/asv4/vision/detections_2d/projected/filtered",
            "/asv4/robotx/filtered_detections",
            self.detected_objects_callback,
            10,
        )
        self.odom_msg = None
        self.odom_subscription = self.create_subscription(
            Odometry, "/asv4/nav/world", self.odom_callback, 10
        )
        self.create_timer(0.1, self.show_buoys)
        
    def merge_close_detections(self, buoys, threshold=2.0):
        if len(buoys) < 2:
            return buoys

        buoy_list = list(buoys.items())
        positions = np.array([buoy[1][:2] for buoy in buoy_list])  # Extract (x, y) positions
        distances = squareform(pdist(positions))  # Pairwise distances

        merged_buoys = {}
        merged_ids = set()

        for i, (buoy_id, buoy_data) in enumerate(buoy_list):
            if buoy_id in merged_ids:
                continue

            # Initialize merged buoy data and count
            merged_buoy = np.array(buoy_data, dtype=object)
            merged_count = 1

            # Find buoys within the threshold
            for j in range(i + 1, len(buoy_list)):
                other_buoy_id, other_buoy_data = buoy_list[j]
                if other_buoy_id in merged_ids or distances[i, j] >= threshold:
                    continue

                # Update position (average coordinates)
                merged_buoy[:2] = (merged_buoy[:2] * merged_count + other_buoy_data[:2]) / (merged_count + 1)

                # Add class probabilities
                merged_buoy[2] = [
                    merged_buoy[2][0] + other_buoy_data[2][0],
                    merged_buoy[2][1] + other_buoy_data[2][1],
                ]

                merged_ids.add(other_buoy_id)
                merged_count += 1

            # Save the merged buoy
            merged_buoys[buoy_id] = merged_buoy.tolist()

        return merged_buoys


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

    @staticmethod
    def quaternion_to_yaw(q):
        """
        Extract the yaw (rotation around z-axis) from a quaternion.
        The quaternion is assumed to be in the form [w, x, y, z].
        """
        w, x, y, z = q
        # Compute yaw (rotation about z-axis)
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return yaw

    @staticmethod
    def forward_unit_vector_from_quaternion(q):
        """
        Compute the forward unit vector in the global frame from a quaternion.
        The quaternion is assumed to encode mainly yaw, with small roll and pitch.
        """
        yaw = RedGreenGateDetection.quaternion_to_yaw(q)
        # Forward direction (unit vector in 2D, x and y)
        forward_vector = np.array([np.cos(yaw), np.sin(yaw)])
        return forward_vector

    def odom_callback(self, msg):
        self.odom_msg = msg
        self.vehicle_forward_direction = self.forward_unit_vector_from_quaternion(
            [
                msg.pose.pose.orientation.w,
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
            ]
        )

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
            if det.hypothesis.class_id == self.red_buoy_id:
                self.buoys[det.hypothesis.track_id][2][0] += det.hypothesis.probability
            elif det.hypothesis.class_id == self.green_buoy_id:
                self.buoys[det.hypothesis.track_id][2][1] += det.hypothesis.probability
            elif det.hypothesis.class_id == self.unknown_id:
                self.buoys[det.hypothesis.track_id][2][0] += (
                    det.hypothesis.probability / 2
                )
                self.buoys[det.hypothesis.track_id][2][1] += (
                    det.hypothesis.probability / 2
                )

            # unsupported API by ML
            # for class_ in det.hypothesis.classes:
            #     self.get_logger().info(f"Class: {class_.class_id}, Score: {class_.score}")
            #     if class_.class_id == self.red_buoy_id:  # red
            #         self.buoys[det.hypothesis.track_id][2][0] += class_.score
            #     elif class_.class_id == self.green_buoy_id:  # green
            #         self.buoys[det.hypothesis.track_id][2][1] += class_.score
            #     elif class_.class_id == self.unknown_id:
            #         self.buoys[det.hypothesis.track_id][2][0] += class_.score / 2
            #         self.buoys[det.hypothesis.track_id][2][1] += class_.score / 2

        # Merge close detections within 2 meters
        self.buoys = self.merge_close_detections(self.buoys, threshold=2.0)

    def calculate_gate_pose(self, cluster):
        # cluster the cluster into 2 clusters based on the x,y positions of the buoys
        if len(cluster) < 2:
            return None, None, None, None
        # print("Kmeans")
        km = KMeans(n_clusters=2)
        positions = np.array([[t[1][0], t[1][1]] for t in cluster])
        green_red_clusters = km.fit_predict(positions)
        cluster_centers = km.cluster_centers_
        if (
            np.linalg.norm(cluster_centers[0] - cluster_centers[1]) < 6
        ):  # keep buoys at least 6m apart
            return None, None, None, None
        green_identities = [0, 0]  # green_red_cluster_0, green_red_cluster_1
        red_identities = [0, 0]  # red_green_cluster_0, red_green_cluster_1
        for i, cluster_number in enumerate(green_red_clusters):
            probabilities = cluster[i][1][2]
            green_identities[cluster_number] += probabilities[1]
            red_identities[cluster_number] += probabilities[0]

        # if (
        #     (
        #         green_identities[0] == 0
        #         and red_identities[0] > 0
        #         and red_identities[1] > 0
        #     )
        #     or (
        #         green_identities[1] == 0
        #         and red_identities[0] > 0
        #         and red_identities[1] > 0
        #     )
        #     or (
        #         red_identities[0] == 0
        #         and green_identities[0] > 0
        #         and green_identities[1] > 0
        #     )
        #     or (
        #         red_identities[1] == 0
        #         and green_identities[0] > 0
        #         and green_identities[1] > 0
        #     )
        # ):
        if (sum(green_identities) == 0 or sum(red_identities) == 0):
            return (
                None,
                None,
                None,
                None,
            )  # missing any colour on both sides
        if (
            green_identities[0] - red_identities[0]
            > green_identities[1] - red_identities[1]
        ):
            green_buoy_cluster = 0
        elif (
            green_identities[0] - red_identities[0]
            < green_identities[1] - red_identities[1]
        ):
            green_buoy_cluster = 1
        else:
            return None, None, None, None  # undetermined

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
        # the probability of gate being a gate depends on the difference in identities of the buoys in each cluster.
        probability = abs(
            (
                green_identities[0]
                - green_identities[1]
                + red_identities[1]
                - red_identities[0]
            )
        ) / sum(green_identities + red_identities)
        if self.is_ned:
            gate_orientation += -np.pi / 2
        else:
            gate_orientation += np.pi / 2
        return gate_position, gate_orientation, gate_width, probability

    @staticmethod
    def distance_point_to_vector(v1, v2):
        # Calculate the difference between the point v2 and the vector v1's origin
        diff_x = v2[0] - v1[0]
        diff_y = v2[1] - v1[1]
        # Compute the distance using the formula for 2D point to line distance
        return np.abs(diff_x * v1[3] - diff_y * v1[2]) / np.sqrt(
            v1[2] ** 2 + v1[3] ** 2
        )

    def calculate_gate_entrance_exit_pairs(self, gate_detections):
        def det_to_xyv(detection):
            p = detection.hypothesis.kinematics.pose_with_covariance.pose
            x, y = p.position.x, p.position.y
            mat = quat2mat(attrgetter("w", "x", "y", "z")(p.orientation))
            x1, y1, _ = mat @ np.array([1, 0, 0])
            return (x, y, x1, y1)

        def score_between_vectors(v1, v2):
            # Distance from b to a and a to b
            dist_b_to_a = self.distance_point_to_vector(v1, v2)
            dist_a_to_b = self.distance_point_to_vector(v2, v1)
            # Calculate the score
            return 0.5 * (dist_b_to_a + dist_a_to_b)

        # print(gate_detections)
        vectors = np.array(
            [
                det_to_xyv(gate_detections.objects[i])
                for i in range(len(gate_detections.objects))
            ]
        )
        scores = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                score = score_between_vectors(vectors[i], vectors[j])
                scores.append(((i, j), score))

        best_pairs = []
        visited = set()
        for matches in sorted(scores, key=lambda x: x[1]):
            if any(x in visited for x in matches[0]):
                continue
            visited |= set(matches[0])
            best_pairs.append(matches)
        output_gates = []
        for pair in best_pairs:
            gates = vectors[pair[0][0]], vectors[pair[0][1]]
            gates = sorted(
                gates, key=lambda x: np.dot(x[:2], self.vehicle_forward_direction)
            )
            for gate in gates:
                if np.dot(gate[2:], self.forward_direction) < 0:
                    gate[2:] = -gate[2:]
            output_gates.append(gates)
        return output_gates

    def get_best_entrance_exit(self, pairs):
        if self.odom_msg is None:
            return None

        def dist(odom_msg, v1):
            if self.is_ned != odom_msg.header.frame_id.endswith("ned"):
                return self.distance_point_to_vector(
                    v1,
                    (
                        odom_msg.pose.pose.position.y,
                        odom_msg.pose.pose.position.x,
                        0,
                        0,
                    ),
                )
            else:
                return self.distance_point_to_vector(
                    v1,
                    (
                        odom_msg.pose.pose.position.x,
                        odom_msg.pose.pose.position.y,
                        0,
                        0,
                    ),
                )

        return min(pairs, key=lambda x: dist(self.odom_msg, x[0]))

    def publish_gate_transform(self, entrance_gate, exit_gate):
        # Extract position and direction for entrance gate
        x_e, y_e, vx_e, vy_e = entrance_gate
        # Extract position and direction for exit gate
        x_ex, y_ex, vx_ex, vy_ex = exit_gate
        # Compute yaw from vx, vy for entrance gate
        yaw_e = np.arctan2(vy_e, vx_e)
        # Compute yaw from vx, vy for exit gate
        yaw_ex = np.arctan2(vy_ex, vx_ex)

        quat_e = euler2quat(0, 0, yaw_e)
        quat_ex = euler2quat(0, 0, yaw_ex)

        # Create a TransformStamped message for entrance gate
        entrance_transform = TransformStamped()
        entrance_transform.header.stamp = self.header.stamp
        entrance_transform.header.frame_id = "map" + ("_ned" if self.is_ned else "")
        entrance_transform.child_frame_id = "entrance_gate" + (
            "_ned" if self.is_ned else ""
        )
        entrance_transform.transform.translation.x = x_e
        entrance_transform.transform.translation.y = y_e
        entrance_transform.transform.translation.z = 0.0
        entrance_transform.transform.rotation.w = quat_e[0]
        entrance_transform.transform.rotation.x = quat_e[1]
        entrance_transform.transform.rotation.y = quat_e[2]
        entrance_transform.transform.rotation.z = quat_e[3]

        # Create a TransformStamped message for exit gate
        exit_transform = TransformStamped()
        exit_transform.header.stamp = self.get_clock().now().to_msg()
        exit_transform.header.frame_id = "map" + ("_ned" if self.is_ned else "")
        exit_transform.child_frame_id = "exit_gate" + ("_ned" if self.is_ned else "")
        exit_transform.transform.translation.x = x_ex
        exit_transform.transform.translation.y = y_ex
        exit_transform.transform.translation.z = 0.0
        exit_transform.transform.rotation.w = quat_ex[0]
        exit_transform.transform.rotation.x = quat_ex[1]
        exit_transform.transform.rotation.y = quat_ex[2]
        exit_transform.transform.rotation.z = quat_ex[3]

        # Broadcast the transforms
        self.tf_broadcaster.sendTransform(entrance_transform)
        self.tf_broadcaster.sendTransform(exit_transform)


    def get_relative_position(self, gate):
        """Calculate the relative position of the gate to the vehicle."""
        if self.odom_msg is None:
            return
        return np.array([
            gate.hypothesis.kinematics.pose_with_covariance.pose.position.x - self.odom_msg.pose.pose.position.x,
            gate.hypothesis.kinematics.pose_with_covariance.pose.position.y - self.odom_msg.pose.pose.position.y
        ])

    def dot_with_forward_direction(self, relative_position):
        """Compute the dot product of the relative position with the vehicle's forward direction."""
        return np.dot(relative_position, self.vehicle_forward_direction)

    def show_buoys(self):
        if not self.buoys:
            return
    
        if self.odom_msg is None:
            return

        # Extract buoy positions for clustering
        positions = np.array([[buoy[0], buoy[1]] for buoy in self.buoys.values()])
        if self.use_heading:
            positions = positions @ self.clustering_T.T

        # # Apply DBSCAN clustering
        # dbscan = DBSCAN(
        #     eps=15, min_samples=2
        # )  # Adjust eps based on required distance threshold e.g. 10m between robotx buoys
        # cluster_labels = dbscan.fit_predict(positions)
        hierarchical_clusterer = AgglomerativeClustering(
            n_clusters=None,  # Set to None to allow distance-based threshold
            # distance_threshold=12,  # Similar to 'eps', defines max distance for clusters
            # linkage="ward",  # Linkage method; 'ward', 'complete', 'average', or 'single'
            distance_threshold=20,  # Similar to 'eps', defines max distance for clusters
            linkage="ward",  # Linkage method; 'ward', 'complete', 'average', or 'single'
        )
        if len(positions) == 1:
            cluster_labels = [0]
        else:
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
        gate_detections.header.stamp = self.get_clock().now().to_msg()
        gate_detections.name = "red_green_gate_detector"
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
            gate_position, yaw, width, probability = self.calculate_gate_pose(
                cluster_details
            )
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
            gate_detection.hypothesis.probability = probability
            gate_detections.objects.append(gate_detection)

        # closest gate
        gate_detections.objects = sorted(
            gate_detections.objects,
            key=lambda x: np.linalg.norm(
                np.array(
                    [
                        x.hypothesis.kinematics.pose_with_covariance.pose.position.x,
                        x.hypothesis.kinematics.pose_with_covariance.pose.position.y,
                    ]
                )
                - np.array(
                    [
                        self.odom_msg.pose.pose.position.x,
                        self.odom_msg.pose.pose.position.y,
                    ]
                )
            ),
        )

        closest_gate = None
        second_closest_gate = None
        closest_gate_infront = None
        if len(gate_detections.objects) > 0:
            closest_gate = gate_detections.objects[0]
            # print(f"closest gate: {closest_gate}")
        if len(gate_detections.objects) > 1:
            second_closest_gate = gate_detections.objects[1]
            # print(f"second closest gate: {second_closest_gate}")
        gates_infront = [
            gate
            for gate in gate_detections.objects
            if np.dot(
                [
                    gate.hypothesis.kinematics.pose_with_covariance.pose.position.x
                    - self.odom_msg.pose.pose.position.x,
                    gate.hypothesis.kinematics.pose_with_covariance.pose.position.y
                    - self.odom_msg.pose.pose.position.y,
                ],
                self.vehicle_forward_direction,
            )
            > 0
        ]
        # # remove gates > 40m to left or right of vehicle
        # gates_infront = [
        #     gate
        #     for gate in gates_infront
        #     if np.abs(
        #         np.dot(
        #             [
        #                 gate.hypothesis.kinematics.pose_with_covariance.pose.position.x
        #                 - self.odom_msg.pose.pose.position.x,
        #                 gate.hypothesis.kinematics.pose_with_covariance.pose.position.y
        #                 - self.odom_msg.pose.pose.position.y,
        #             ],
        #             self.vehicle_forward_direction,
        #         )
        #     )
        #     < 40
        # ]
        # sort by distance from vehicle forward direction
        gates_infront = sorted(
            gates_infront,
            key=lambda x: np.abs(
                np.dot(
                    np.array(
                        [
                            x.hypothesis.kinematics.pose_with_covariance.pose.position.x
                            - self.odom_msg.pose.pose.position.x,
                            x.hypothesis.kinematics.pose_with_covariance.pose.position.y
                            - self.odom_msg.pose.pose.position.y,
                        ]
                    ),
                    self.vehicle_forward_direction,
                )
            ),
        )
        if len(gates_infront) > 0:
            closest_gate_infront = gates_infront[0]
            # print(f"closest gate infront: {gates_infront[0]}")

        self.gate_detections_pub.publish(gate_detections)
        gate_pairs = self.calculate_gate_entrance_exit_pairs(gate_detections)
        if len(gate_pairs) == 0:
            best_pair = None
        else:
            best_pair = self.get_best_entrance_exit(gate_pairs)
        if best_pair is not None:
            self.publish_gate_transform(*best_pair)

        if self.debug:
            if self.odom_msg is None:
                return
            odom_x = self.odom_msg.pose.pose.position.x
            odom_y = self.odom_msg.pose.pose.position.y
            min_x = min(buoy[0] for buoy in self.buoys.values())
            max_x = max(buoy[0] for buoy in self.buoys.values())
            min_y = min(buoy[1] for buoy in self.buoys.values())
            max_y = max(buoy[1] for buoy in self.buoys.values())
            min_x = min(odom_x, min_x)
            max_x = max(odom_x, max_x)
            min_y = min(odom_y, min_y)
            max_y = max(odom_y, max_y)

            extension = max(0, 50 - max_x + min_x, 50 - max_y + min_y)
            min_x -= int(extension / 2)
            max_x += int(extension / 2)
            min_y -= int(extension / 2)
            max_y += int(extension / 2)

            min_x *= self.debug_image_scale
            max_x *= self.debug_image_scale
            min_y *= self.debug_image_scale
            max_y *= self.debug_image_scale
            if max_y - min_y > 2000 or max_x - min_x > 2000:
                print("Image too large, not publishing")
                return

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

            # draw arrow where vehicle is
            # Plot the vehicle's current odometry
            odom_x_scaled = int(odom_x * self.debug_image_scale - min_x)
            odom_y_scaled = int(odom_y * self.debug_image_scale - min_y)
            # Draw a star for the vehicle's current position
            cv2.drawMarker(
                image,
                (odom_x_scaled, odom_y_scaled),
                (0, 0, 0),
                markerType=cv2.MARKER_STAR,
                markerSize=20,
                thickness=2,
            )

            # for each gate detection, draw arrow, direction based on yaw
            for gate_detection in gate_detections.objects:
                gate_pose = (
                    gate_detection.hypothesis.kinematics.pose_with_covariance.pose
                )
                x = int((gate_pose.position.x) * self.debug_image_scale - min_x)
                y = int((gate_pose.position.y) * self.debug_image_scale - min_y)
                yaw = quat2euler(attrgetter("w", "x", "y", "z")(gate_pose.orientation))[
                    2
                ]
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

            # draw the closest, second closest and closest infront gate with different colors
            if closest_gate is not None:
                gate_pose = closest_gate.hypothesis.kinematics.pose_with_covariance.pose
                x = int((gate_pose.position.x) * self.debug_image_scale - min_x)
                y = int((gate_pose.position.y) * self.debug_image_scale - min_y)
                cv2.circle(image, (x, y), 10, (0, 255, 0), 1, 1)
            if second_closest_gate is not None:
                gate_pose = (
                    second_closest_gate.hypothesis.kinematics.pose_with_covariance.pose
                )
                x = int((gate_pose.position.x) * self.debug_image_scale - min_x)
                y = int((gate_pose.position.y) * self.debug_image_scale - min_y)
                cv2.circle(image, (x, y), 15, (0, 0, 255), 1, 1)
            if closest_gate_infront is not None:
                gate_pose = (
                    closest_gate_infront.hypothesis.kinematics.pose_with_covariance.pose
                )
                x = int((gate_pose.position.x) * self.debug_image_scale - min_x)
                y = int((gate_pose.position.y) * self.debug_image_scale - min_y)
                cv2.circle(image, (x, y), 20, (255, 0, 0), 1, 1)
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))


def main(args=None):
    rclpy.init(args=args)
    red_green_detection = RedGreenGateDetection()
    rclpy.spin(red_green_detection)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
