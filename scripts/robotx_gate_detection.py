#!/usr/bin/env python3
"""
Red/Green Gate Detection Node for ROS2

This ROS2 node detects gates based on 3D buoy data from a LiDAR sensor. It uses clustering to group buoys into gates and calculates the position, orientation, and width of each gate. The node also publishes debug images showing the detected buoys and gates.

The node performs the following tasks:
1. Subscribes to a `DetectedObject3DArray` topic to receive buoy detection data.
2. Clusters buoys using Agglomerative Clustering to group them into potential clusters of 4 buoys (gates).
    Unlike quali gate estimation, doesn't scale buoy positions (identity mat)
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

"""
ros2 service call /robotx24/configure_gate_task bb_robotx_msgs/srv/ConfigureGateTask "active: true 
use_heading: false
estimated_pose:
  header:
    stamp:
      sec: 0
      nanosec: 0
    frame_id: 'map'
  pose:
    position:
      x: 190.0
      y: 250.0
      z: 0.0
    orientation:
      x: 0.0
      y: 0.0
      z: 1.0
      w: 0.0"
"""

import colorsys
import random
from pathlib import Path
from operator import attrgetter

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from bb_perception_msgs.msg import (
    DetectedObject3D,
    DetectedObject3DArray,
    DetectorSource,
    ObjectHypothesis,
)
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, Quaternion, TransformStamped
from bb_robotx_msgs.srv import ConfigureGateTask
from ml_detector.schema_validator import get_config, load_schema
from rclpy.node import Node
from sklearn.cluster import AgglomerativeClustering
from transforms3d.euler import euler2quat, quat2mat, quat2euler
from tf2_ros.transform_broadcaster import TransformBroadcaster
from nav_msgs.msg import Odometry
from typing import Optional, Tuple, List
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

np.seterr(divide="ignore", invalid="ignore")


class GateDetection(Node):
    def __init__(self):
        super().__init__("gate_detection")
        self.running = False
        self.buoys = {}
        self.bridge = CvBridge()
        self.image = None
        self.declare_parameter("debug", True)
        self.debug = self.get_parameter("debug").get_parameter_value().bool_value
        self.past_buoy_ids = set()
        self.is_ned = False
        self.header = None
        self.initial_pose_estimate = None
        self.R = np.eye(3)
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
        self.gate_estimate_valid = False
        self.gate_estimate_valid_pub = self.create_publisher(
            Bool, "/asv4/vision/gate_estimate_valid", 10
        )
        self.id_to_name = {
            obj["label"]: obj["name"] for obj in self.objects_config["objects"]
        }
        self.gate_geofence = None
        self.name_to_id = {v: k for k, v in self.id_to_name.items()}
        self.green_buoy_id = self.name_to_id["green_cylinder"]
        self.red_buoy_id = self.name_to_id["red_cylinder"]
        self.white_buoy_id = self.name_to_id["white_cylinder"]
        self.unknown_id = self.name_to_id["unknown"]
        self.gate_id = self.name_to_id["gate"]
        self.vehicle_position = None

        self.latest_poses = None

        # scale all points in direction by factor to account for case where distance from start->end gate < width of gate
        self.use_heading = True
        # TODO: expose this as service or parameter
        self.heading_direction = np.deg2rad(
            180
        )  # degrees enu # for nbpark # point west
        # self.heading_direction = np.deg2rad(-30) # degrees enu for rsyc

        # calculate 2x2 matrix to transform all x y coordinates to stretch coordinates in direction by 2x
        c, s = np.cos(self.heading_direction), np.sin(self.heading_direction)
        R = np.array([(c, -s), (s, c)])
        self.clustering_T = R @ np.array([[2, 0], [0, 1]]) @ R.T
        # self.clustering_T = np.eye(2)
        self.inv_clustering_T = np.linalg.inv(self.clustering_T)
        self.forward_direction = R @ np.array([1, 0])

        self.buoy_id_to_gate_id = {}
        self.latest_gate_id = -1
        self.gate_poses = {}
        self.debug_image_scale = 10
        # self.debug_pub = self.create_publisher(Image, "/asv4/robotx/gates/debug", 10)
        self.gate_detections_pub = self.create_publisher(
            DetectedObject3DArray, "/asv4/vision/gate_detections", 10
        )

        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom_sub = self.create_subscription(
            Odometry,
            "/asv4/nav/world",
            self.odom_callback,
            10,
        )
        self.subscription = self.create_subscription(
            DetectedObject3DArray,
            # "/asv4/vision/lidar_small_objects/dets_3d/labelled",
            "/asv4/robotx/filtered_detections",
            self.detected_objects_callback,
            10,
        )
        self.get_logger().info("subscribed to : /asv4/robotx/filtered_detections")
        self.gate_task_config_service = self.create_service(
            ConfigureGateTask,
            "/robotx24/configure_gate_task",
            self.configure_gate_task_callback,
        )
        self.create_timer(0.1, self.show_buoys)

    def configure_gate_task_callback(
        self, req: ConfigureGateTask.Request, res: ConfigureGateTask.Response
    ):
        if not req.active:
            self.running = False
            self.buoys = {}
            self.past_buoy_ids = set()
            self.buoy_id_to_gate_id = {}
            self.latest_gate_id = -1
            self.gate_poses = {}
            self.gate_estimate_valid = False
            res.success = True
            return res
        self.use_heading = req.use_heading
        self.initial_pose_estimate = req.estimated_pose
        if self.use_heading:
            self.heading_direction = quat2euler(
                attrgetter("w", "x", "y", "z")(self.initial_pose_estimate.pose.orientation)
            )[2]
        if self.use_heading:
            self.gate_geofence = self.compute_geofence(self.initial_pose_estimate, 60, 30)
        else:  # set geofence as square
            self.gate_geofence = self.compute_geofence(self.initial_pose_estimate, 60, 60)
        self.get_logger().info(f"Setting geofence to {self.gate_geofence}")
        self.clustering_T = self.R @ np.array([[2, 0], [0, 1]]) @ self.R.T
        self.forward_direction = self.R @ np.array([1, 0])
        self.running = True
        res.success = True
        return res

    def compute_geofence(self, pose, width, length):
        points = np.array(
            [
                [-length / 2, -width / 2],
                [-length / 2, width / 2],
                [length / 2, width / 2],
                [length / 2, -width / 2],
            ]
        )
        c, s = np.cos(self.heading_direction), np.sin(self.heading_direction)
        self.R = np.array([(c, -s), (s, c)])
        points = points @ self.R.T
        points += np.array([pose.pose.position.x, pose.pose.position.y])
        return Polygon(points)

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

    def odom_callback(self, msg):
        self.vehicle_position = msg.pose.pose.position

    def detected_objects_callback(self, msg):
        if not self.running:
            return
        if len(msg.objects) != 0:
            self.buoys = {}
            self.is_ned = msg.objects[0].hypothesis.kinematics.header.frame_id.endswith(
                "ned"
            )
            self.header = msg.objects[0].hypothesis.kinematics.header
        for det in msg.objects:
            is_green_red_buoy = (
                det.hypothesis.class_id == self.red_buoy_id
                or det.hypothesis.class_id == self.green_buoy_id
                or det.hypothesis.class_id == self.white_buoy_id
                or det.hypothesis.class_id == self.unknown_id
            )
            if not is_green_red_buoy:
                continue
            if not self.gate_geofence.contains(Point(
                det.hypothesis.kinematics.pose_with_covariance.pose.position.x, det.hypothesis.kinematics.pose_with_covariance.pose.position.y)):
                continue
            pose = det.hypothesis.kinematics.pose_with_covariance.pose
            if det.hypothesis.track_id not in self.buoys:
                self.buoys[det.hypothesis.track_id] = [
                    pose.position.x,
                    pose.position.y,
                    [0, 0, 0],  # red, green, white
                ]
            self.buoys[det.hypothesis.track_id][0] = pose.position.x
            self.buoys[det.hypothesis.track_id][1] = pose.position.y
            if det.hypothesis.class_id == self.red_buoy_id:
                self.buoys[det.hypothesis.track_id][2][0] += det.hypothesis.probability
            elif det.hypothesis.class_id == self.green_buoy_id:
                self.buoys[det.hypothesis.track_id][2][1] += det.hypothesis.probability
            elif det.hypothesis.class_id == self.white_buoy_id:
                self.buoys[det.hypothesis.track_id][2][2] += det.hypothesis.probability
            elif det.hypothesis.class_id == self.unknown_id:
                self.buoys[det.hypothesis.track_id][2][0] += (
                    det.hypothesis.probability / 3
                )
                self.buoys[det.hypothesis.track_id][2][1] += (
                    det.hypothesis.probability / 3
                )
                self.buoys[det.hypothesis.track_id][2][2] += (
                    det.hypothesis.probability / 3
                )
            # for class_ in det.hypothesis.classes:
            #     if class_.class_id == self.red_buoy_id:  # red
            #         self.buoys[det.hypothesis.track_id][2][0] += class_.score
            #     elif class_.class_id == self.green_buoy_id:  # green
            #         self.buoys[det.hypothesis.track_id][2][1] += class_.score
            #     elif class_.class_id == self.white_buoy_id:  # white
            #         self.buoys[det.hypothesis.track_id][2][2] += class_.score
            #     elif class_.class_id == self.unknown_id:
            #         self.buoys[det.hypothesis.track_id][2][0] += class_.score / 3
            #         self.buoys[det.hypothesis.track_id][2][1] += class_.score / 3
            #         self.buoys[det.hypothesis.track_id][2][2] += class_.score / 3

    def inject_missing_pose_between(self, poses):
        assert len(poses) == 3
        poses = np.array(poses)
        output_poses = [p for p in poses]
        d1 = np.linalg.norm(poses[0] - poses[1])
        d2 = np.linalg.norm(poses[1] - poses[2])
        if d1 > d2:
            output_poses.insert(1, (poses[0] + poses[1]) / 2)
        else:
            output_poses.insert(2, (poses[1] + poses[2]) / 2)
        return output_poses

    def calculate_gate_poses(
        self, cluster
    ) -> Optional[Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], float]]:
        # cluster the cluster into 2 clusters based on the x,y positions of the buoys
        if len(cluster) < 3:
            self.get_logger().info(f"Not enough buoys to form a gate {cluster} {len(cluster)}", throttle_duration_sec=2.0)
            return None
        recluster = AgglomerativeClustering(
            n_clusters=None, distance_threshold=8, linkage="ward"
        )
        cluster_labels = recluster.fit_predict(np.array([c[1][:2] for c in cluster]))
        num_children = recluster.n_clusters_
        if num_children < 3 or num_children > 4:
            self.get_logger().info(f"Invalid number of clusters {num_children}", throttle_duration_sec=2.0)
            return None
        cluster_centroids = np.zeros((num_children, 2))
        for i in range(num_children):
            cluster_centroids[i] = np.mean(
                np.array([c[1][:2] for c in cluster])[cluster_labels == i], axis=0
            )

        m, b = np.polyfit(cluster_centroids[:, 0], cluster_centroids[:, 1], 1)
        v = np.array([1, m])
        v /= np.linalg.norm(v)
        yaw = np.arctan2(v[1], v[0])
        if self.is_ned:
            yaw += -np.pi / 2
        else:
            yaw += np.pi / 2
        order = np.argsort(cluster_centroids @ v.T)
        id_to_order = {order[i]: i for i in range(num_children)}

        if num_children == 4:
            gate_poses = [
                np.mean(
                    [cluster_centroids[order[0]], cluster_centroids[order[1]]], axis=0
                ),
                np.mean(
                    [cluster_centroids[order[1]], cluster_centroids[order[2]]], axis=0
                ),
                np.mean(
                    [cluster_centroids[order[2]], cluster_centroids[order[3]]], axis=0
                ),
            ]

            buoy_colours = np.zeros(
                (4, 3)
            )  # first row is left most cluster, columns are red green white
            for i, c in enumerate(cluster):
                internal_cluster_id = cluster_labels[i]
                buoy_colours[id_to_order[internal_cluster_id]] += c[1][2]
            for i in range(4):
                if np.sum(buoy_colours[i]) == 0:
                    continue
                buoy_colours[i] /= np.sum(buoy_colours[i])

            best_colours = np.argmax(buoy_colours, axis=1)
            if np.all(best_colours == np.array([0, 2, 2, 1])) or (
                best_colours[0] == 0 and best_colours[-1] == 1
            ):
                # red on left,
                return gate_poses, yaw
            elif np.all(best_colours == np.array([1, 2, 2, 0])) or (
                best_colours[0] == 1 and best_colours[-1] == 0
            ):
                # red on right
                return gate_poses[::-1], yaw + np.pi
            else:
                # self.get_logger().info(f"Gate colours does not match r w w g: {buoy_colours}")
                return None
        elif num_children == 3:
            # case 1: missing buoy on either edge
            d12 = np.linalg.norm(
                cluster_centroids[order[1]] - cluster_centroids[order[0]]
            )
            d23 = np.linalg.norm(
                cluster_centroids[order[2]] - cluster_centroids[order[1]]
            )
            buoy_colours = np.zeros(
                (3, 3)
            )  # first row is left most cluster, columns are red green white
            cluster_centroid_ordered = np.array(
                [cluster_centroids[order[i]] for i in range(3)]
            )
            for i, c in enumerate(cluster):
                internal_cluster_id = cluster_labels[i]
                buoy_colours[id_to_order[internal_cluster_id]] += c[1][2]
            for i in range(3):
                buoy_colours[i] /= np.sum(buoy_colours[i])
            missing_buoy_in_middle = d12 > d23 * 1.75 or d23 > d12 * 1.75
            gate_poses = None
            output_yaw = yaw
            if missing_buoy_in_middle:
                updated_poses = self.inject_missing_pose_between(
                    cluster_centroid_ordered
                )
                if np.all(np.argmax(buoy_colours, axis=1) == np.array([0, 2, 1])):
                    gate_poses = updated_poses
                    self.get_logger().info("Detected 3/4 gates with missing white")
                elif np.all(np.argmax(buoy_colours, axis=1) == np.array([1, 2, 0])):
                    gate_poses = updated_poses[::-1]
                    self.get_logger().info("Detected 3/4 gates with missing white")
                else:
                    # self.get_logger().info("Gate colours does not match r w g")
                    return None
            else:
                if np.all(
                    np.argmax(buoy_colours, axis=1) == np.array([0, 2, 2])
                ):  # rww
                    self.get_logger().info(f"{v}")
                    gate_poses = [
                        cluster_centroids[order[0]],
                        cluster_centroids[order[1]],
                        cluster_centroids[order[2]],
                        cluster_centroids[order[2]]
                        + v
                        * np.linalg.norm(
                            cluster_centroids[order[2]] - cluster_centroids[order[1]]
                        ),
                    ]
                    self.get_logger().info("Detected 3/4 gates with missing green")
                elif np.all(
                    np.argmax(buoy_colours, axis=1) == np.array([2, 2, 0])
                ):  # wwr
                    gate_poses = [
                        cluster_centroids[order[2]],
                        cluster_centroids[order[1]],
                        cluster_centroids[order[0]],
                        cluster_centroids[order[0]]
                        - v
                        * np.linalg.norm(
                            cluster_centroids[order[0]] - cluster_centroids[order[1]]
                        ),
                    ]
                    output_yaw += np.pi
                    self.get_logger().info("Detected 3/4 gates with missing green")
                elif np.all(
                    np.argmax(buoy_colours, axis=1) == np.array([2, 2, 1])
                ):  # wwg
                    gate_poses = [
                        cluster_centroids[order[0]]
                        - v
                        * np.linalg.norm(
                            cluster_centroids[order[0]] - cluster_centroids[order[1]]
                        ),
                        cluster_centroids[order[0]],
                        cluster_centroids[order[1]],
                        cluster_centroids[order[2]],
                    ]
                    self.get_logger().info("Detected 3/4 gates with missing red")
                elif np.all(
                    np.argmax(buoy_colours, axis=1) == np.array([1, 2, 2])
                ):  # gww
                    gate_poses = [
                        cluster_centroids[order[2]]
                        + v
                        * np.linalg.norm(
                            cluster_centroids[order[2]] - cluster_centroids[order[1]]
                        ),
                        cluster_centroids[order[2]],
                        cluster_centroids[order[1]],
                        cluster_centroids[order[0]],
                    ]
                    output_yaw += np.pi
                    self.get_logger().info("Detected 3/4 gates with missing red")
                else:
                    return None
            return [
                np.mean([gate_poses[0], gate_poses[1]], axis=0),
                np.mean([gate_poses[1], gate_poses[2]], axis=0),
                np.mean([gate_poses[2], gate_poses[3]], axis=0),
            ], output_yaw

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
            gates = sorted(gates, key=lambda x: np.dot(x[:2], self.forward_direction))
            for gate in gates:
                if np.dot(gate[2:], self.forward_direction) < 0:
                    gate[2:] = -gate[2:]
            output_gates.append(gates)
        return output_gates

    def publish_gate_transform(self, gate_detections: List[DetectedObject3D]):
        # Extract position and direction for entrance gate
        if len(gate_detections) != 3:
            return
        self.get_logger().info("Publishing gate transforms", throttle_duration_sec=2.0)
        frame_ids = [
            "gate_left" + ("_ned" if self.is_ned else ""),
            "gate_middle" + ("_ned" if self.is_ned else ""),
            "gate_right" + ("_ned" if self.is_ned else ""),
        ]
        for i, gate in enumerate(gate_detections):
            print(i, gate)
            gate_transform = TransformStamped()
            gate_transform.header.stamp = self.get_clock().now().to_msg()
            gate_transform.header.frame_id = "map" + ("_ned" if self.is_ned else "")
            gate_transform.child_frame_id = frame_ids[i]
            gate_transform.transform.translation.x = (
                gate.hypothesis.kinematics.pose_with_covariance.pose.position.x
            )
            gate_transform.transform.translation.y = (
                gate.hypothesis.kinematics.pose_with_covariance.pose.position.y
            )
            gate_transform.transform.translation.z = 0.0
            gate_transform.transform.rotation = (
                gate.hypothesis.kinematics.pose_with_covariance.pose.orientation
            )
            self.tf_broadcaster.sendTransform(gate_transform)

    def show_buoys(self):
        if not self.running:
            self.get_logger().info("Not running", throttle_duration_sec=2.0)
        if self.vehicle_position is None:
            return
        if len(self.buoys) == 0:
            self.get_logger().info("No buoys detected", throttle_duration_sec=2.0)
            return
        else:
            self.get_logger().info(
                f"{len(self.buoys)} buoys detected : {self.buoys}", throttle_duration_sec=2.0
            )

        # Extract buoy positions for clustering
        positions = (
            np.array([[buoy[0], buoy[1]] for buoy in self.buoys.values()])
            @ self.clustering_T.T
        )

        hierarchical_clusterer = AgglomerativeClustering(
            n_clusters=None,  # Set to None to allow distance-based threshold
            distance_threshold=15,  # Similar to 'eps', defines max distance for clusters, larger than estimate distance0
            linkage="single",  # Linkage method; 'ward', 'complete', 'average', or 'single'
        )
        if len(positions) == 1:
            self.get_logger().info("Only one buoy detected", throttle_duration_sec=2.0)
            return
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
        gate_detections.name = "gate_detector"
        gate_detections.source = self.detector_source
        self.gate_estimate_valid = False
        best_detections = []
        closest_distance = 1000
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
            self.get_logger().info(
                f"Cluster {cluster} with tids {cluster_tids}: {len(cluster_details)}",
                throttle_duration_sec=2.0,
            )
            poses = self.calculate_gate_poses(cluster_details)
            if poses is None:
                continue
            self.gate_estimate_valid = True
            gate_position, yaw = poses
            gate_quat = euler2quat(0, 0, yaw)
            gate_quat = Quaternion(
                w=gate_quat[0], x=gate_quat[1], y=gate_quat[2], z=gate_quat[3]
            )
            self.get_logger().info(
                f"Detected gate at {gate_position} with yaw {yaw}",
                throttle_duration_sec=2.0,
            )
            detections = []
            for i, (gate_name, pos) in enumerate(
                zip(["gate_left", "gate_middle", "gate_right"], gate_position)
            ):
                gate_detection = DetectedObject3D()
                gate_detection.hypothesis.mode = ObjectHypothesis.MODE_TRACKED
                gate_detection.hypothesis.track_id = self.buoy_id_to_gate_id[
                    cluster_tids[0]
                ]
                gate_detection.hypothesis.class_id = self.name_to_id[
                    gate_name
                ]  # gate class id
                gate_detection.hypothesis.kinematics.header = self.header
                gate_pose = Pose()
                gate_pose.position.x = pos[0]
                gate_pose.position.y = pos[1]
                gate_pose.position.z = 0.0

                gate_pose.orientation = gate_quat
                gate_detection.hypothesis.kinematics.pose_with_covariance.pose = (
                    gate_pose
                )
                gate_detection.hypothesis.shape.dimensions.x = 0.5
                gate_detection.hypothesis.shape.dimensions.y = 10.0
                gate_detection.hypothesis.shape.dimensions.z = 1.0

                gate_detection.hypothesis.probability = 0.5

                detections.append(gate_detection)

            p1 = detections[i].hypothesis.kinematics.pose_with_covariance.pose.position
            p2 = self.vehicle_position
            distances = np.mean(
                [
                    np.linalg.norm(np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z]))
                    for _ in range(3)
                ]
            )
            if len(best_detections) == 0 or distances < closest_distance:
                best_detections = detections
                closest_distance = distances
        gate_detections.objects = best_detections
        self.publish_gate_transform(gate_detections.objects)
        self.gate_detections_pub.publish(gate_detections)
        self.gate_estimate_valid_pub.publish(Bool(data=self.gate_estimate_valid))

        # if self.debug:
        #     pass


def main(args=None):
    rclpy.init(args=args)
    buoy_detection = GateDetection()
    rclpy.spin(buoy_detection)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
