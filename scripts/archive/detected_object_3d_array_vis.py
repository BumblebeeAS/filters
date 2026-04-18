#!/usr/bin/env python3

import logging
from pathlib import Path
from typing import List

from copy import deepcopy

import rclpy
import tf2_ros
from ament_index_python.packages import get_package_share_directory
from bb_perception_msgs.msg import DetectedObject3D, DetectedObject3DArray, ObjectHypothesis
from cv_bridge import CvBridge
from geometry_msgs.msg import Vector3
from message_filters import Subscriber, TimeSynchronizer
from ml_detector.helpers.log import RclLogHandler
from ml_detector.schema_validator import get_config, load_schema
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration


class DetectedObject3DArrayVisNode(Node):
    def __init__(self):
        super().__init__("detected_object_3d_visualization_node")
        self.declare_parameter("input_detections_topics", ["/asv4/vision/detections_3d_zed_left"])
        self.declare_parameter("output_markers_topic", "debug_markers_topic")
        self.declare_parameter("objects_config", "robotx.yaml")
        self.declare_parameter("publish_tf", False)
        # if publish_tf_unique true, publish tf as the object name. else, publish as object name _ id
        self.declare_parameter("publish_tf_unique", False)

        self.input_detections_topics = (
            self.get_parameter("input_detections_topics")
            .get_parameter_value()
            .string_array_value
        )
        self.output_markers_topic = (
            self.get_parameter("output_markers_topic").get_parameter_value().string_value
        )
        self.publish_tf = self.get_parameter("publish_tf").get_parameter_value().bool_value
        self.publish_tf_unique = self.get_parameter("publish_tf_unique").get_parameter_value().bool_value
        self.logger = logging.getLogger("detected_object_3d_visualization")
        self.logger.level = logging.INFO
        self.logger.propagate = False
        self.logger.addHandler(
            RclLogHandler(self.get_logger(), "detected_object_3d_visualization")
        )
        self.logger.info("Initializing 3D Det Vis Node...")

        objects_schema_path = (
            Path(get_package_share_directory("ml_detector"))
            / "configs"
            / "objects_schema.json"
        )
        self.objects_schema = load_schema(objects_schema_path)
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
        self.bridge = CvBridge()

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.publisher = self.create_publisher(
            MarkerArray,
            self.output_markers_topic, 10)

        self.tf_broadcast = tf2_ros.TransformBroadcaster(self)

        self.detection_subscribers = [
            Subscriber(
                self, DetectedObject3DArray, topic, qos_profile=qos
            ) for topic in self.input_detections_topics
        ]
        self.time_sync = TimeSynchronizer(
            [*self.detection_subscribers], 10
        )
        self.time_sync.registerCallback(self.callback)

    def get_color(self, i):
        # random color
        r = ((hash(i) * 37) % 256) / 255.0
        g = ((hash(i) * 57) % 256) / 255.0
        b = ((hash(i) * 79) % 256) / 255.0
        return ColorRGBA(r=float(r), g=float(g), b=float(b), a=1.0)

    def callback(self, *detection_msgs):
        markers = MarkerArray()
        i = -1
        for detection_msg in detection_msgs:
            objects: List[DetectedObject3D] = detection_msg.objects

            for detection in objects:
                i += 1
                class_name = self.id_to_name[detection.hypothesis.class_id]
                marker = Marker()
                marker.header.frame_id = detection.hypothesis.kinematics.header.frame_id
                marker.header.stamp = detection.hypothesis.kinematics.header.stamp
                marker.ns = class_name
                marker.id = i
                marker.type = Marker.CUBE
                if "cylinder" in class_name:
                    marker.type = Marker.CYLINDER
                elif "sphere" in class_name:
                    marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose = deepcopy(detection.hypothesis.kinematics.pose_with_covariance.pose)
                marker.pose.position.z += detection.hypothesis.shape.dimensions.z / 2
                marker.scale = deepcopy(detection.hypothesis.shape.dimensions)
                marker.scale.x = max(0.5, marker.scale.x)
                marker.scale.y = max(0.5, marker.scale.y)
                marker.scale.z = max(0.5, marker.scale.z)
                tid = detection.hypothesis.track_id
                if detection.hypothesis.mode == ObjectHypothesis.MODE_DETECTED:
                    marker.color = self.get_color(detection.hypothesis.class_id)
                    marker.color.a = 0.2
                else:
                    marker.color = self.get_color(tid)

                # Create an arrow Marker for the yaw direction
                arrow_marker = Marker()
                arrow_marker.header.frame_id = marker.header.frame_id
                arrow_marker.header.stamp = marker.header.stamp
                arrow_marker.ns = f"{class_name}_arrow"
                arrow_marker.id = i + 2000  # Ensure unique ID
                arrow_marker.type = Marker.ARROW
                arrow_marker.action = Marker.ADD
                arrow_marker.pose = deepcopy(marker.pose)  # Use the same pose as the object
                arrow_marker.scale.x = 1.0  # Length of the arrow (forward)
                arrow_marker.scale.y = 0.1  # Width of the arrow shaft
                arrow_marker.scale.z = 0.1  # Height of the arrow shaft
                arrow_marker.color.r = 1.0  # Red arrow for yaw
                arrow_marker.color.g = 0.0
                arrow_marker.color.b = 0.0
                arrow_marker.color.a = 1.0  # Make the arrow fully visible
                markers.markers.append(arrow_marker)

                # Add the object and text marker as before
                markers.markers.append(marker)
                text_marker = Marker()
                text_marker.header.frame_id = marker.header.frame_id
                text_marker.header.stamp = marker.header.stamp
                text_marker.ns = f"{class_name}_text"
                text_marker.id = i + 1000  # Ensure unique ID
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.text = f"{class_name}: {tid}" if tid else class_name
                text_marker.action = Marker.ADD
                text_marker.pose.position = deepcopy(marker.pose.position)
                text_marker.pose.position.z += marker.scale.z + 0.5  # Lift text above the object
                text_marker.scale.z = 1.0  # Set the scale of the text
                text_marker.color = marker.color
                text_marker.color.a = 1.0  # Make text fully opaque
                text_marker.lifetime = marker.lifetime
                markers.markers.append(text_marker)

                if self.publish_tf:
                    transform = tf2_ros.TransformStamped()
                    transform.header.stamp = detection.hypothesis.kinematics.header.stamp
                    transform.header.frame_id = detection.hypothesis.kinematics.header.frame_id
                    if self.publish_tf_unique:
                        transform.child_frame_id = f"{class_name}"
                    else:
                        transform.child_frame_id = f"{class_name}_{i}"
                    transform.transform.translation = Vector3(
                        x=marker.pose.position.x,
                        y=marker.pose.position.y,
                        z=marker.pose.position.z
                    )
                    transform.transform.rotation = marker.pose.orientation
                    self.tf_broadcast.sendTransform(transform)

        self.publisher.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = DetectedObject3DArrayVisNode()
    rclpy.spin(node)
    # node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
