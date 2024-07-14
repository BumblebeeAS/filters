
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from bb_perception_msgs.msg import DetectedObject3DArray, DetectedObject3D
from geometry_msgs.msg import Vector3
from message_filters import TimeSynchronizer, Subscriber
from typing import List
from ml_detector.schema_validator import get_config, load_schema
from ml_detector.helpers.log import RclLogHandler
import logging
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
import tf2_ros

class DetectedObject3DArrayVisNode(Node):
    def __init__(self):
        super().__init__("detected_object_3d_visualization_node")
        self.declare_parameter("input_detections_topics", ["/asv4/vision/detections_3d_zed_left"])
        self.declare_parameter("output_markers_topic", "debug_markers_topic")
        self.declare_parameter("objects_config", "robotx.yaml")

        self.input_detections_topics = (
            self.get_parameter("input_detections_topics")
            .get_parameter_value()
            .string_array_value
        )
        self.output_markers_topic = (
            self.get_parameter("output_markers_topic").get_parameter_value().string_value
        )
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
        r = ((hash(i) >> 16) % 256)/255.0
        g = ((hash(i) >> 8) % 256)/255.0
        b = (hash(i) % 256)/255.0
        return ColorRGBA(r=float(r), g=float(g), b=float(b), a=1.0)

    def callback(self, *detection_msgs):
        markers = MarkerArray()
        i = -1
        for detection_msg in detection_msgs:
            objects: List[DetectedObject3D] = detection_msg.objects

            for detection in objects:
                i+=1
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
                marker.pose = detection.hypothesis.kinematics.pose_with_covariance.pose
                print(detection.hypothesis.shape.dimensions)
                marker.scale = detection.hypothesis.shape.dimensions
                marker.color = self.get_color(class_name)
                markers.markers.append(marker)

                transform = tf2_ros.TransformStamped()
                transform.header.stamp = detection.hypothesis.kinematics.header.stamp
                transform.header.frame_id = "map"
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