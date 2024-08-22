#!/usr/bin/env python3
import re
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
from bb_perception_msgs.msg import (
    DetectedObject3DArray,
    DetectedObject3D,
    ObjectHypothesis,
    ObjectKinematics,
    DetectorSource,
    Shape,
    LabelInfo,
)
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Vector3


class SimObstacleRepublisher(Node):

    def __init__(self):
        super().__init__("obstacle_detector")
        self.publisher_ = self.create_publisher(
            DetectedObject3DArray, "/robotx/detections", 10
        )
        # refer to robotx.yaml
        self.object_id_map = {
            ".*obstacle.*": "black_sphere",
            ".*red_bound.*": "red_cylinder",
            ".*green_bound.*": "green_cylinder",
            ".*dock.*": "dock",
            ".*light_buoy.*": "light_tower"
        }

        self.labels = None
        self.id_to_name = None
        self.name_to_id = None
        self.label_info_sub = self.create_subscription(
            LabelInfo, "/label_info", self.label_info_sub, qos_profile_sensor_data
        )
        self.subscription = self.create_subscription(
            Odometry, "/robotx/obstacles", self.odometry_callback, 10
        )

    def label_info_sub(self, msg):
        self.labels = msg
        self.id_to_name = {
            obj.label: obj.label_name for obj in self.labels.classes
        }
        self.name_to_id = {v: k for k, v in self.id_to_name.items()}

    def get_shape_from_label(self, label_name):
        """Returns the shape associated with the given label name."""
        if self.labels is None:
            return None

        for vision_class in self.labels.classes:
            if vision_class.label_name == label_name:
                return vision_class.shape

        return None

    def odometry_callback(self, msg):
        if self.labels is None:
            self.get_logger().warn("labels not found, skipping")
            return
        detection_array = DetectedObject3DArray()
        detection_array.header = msg.header
        detection_array.header.frame_id = "world"
        detection_array.name = "gz sim ground truth"
        detection_array.source = DetectorSource(
            category=DetectorSource.LIDAR, sensor_name="sim_gt", frame_id="world"
        )  # Example source
        detection_array.sensor_pose = Pose()

        detected_object = DetectedObject3D()
        kinematics = ObjectKinematics()
        kinematics.header = msg.header
        kinematics.pose_with_covariance.pose = msg.pose.pose
        kinematics.twist_with_covariance.twist = msg.twist.twist

        detected_object.hypothesis = ObjectHypothesis(
            mode=ObjectHypothesis.MODE_DETECTED,
            class_id=1,  # Example class ID
            kinematics=kinematics,
            classification_age=0,
            track_id=0,
            probability=0.9,
        )
        detected_object.color = 0
        detected_object.id = str(msg.child_frame_id)

        # Determine object shape from label information
        for pattern, label_name in self.object_id_map.items():
            if re.match(pattern, msg.child_frame_id):
                detected_object.hypothesis.class_id = self.name_to_id[label_name]
                object_shape = self.get_shape_from_label(label_name)
                if object_shape:
                    detected_object.hypothesis.shape = object_shape
                else:
                    self.get_logger().warn(
                        f"Shape not found for label {label_name}, using default shape."
                    )
                    detected_object.hypothesis.shape = Shape(
                        category=Shape.CUBOID,  # Assuming CUBOID for default
                        dimensions=Vector3(x=0.5, y=0.5, z=0.5),  # Default dimensions
                        dimension_variance=Vector3(
                            x=0.01, y=0.01, z=0.01
                        ),  # Example variance
                    )
                break
        else:
            self.get_logger().warn(
                f"No matching pattern for {msg.child_frame_id}, using default shape."
            )
            detected_object.hypothesis.shape = Shape(
                category=Shape.CUBOID,  # Assuming CUBOID for default
                dimensions=Vector3(x=0.5, y=0.5, z=0.5),  # Default dimensions
                dimension_variance=Vector3(x=0.01, y=0.01, z=0.01),  # Example variance
            )

        detection_array.objects.append(detected_object)

        self.publisher_.publish(detection_array)


def main(args=None):
    rclpy.init(args=args)
    obstacle_detector = SimObstacleRepublisher()
    rclpy.spin(obstacle_detector)
    obstacle_detector.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
