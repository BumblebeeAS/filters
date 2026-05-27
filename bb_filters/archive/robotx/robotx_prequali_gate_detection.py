#!/usr/bin/env python3

import rclpy
import math
from rclpy.node import Node
from bb_perception_msgs.msg import DetectedObject3DArray
from geometry_msgs.msg import Pose, TransformStamped
from geometry_msgs.msg import Pose
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA

from tf2_ros import TransformBroadcaster
import time


class PreQualiGateDetectionNode(Node):
    def __init__(self):
        super().__init__('prequali_gate_detection')

        # Declare and get parameters
        self.declare_parameter("gates_topic", "/asv4/vision/red_green_gate_detections")
        self.declare_parameter("output_markers_topic", "/prequaligates_markers")
        self.declare_parameter("predefined_coordinate", [39.7, -9.0, 0.0])  # Entry reference point (map frame)
        self.declare_parameter("min_exit_distance", 10.0)  # Minimum distance between entry and exit gates
        self.declare_parameter("max_exit_distance", 30.0)  # Maximum distance between entry and exit gates
        self.declare_parameter("max_offset", 10.0)  # Maximum lateral offset for the exit gate
        self.declare_parameter("publish_interval", 1.0)  # Interval (in seconds) for publishing transforms

        # Parameters
        self.gates_topic = self.get_parameter("gates_topic").get_parameter_value().string_value
        self.output_markers_topic = self.get_parameter("output_markers_topic").get_parameter_value().string_value
        self.predefined_coordinate = self.get_parameter("predefined_coordinate").get_parameter_value().double_array_value
        self.min_exit_distance = self.get_parameter("min_exit_distance").get_parameter_value().double_value
        self.max_exit_distance = self.get_parameter("max_exit_distance").get_parameter_value().double_value
        self.max_offset = self.get_parameter("max_offset").get_parameter_value().double_value
        self.publish_interval = self.get_parameter("publish_interval").get_parameter_value().double_value

        # Initialize variables
        self.gate_buffer = []  # Store gates over the last 10 seconds
        self.buffer_duration = 10.0  # 10-second window for averaging
        self.last_publish_time = time.time()  # Track last publish time

        # Create a TransformBroadcaster for publishing TF
        self.tf_broadcaster = TransformBroadcaster(self)

        # Publishers and Subscribers
        self.publisher = self.create_publisher(MarkerArray, self.output_markers_topic, 10)
        self.gates_subscriber = self.create_subscription(
            DetectedObject3DArray, self.gates_topic, self.gates_callback, 10
        )

    def gates_callback(self, msg: DetectedObject3DArray):
        # List to hold gate markers
        gates = []
        for obj in msg.objects:
            gates.append(self.pose_to_coordinates(obj.hypothesis.kinematics.pose_with_covariance.pose))

        if not gates:
            # self.get_logger().info("No gates detected.")
            return

        entry_gate = self.find_closest_gate(gates, self.predefined_coordinate)
        exit_gate = self.find_exit_gate(gates, entry_gate)

        # Visualize gates
        if entry_gate or exit_gate:
            markers = MarkerArray()
            if entry_gate:
                markers.markers.append(self.create_marker(entry_gate, "entry_gate", 0, ColorRGBA(r=0., g=1., b=0., a=1.)))  # Green
            if exit_gate:
                markers.markers.append(self.create_marker(exit_gate, "exit_gate", 1, ColorRGBA(r=1., g=0., b=0., a=1.)))   # Red
            self.publisher.publish(markers)

            # Publish TF for entry and exit gates at regular intervals
            if time.time() - self.last_publish_time >= self.publish_interval:
                if entry_gate is not None:
                    self.publish_tf(entry_gate, "entry_gate")
                if exit_gate is not None:
                    self.publish_tf(exit_gate, "exit_gate")
                self.last_publish_time = time.time()

    def clean_old_gates(self, current_time):
        """Remove gates from the buffer that are older than 10 seconds."""
        current_time_sec = self.get_clock().now().to_msg().sec
        self.gate_buffer = [(gates, timestamp) for gates, timestamp in self.gate_buffer if
                            (current_time_sec - timestamp.sec) < self.buffer_duration]

    def find_averaged_gate(self, find_gate_func, *args):
        """Averages the gate positions over the 10-second window."""
        accumulated_gate = [0.0, 0.0, 0.0]
        valid_gate_count = 0

        for gates, _ in self.gate_buffer:
            gate = find_gate_func(gates, *args)
            if gate:
                accumulated_gate[0] += gate[0]
                accumulated_gate[1] += gate[1]
                accumulated_gate[2] += gate[2]
                valid_gate_count += 1

        if valid_gate_count > 0:
            return [accumulated_gate[0] / valid_gate_count,
                    accumulated_gate[1] / valid_gate_count,
                    accumulated_gate[2] / valid_gate_count]
        return None


    def pose_to_coordinates(self, pose: Pose):
        """Converts Pose to a tuple of (x, y, z) coordinates."""
        return (pose.position.x, pose.position.y, pose.position.z)

    def find_closest_gate(self, gates, predefined_coordinate):
        """Find the gate closest to the predefined coordinate."""
        closest_gate = None
        closest_distance = float('inf')
        for gate in gates:
            distance = self.calculate_distance(gate, predefined_coordinate)
            if distance < closest_distance:
                closest_gate = gate
                closest_distance = distance
        return closest_gate

    def find_exit_gate(self, gates, entry_gate):
        """Find the exit gate within a certain range in front of the entry gate."""
        exit_gate = None
        for gate in gates:
            if gate == entry_gate:
                continue
            # Calculate distance and offset
            distance = self.calculate_distance(gate, entry_gate)
            lateral_offset = abs(gate[1] - entry_gate[1])
            if self.min_exit_distance <= distance <= self.max_exit_distance and lateral_offset <= self.max_offset:
                exit_gate = gate
                break
        return exit_gate

    def calculate_distance(self, coord1, coord2):
        """Calculates Euclidean distance between two 3D coordinates."""
        return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2 + (coord1[2] - coord2[2]) ** 2)

    def create_marker(self, gate, ns, marker_id, color):
        """Create a marker for the gate for visualization."""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD  
        marker.pose.position.x = gate[0]
        marker.pose.position.y = gate[1]
        marker.pose.position.z = gate[2]
        marker.scale.x = 1.0  # Assuming gate size is 1m for visualization
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color = color
        return marker

    def publish_tf(self, gate_position, gate_name):
        """Publish a tf for a gate."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = gate_name
        t.transform.translation.x = gate_position[0]
        t.transform.translation.y = gate_position[1]
        t.transform.translation.z = gate_position[2]
        t.transform.rotation.w = 1.0  # Identity rotation (no orientation for the gate)
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = PreQualiGateDetectionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
