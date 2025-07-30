#!/usr/bin/env python3
from operator import attrgetter
from typing import List

import numpy as np
import rclpy
import tf2_ros
from bb_filters.cluster import (
    get_average_pose,
    get_idxs_in_largest_cluster,
    get_position_tuple_from_pose,
)
from geometry_msgs.msg import (
    PoseArray,
    PoseWithCovarianceStamped,
    Quaternion,
    TransformStamped,
    Vector3,
)
from message_filters import Cache, Subscriber
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from sklearn.cluster import HDBSCAN


class ClusterPosesNode(Node):
    def __init__(self):
        super().__init__("cluster_poses_node")

        self.declare_parameter(
            "input_pose_topic",
            "input_pose",
            ParameterDescriptor(
                description="Input ROS topic for PoseWithCovarianceStamped messages."
            ),
        )
        self.declare_parameter(
            "output_pose_topic",
            "output_pose",
            ParameterDescriptor(
                description="Output ROS topic for filtered PoseWithCovarianceStamped."
            ),
        )
        self.declare_parameter(
            "output_pose_array_topic",
            "output_pose_array",
            ParameterDescriptor(
                description="Output ROS topic for array of input poses fed to clustering. For debugging."
            ),
        )
        self.declare_parameter(
            "child_frame_id",
            "auv4/gate/clustered",
            ParameterDescriptor(
                description="Child frame id of the published transform."
            ),
        )
        self.declare_parameter(
            "cluster_interval",
            1.0,
            ParameterDescriptor(
                description="Time interval in seconds between computations of filtered pose."
            ),
        )
        self.declare_parameter(
            "num_poses",
            20,
            ParameterDescriptor(description="Number of poses to cache."),
        )
        self.declare_parameter(
            "min_cluster_size",
            2,
            ParameterDescriptor(description="Argument for sklearn's HDBSCAN."),
        )
        self.declare_parameter(
            "min_samples",
            1,
            ParameterDescriptor(description="Argument for sklearn's HDBSCAN."),
        )

        input_pose_topic = (
            self.get_parameter("input_pose_topic").get_parameter_value().string_value
        )
        output_pose_topic = (
            self.get_parameter("output_pose_topic").get_parameter_value().string_value
        )
        output_pose_array_topic = (
            self.get_parameter("output_pose_array_topic")
            .get_parameter_value()
            .string_value
        )
        self.child_frame_id = (
            self.get_parameter("child_frame_id").get_parameter_value().string_value
        )
        cluster_interval = (
            self.get_parameter("cluster_interval").get_parameter_value().double_value
        )
        num_poses = self.get_parameter("num_poses").get_parameter_value().integer_value
        self.min_cluster_size = (
            self.get_parameter("min_cluster_size").get_parameter_value().integer_value
        )
        self.min_samples = (
            self.get_parameter("min_samples").get_parameter_value().integer_value
        )

        self.pose_subscriber = Subscriber(
            self, PoseWithCovarianceStamped, input_pose_topic
        )
        self.pose_cache = Cache(self.pose_subscriber, cache_size=num_poses)
        self.pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped, output_pose_topic, 10
        )
        self.pose_array_publisher = self.create_publisher(
            PoseArray, output_pose_array_topic, 10
        )
        self.br = tf2_ros.TransformBroadcaster(self)
        self.cluster_tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.create_timer(
            cluster_interval,
            self.cluster_timer_callback,
        )

        self.hdbscan = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=0.05,
            allow_single_cluster=True,
            store_centers="centroid",
        )

    def cluster_timer_callback(self):
        oldest_time = self.pose_cache.getOldestTime()
        latest_time = self.pose_cache.getLastestTime()

        if oldest_time is None or latest_time is None:
            self.get_logger().warn("No poses in cache.")
            return

        pose_msgs: List[PoseWithCovarianceStamped] = self.pose_cache.getInterval(
            oldest_time, latest_time
        )

        min_num_poses = max(self.min_cluster_size, self.min_samples)
        if len(pose_msgs) < min_num_poses:
            self.get_logger().warn(
                f"Not enough poses to cluster. Received: {len(pose_msgs)}. Required: {min_num_poses}."
            )
            return

        # Publish the array of poses for debugging
        pose_array_msg = PoseArray()
        pose_array_msg.header = pose_msgs[-1].header
        pose_array_msg.poses = [pose.pose.pose for pose in pose_msgs]
        self.pose_array_publisher.publish(pose_array_msg)

        positions = np.array([get_position_tuple_from_pose(pose) for pose in pose_msgs])
        filtered_idxs = get_idxs_in_largest_cluster(self.hdbscan, positions)
        if len(filtered_idxs) == 0:
            self.get_logger().warn("No clusters found.")
            return
        filtered_poses = [pose_msgs[i] for i in filtered_idxs]

        avg_pose = get_average_pose(filtered_poses, self.get_logger())
        latest_msg: PoseWithCovarianceStamped = self.pose_cache.getLast()
        avg_pose.header = latest_msg.header
        self.pose_publisher.publish(avg_pose)

        transform_stamped = TransformStamped()
        transform_stamped.header = latest_msg.header
        transform_stamped.child_frame_id = self.child_frame_id
        t = attrgetter("x", "y", "z")(avg_pose.pose.pose.position)
        qx, qy, qz, qw = attrgetter("x", "y", "z", "w")(avg_pose.pose.pose.orientation)
        transform_stamped.transform.translation = Vector3(x=t[0], y=t[1], z=t[2])
        transform_stamped.transform.rotation = Quaternion(x=qx, y=qy, z=qz, w=qw)

        self.br.sendTransform(transform_stamped)


def main(args=None):
    rclpy.init(args=args)
    node = ClusterPosesNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
