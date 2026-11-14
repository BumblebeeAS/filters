#!/usr/bin/env python3
"""Continuously cluster a bounded, source-time window from several pose streams."""

from collections import deque
from copy import deepcopy
import math

import rclpy
from bb_perception_msgs.msg import ClusterPoseResultArray
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster

from bb_filters.nodes.cluster.cluster_poses_node import (
    ClusterParams, ClusterPosesNode, fill_cluster_result_array,
)
from bb_filters.utils.cluster.cluster import ClusterSortKey


def stamp_ns(message):
    return message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec


class RollingPoseClusterNode(ClusterPosesNode):
    def __init__(self):
        super().__init__('rolling_pose_cluster_node')
        def param(name, default):
            return self.declare_parameter(name, default).value
        topics = param('pose_stamped_topics', ['/uav/wide_a/asv/pose', '/uav/narrow_a/asv/pose'])
        odom_topic = param('odom_topic', '/uav/odom_ned')
        self.local_frame = param('local_frame', 'odom_ned')
        self.child_frame = param('clustered_child_frame_id', 'asv/clustered')
        self.window = param('window_seconds', 5.0)
        interval = param('cluster_interval', 0.5)
        self.capacity = param('max_poses_per_topic', 1000)
        self.params = ClusterParams(
            min_poses=param('min_poses', 5), min_cluster_size=param('min_cluster_size', 5),
            min_samples=param('min_samples', 2),
            cluster_selection_epsilon=param('cluster_selection_epsilon', 0.1), top_k=1,
        )
        if (not topics or not self.local_frame or not self.child_frame
                or not all(math.isfinite(v) and v > 0 for v in (self.window, interval))
                or self.capacity < max(self.params.min_poses, self.params.min_cluster_size)
                or self.params.min_poses < 1 or self.params.min_cluster_size < 2 or self.params.min_samples < 1
                or not math.isfinite(self.params.cluster_selection_epsilon)
                or self.params.cluster_selection_epsilon < 0):
            raise ValueError('Invalid rolling window, interval, frame, or sample bounds')
        self.pose_pub = self.create_publisher(PoseStamped, param('output_pose_topic', 'clustered/pose'), 10)
        self.result_pub = self.create_publisher(ClusterPoseResultArray, param('output_results_topic', 'clustered/results'), 10)
        self.tf_pub = TransformBroadcaster(self)
        self.last_published = 0
        self._start_subscribers(odom_topic=odom_topic, pose_topics=topics,
                                sync_tolerance=param('sync_tolerance', 0.1),
                                sync_queue_size=self._sync_queue_size, max_detection_age_s=self.window)
        self._stream_data = [deque(maxlen=self.capacity) for _ in topics]
        self.timer = self.create_timer(interval, self._tick)
        self.get_logger().info(f'Rolling {self.window}s cluster from {topics} in {self.local_frame}')

    def _collect_pose(self, odom_msg, pose_msg, stream_index):
        stamp = stamp_ns(pose_msg)
        now = self.get_clock().now().nanoseconds
        if (odom_msg.header.frame_id != self.local_frame or not pose_msg.header.frame_id
                or stamp <= 0 or stamp < now - self.window * 1e9 or stamp > now + 1e8):
            self.get_logger().warning('Discarding stale, future, or wrong-frame pose', throttle_duration_sec=5.)
            return
        stream = self._stream_data[stream_index]
        if any(stamp_ns(pose) == stamp for _, pose in stream):
            return
        stream.append((odom_msg, pose_msg))

    def _tick(self):
        now = self.get_clock().now().nanoseconds
        if now < self.last_published:
            for stream in self._stream_data:
                stream.clear()
            self.last_published = 0
            return
        cutoff = now - self.window * 1e9
        samples = []
        for stream in self._stream_data:
            retained = [pair for pair in stream if cutoff <= stamp_ns(pair[1]) <= now + 1e8]
            stream.clear()
            stream.extend(retained)
            samples.extend(retained)
        if len(samples) < self.params.min_poses:
            return
        samples.sort(key=lambda pair: stamp_ns(pair[1]))
        clustered, transformed, collected = self._run_clustering(samples, self.params)
        if not clustered:
            return
        pose, result = clustered[0]
        newest = max((transformed[i] for i in result.idxs), key=stamp_ns)
        if stamp_ns(newest) <= self.last_published:
            return
        header = deepcopy(newest.header)
        header.frame_id = self.local_frame
        self.pose_pub.publish(PoseStamped(header=header, pose=pose))
        results = ClusterPoseResultArray()
        fill_cluster_result_array(results, clustered, collected, header, ClusterSortKey.NUM_CLUSTER_POSES)
        self.result_pub.publish(results)
        transform = TransformStamped(header=header, child_frame_id=self.child_frame)
        transform.transform.translation.x = pose.position.x
        transform.transform.translation.y = pose.position.y
        transform.transform.translation.z = pose.position.z
        transform.transform.rotation = pose.orientation
        self.tf_pub.sendTransform(transform)
        self.last_published = stamp_ns(newest)


def main(args=None):
    rclpy.init(args=args)
    node = RollingPoseClusterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
