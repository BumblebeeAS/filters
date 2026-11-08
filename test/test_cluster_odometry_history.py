"""Delayed poses use real retained odometry, without reusing prior-goal poses."""
import json
from unittest.mock import Mock

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.duration import Duration

from bb_filters.nodes.cluster.cluster_poses_action_node import ClusterPosesActionNode


def start(node, odom='/history/odom'):
    node._start_subscribers(odom_topic=odom, pose_topics=['/history/pose'], sync_tolerance=.05)


def odometry(node, age):
    message = Odometry()
    message.header.stamp = (node.get_clock().now() - Duration(seconds=age)).to_msg()
    message.header.frame_id = 'odom'
    message.child_frame_id = 'body'
    message.pose.pose.orientation.w = 1.
    return message


def pose_at(message, shift_ns=0):
    from rclpy.time import Time
    pose = PoseStamped()
    pose.header.frame_id = 'camera'
    pose.header.stamp = (Time.from_msg(message.header.stamp) + Duration(nanoseconds=shift_ns)).to_msg()
    pose.pose.orientation.w = 1.
    return pose


def test_next_window_replays_odometry_and_logs_exact_match(rclpy_context):
    node = ClusterPosesActionNode()
    try:
        start(node)
        subscription = node._persistent_odom_subscription
        node._cleanup_subscribers()
        old = odometry(node, .75)
        node._remember_odometry(old)  # Arrived while no collection was running.
        start(node)
        assert node._persistent_odom_subscription is subscription
        assert node._num_collected() == 0  # No previous-goal detection samples.
        publisher = Mock()
        node.sync_diagnostics.publisher = publisher
        pose = pose_at(old, 12_000_000)
        node._pose_subscribers[0].signalMessage(pose)
        assert node._num_collected() == 1
        matched_odom, matched_pose = node._stream_data[0][0]
        assert matched_odom is old and matched_pose is pose
        diagnostic = json.loads(publisher.publish.call_args.args[0].data)
        assert diagnostic['odom_minus_observation_s'] == -.012
        assert diagnostic['observation_arrival_age_s'] > .7
        assert node.sync_diagnostics.summary()['/history/pose']['matches'] == 1
        old_relay = node._odom_relay
        node._cleanup_subscribers()
        node._reset_collection()
        start(node)
        assert node._num_collected() == 0
        assert node._odom_relay is not old_relay
        # A live sample reaches only the new synchronizers, not old goal buffers.
        node._remember_odometry(odometry(node, 0))
        assert node._num_collected() == 0
    finally:
        node.destroy_node()


def test_history_does_not_bypass_slop_and_topic_switch_clears_it(rclpy_context):
    node = ClusterPosesActionNode()
    try:
        start(node)
        old = odometry(node, .7)
        node._remember_odometry(old)
        node._cleanup_subscribers()
        start(node)
        node._pose_subscribers[0].signalMessage(pose_at(old, 100_000_000))
        assert node._num_collected() == 0
        node._cleanup_subscribers()
        start(node, '/different/odom')
        assert not node._odom_history
        node._pose_subscribers[0].signalMessage(pose_at(old))
        assert node._num_collected() == 0
    finally:
        node.destroy_node()


def test_stale_history_and_clock_reset_are_bounded(rclpy_context):
    node = ClusterPosesActionNode()
    try:
        start(node)
        old = odometry(node, 10.)
        node._remember_odometry(old)
        node._cleanup_subscribers()
        start(node)
        node._pose_subscribers[0].signalMessage(pose_at(old))
        assert node._num_collected() == 0
        fresh = odometry(node, 0.)
        node._remember_odometry(fresh)
        assert len(node._odom_history) == 1
        backwards = odometry(node, 1.)
        node._remember_odometry(backwards)
        assert list(node._odom_history) == [backwards]
    finally:
        node.destroy_node()
