"""Delayed poses use real retained odometry, without reusing prior-goal poses."""
import json
import time
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


def test_next_window_replays_odometry_and_summarizes_exact_match(rclpy_context, monkeypatch):
    node = ClusterPosesActionNode()
    try:
        start(node)
        subscription = node._odometry.subscription
        node._cleanup_subscribers()
        old = odometry(node, .75)
        node._odometry.remember(old)  # Arrived while no collection was running.
        start(node)
        assert node._odometry.subscription is subscription
        assert node._num_collected() == 0  # No previous-goal detection samples.
        logger = Mock()
        monkeypatch.setattr(node, 'get_logger', lambda: logger)
        node.sync_diagnostics.last_log = time.monotonic() - 5.
        pose = pose_at(old, 12_000_000)
        node._pose_subscribers[0].signalMessage(pose)
        assert node._num_collected() == 1
        matched_odom, matched_pose = node._stream_data[0][0]
        assert matched_odom is old and matched_pose is pose
        summary = node.sync_diagnostics.summary()
        diagnostic = summary['/history/pose']
        assert diagnostic['matches'] == 1
        assert diagnostic['abs_delta_ms_p50_p95_max'] == [12., 12., 12.]
        assert diagnostic['arrival_age_ms_p50_p95_max'][0] > 700.
        logger.info.assert_any_call('Accepted timestamp matches: ' + json.dumps(summary))
        node.sync_diagnostics.record(pose, old, '/history/pose')
        logger.info.assert_called_once()  # New pairs do not log again before the interval.
        old_relay = node._odometry.relay
        node._cleanup_subscribers()
        node._reset_collection()
        start(node)
        assert node._num_collected() == 0
        assert node._odometry.relay is not old_relay
        # A live sample reaches only the new synchronizers, not old goal buffers.
        node._odometry.remember(odometry(node, 0))
        assert node._num_collected() == 0
    finally:
        node.destroy_node()


def test_history_does_not_bypass_slop_and_topic_switch_clears_it(rclpy_context):
    node = ClusterPosesActionNode()
    try:
        start(node)
        old = odometry(node, .7)
        node._odometry.remember(old)
        node._cleanup_subscribers()
        start(node)
        node._pose_subscribers[0].signalMessage(pose_at(old, 100_000_000))
        assert node._num_collected() == 0
        node._cleanup_subscribers()
        start(node, '/different/odom')
        assert not node._odometry.messages
        node._pose_subscribers[0].signalMessage(pose_at(old))
        assert node._num_collected() == 0
    finally:
        node.destroy_node()


def test_stale_history_and_clock_reset_are_bounded(rclpy_context):
    node = ClusterPosesActionNode()
    try:
        start(node)
        old = odometry(node, 10.)
        node._odometry.remember(old)
        node._cleanup_subscribers()
        start(node)
        node._pose_subscribers[0].signalMessage(pose_at(old))
        assert node._num_collected() == 0
        fresh = odometry(node, 0.)
        node._odometry.remember(fresh)
        assert len(node._odometry.messages) == 1
        backwards = odometry(node, 1.)
        node._odometry.remember(backwards)
        assert list(node._odometry.messages) == [backwards]
    finally:
        node.destroy_node()
