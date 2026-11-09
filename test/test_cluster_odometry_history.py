"""Delayed detections must match recent odometry from the same topic and epoch."""
from rclpy.duration import Duration
from rclpy.time import Time

from cluster_test_utils import make_odom, make_pose


def start(node, topic='/history/odom'):
    node._start_subscribers(odom_topic=topic, pose_topics=['/history/pose'], sync_tolerance=.05)


def odometry(node, age):
    return make_odom((node.get_clock().now() - Duration(seconds=age)).to_msg())


def pose_at(odom, shift_ns=0):
    return make_pose((Time.from_msg(odom.header.stamp) + Duration(nanoseconds=shift_ns)).to_msg(), 0.)


def test_delayed_detection_uses_retained_odometry(cluster_node):
    node = cluster_node
    start(node)
    node._cleanup_subscribers()
    old = odometry(node, .75)
    node._odometry.remember(old)  # Arrived between collections.
    start(node)
    node._pose_subscribers[0].signalMessage(pose_at(old, 12_000_000))
    assert node._num_collected() == 1
    assert node._stream_data[0][0][0].header.stamp == old.header.stamp


def test_history_rejects_stale_mismatched_and_previous_epoch_samples(cluster_node):
    node = cluster_node
    start(node)
    old = odometry(node, 10.)
    node._odometry.remember(old)
    node._cleanup_subscribers()
    start(node)
    node._pose_subscribers[0].signalMessage(pose_at(old))
    assert node._num_collected() == 0  # Older than the five-second history.
    fresh, backwards = odometry(node, 0.), odometry(node, 1.)
    node._odometry.remember(fresh)
    node._odometry.remember(backwards)
    node._cleanup_subscribers()
    start(node)
    node._pose_subscribers[0].signalMessage(pose_at(fresh))
    node._pose_subscribers[0].signalMessage(pose_at(backwards, 100_000_000))
    assert node._num_collected() == 0  # Prior epoch and outside the sync tolerance.
    node._cleanup_subscribers()
    start(node, '/different/odom')
    node._pose_subscribers[0].signalMessage(pose_at(backwards))
    assert node._num_collected() == 0
