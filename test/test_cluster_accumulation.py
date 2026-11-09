"""Periodic feedback shares one goal collection; completion always clears it."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from bb_perception_msgs.action import ClusterPosesStreamAction
from rclpy.time import Time
from rclpy.action import GoalResponse

from cluster_test_utils import identity_tf_static, make_odom, make_pose


def goal():
    g = ClusterPosesStreamAction.Goal()
    g.collection_duration = 2.
    p = g.params
    p.odom_topic = '/accumulation/odom'
    p.pose_stamped_topics = ['/accumulation/pose']
    p.clustered_child_frame_ids = ['accumulated_pad']
    p.sync_tolerance = .05
    p.min_poses = p.min_cluster_size = 5
    p.min_samples = 1
    p.cluster_selection_epsilon = .1
    return g


def start(node, request):
    handle = SimpleNamespace(publish_feedback=Mock())
    assert node._accept_stream_goal(request) == GoalResponse.ACCEPT
    node._goal.handle = handle
    node._start_goal()
    return handle, node._goal.future


def feed(node, indices, world_x=1.):
    for i in indices:
        odom = make_odom(Time(seconds=100 + i*.02).to_msg())
        odom.pose.pose.position.x = 50. + i*2.
        # The same world pad observed from widely separated vehicle positions.
        pose = make_pose(odom.header.stamp, world_x + i*.001 - odom.pose.pose.position.x, 2., 3.)
        node._odometry.remember(odom)
        node._pose_subscribers[0].signalMessage(pose)


@pytest.fixture
def node(cluster_node):
    cluster_node._tf_buffer.set_transform_static(identity_tf_static().transforms[0], 'test')
    return cluster_node


def snapshot(node, handle):
    node._goal.next_cluster = 0.
    node._step_collection()
    return handle.publish_feedback.call_args.args[0].cluster_results


def test_periodic_feedback_uses_whole_collection_and_original_odometry(node):
    h, result = start(node, goal())
    feed(node, range(4))
    assert not snapshot(node, h).results
    assert not result.done() and node._num_collected() == 4
    feed(node, range(4, 13))
    r = snapshot(node, h)
    assert len(r.results) == 1
    top = r.results[0]
    assert top.num_input_poses == top.num_cluster_poses == 13
    assert top.clustered_pose.position.x == pytest.approx(1.006, abs=.001)
    assert top.clustered_pose.position.y == pytest.approx(2.)
    assert top.clustered_pose.position.z == pytest.approx(3.)
    feed(node, [12])
    assert snapshot(node, h).results[0].num_input_poses == 13
    assert not result.done()  # Feedback never terminates the collection.
    feed(node, [13], world_x=10.)  # A rejected outlier cannot supply a new view.
    second = snapshot(node, h)
    assert second.results[0].num_input_poses == 14
    assert second.results[0].num_cluster_poses == 13
    assert second.header.stamp == r.header.stamp


def test_accumulator_and_deduplication_are_bounded_per_camera(node):
    node._accumulation_capacity = 5
    start(node, goal())
    feed(node, range(12))
    assert node._num_collected() == len(node._goal.seen[0]) == 5
