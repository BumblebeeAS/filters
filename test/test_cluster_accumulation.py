"""Periodic feedback shares one goal collection; completion always clears it."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from bb_perception_msgs.action import ClusterPosesAction
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.task import Future
from rclpy.time import Time

from bb_filters.nodes.cluster.cluster_poses_action_node import ClusterPosesActionNode
from cluster_test_utils import identity_tf_static


def goal(interval=2.):
    g = ClusterPosesAction.Goal()
    g.cluster_interval = interval
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
    handle = SimpleNamespace(request=request, publish_feedback=Mock(),
                             succeed=Mock(), abort=Mock(), canceled=Mock(),
                             is_cancel_requested=False)
    node._action_running = True
    node._goal_handle = handle
    future = Future()
    node._result_future = future
    node._start_goal(handle)
    return handle, future


def feed(node, indices, world_x=1.):
    for i in indices:
        odom = Odometry()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base'
        odom.header.stamp = Time(seconds=100 + i*.02).to_msg()
        odom.pose.pose.orientation.w = 1.
        odom.pose.pose.position.x = 50. + i*2.
        pose = PoseStamped()
        pose.header.frame_id = 'camera'
        pose.header.stamp = odom.header.stamp
        pose.pose.orientation.w = 1.
        # Observe the same world pad from widely separated vehicle positions.
        pose.pose.position.x = world_x + i*.001 - odom.pose.pose.position.x
        pose.pose.position.y = 2.
        pose.pose.position.z = 3.
        node._remember_odometry(odom)
        node._pose_subscribers[0].signalMessage(pose)


@pytest.fixture
def node(rclpy_context):
    n = ClusterPosesActionNode()
    n._tf_buffer.set_transform_static(identity_tf_static().transforms[0], 'test')
    yield n
    n.destroy_node()


def snapshot(node, handle):
    node._next_cluster = 0.
    node._step_collection(handle)
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


@pytest.mark.parametrize('interval', [0., 2.])
def test_every_new_goal_starts_fresh(node, interval):
    h, first = start(node, goal())
    feed(node, range(13))
    node._finalize_goal(h)
    assert first.result().cluster_results.results
    assert node._num_collected() == 0 and node._odom_relay is None
    h, result = start(node, goal(interval))
    node._finalize_goal(h)
    assert not result.result().cluster_results.results


def test_cancellation_returns_final_partial_interval_and_cleans_up(node):
    h, result = start(node, goal())
    feed(node, range(13))
    h.is_cancel_requested = True
    node._step_action()
    h.canceled.assert_called_once()
    assert result.result().cluster_results.results[0].num_cluster_poses == 13
    assert node._num_collected() == 0 and node._odom_relay is None


def test_deadline_expires_without_client_or_ros_clock_progress(node, monkeypatch):
    from bb_filters.nodes.cluster import cluster_poses_action_node as module
    h, result = start(node, goal())
    feed(node, range(4))
    monkeypatch.setattr(module.time, 'monotonic', lambda: node._deadline + 1.)
    node._step_action()
    node._step_action()
    assert result.done() and node._odom_relay is None
    assert node._num_collected() == 0 and not node._action_running


def test_accumulator_and_deduplication_are_bounded_per_camera(node):
    node._accumulation_capacity = 5
    start(node, goal())
    feed(node, range(12))
    assert node._num_collected() == len(node._seen[0]) == 5


def test_outlier_cannot_supply_a_new_supporting_viewpoint(node):
    h, _ = start(node, goal())
    feed(node, range(13))
    first = snapshot(node, h)
    feed(node, [13], world_x=10.)
    second = snapshot(node, h)
    assert second.results[0].num_input_poses == 14
    assert second.results[0].num_cluster_poses == 13
    assert second.header.stamp == first.header.stamp
