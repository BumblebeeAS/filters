"""Search evaluations share observations; new searches and ordinary goals do not."""
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


def goal(session='search-one'):
    g = ClusterPosesAction.Goal()
    g.accumulation_id = session
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


def test_four_then_nine_poses_cluster_together_using_original_odometry(node):
    h, first = start(node, goal())
    feed(node, range(4))
    node._finalize_goal(h)
    assert first.done() and not first.result().cluster_results.results
    assert node._num_collected() == 4
    original_relay = node._odom_relay
    # Collection continues even between evaluations, with no subscription gap.
    feed(node, [4])
    assert node._num_collected() == 5
    h, second = start(node, goal())
    assert node._odom_relay is original_relay
    feed(node, range(5, 13))
    node._finalize_goal(h)
    results = second.result().cluster_results.results
    assert len(results) == 1
    assert results[0].num_input_poses == results[0].num_cluster_poses == 13
    assert results[0].clustered_pose.position.x == pytest.approx(1.006, abs=.001)
    assert results[0].clustered_pose.position.y == pytest.approx(2.)
    assert results[0].clustered_pose.position.z == pytest.approx(3.)
    # Re-evaluation doesn't re-add old poses, and duplicates aren't new support.
    h, third = start(node, goal())
    feed(node, [12])
    assert node._num_collected() == 13
    node._finalize_goal(h)
    assert third.result().cluster_results.results[0].num_input_poses == 13


@pytest.mark.parametrize('next_goal', ['new_search', 'ordinary', 'changed_params'])
def test_next_search_or_ordinary_goal_clears_accumulator(node, next_goal):
    h, _ = start(node, goal())
    feed(node, range(4))
    node._finalize_goal(h)
    request = goal('search-two' if next_goal == 'new_search' else
                   '' if next_goal == 'ordinary' else 'search-one')
    if next_goal == 'changed_params':request.params.sync_tolerance = .1
    h, result = start(node, request)
    assert node._num_collected() == 0
    node._finalize_goal(h)
    assert not result.result().cluster_results.results
    if next_goal == 'ordinary':assert node._odom_relay is None


def test_cancellation_closes_session(node):
    h, result = start(node, goal())
    feed(node, range(4))
    h.is_cancel_requested = True
    node._step_action()
    h.canceled.assert_called_once()
    assert result.done()
    assert node._session_id == ''
    assert node._num_collected() == 0
    assert node._odom_relay is None


def test_idle_expiry_closes_abandoned_session(node, monkeypatch):
    from bb_filters.nodes.cluster import cluster_poses_action_node as module
    h, _ = start(node, goal())
    feed(node, range(4))
    node._finalize_goal(h)
    expired = node._session_idle_since + 11.
    monkeypatch.setattr(module.time, 'monotonic', lambda: expired)
    node._expire_idle_session()
    assert node._session_id == '' and node._odom_relay is None
    assert node._num_collected() == 0


def test_accumulator_is_bounded_per_camera(node):
    node._accumulation_capacity = 5
    start(node, goal())
    feed(node, range(12))
    assert node._num_collected() == len(node._session_seen[0]) == 5


def test_new_outlier_does_not_count_as_a_new_supporting_viewpoint(node):
    h, first = start(node, goal())
    feed(node, range(13))
    node._finalize_goal(h)
    header = first.result().cluster_results.header
    h, second = start(node, goal())
    feed(node, [13], world_x=10.)  # A new view produces only a rejected outlier.
    node._finalize_goal(h)
    result = second.result().cluster_results
    assert result.results[0].num_input_poses == 14
    assert result.results[0].num_cluster_poses == 13
    assert result.header.stamp == header.stamp
