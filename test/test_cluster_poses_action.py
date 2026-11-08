"""In-process integration test for ClusterPosesActionNode."""

from unittest.mock import Mock

import pytest
import rclpy
from bb_perception_msgs.action import ClusterPosesAction, ClusterPosesStreamAction
from bb_perception_msgs.msg import ClusterPosesRequest
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor

from cluster_test_utils import (
    EXPECTED_CLUSTER_X,
    attach_synthetic_publishers,
    identity_tf_static,
    spin_for,
    spin_until_done,
)

from bb_filters.nodes.cluster.cluster_poses_action_node import ClusterPosesActionNode

ODOM_TOPIC = "/test_odom_action"
POSE_TOPIC = "/test_pose_action"
ACTION_NAME = "/cluster_poses"


@pytest.fixture
def rig(rclpy_context):
    action_node = ClusterPosesActionNode()
    client_node = rclpy.create_node("cluster_action_test_client")

    executor = SingleThreadedExecutor()
    executor.add_node(action_node)
    executor.add_node(client_node)

    publishers = attach_synthetic_publishers(client_node, ODOM_TOPIC, [POSE_TOPIC])
    action_client = ActionClient(client_node, ClusterPosesAction, ACTION_NAME)

    stream_client = ActionClient(client_node, ClusterPosesStreamAction, '/stream_cluster_poses')
    spin_for(executor, 0.05)
    assert action_client.wait_for_server(timeout_sec=2.0), "action server discovery timed out"

    yield {
        "executor": executor,
        "server": action_node,
        "tf_pub": publishers.tf,
        "publish_timer": publishers.publish_timer,
        "action_client": action_client,
        "stream_client": stream_client,
    }

    executor.shutdown()
    client_node.destroy_node()
    action_node.destroy_node()


def test_action_succeeds(rig):
    rig["tf_pub"].publish(identity_tf_static())
    spin_for(rig["executor"], 0.2)

    goal = ClusterPosesAction.Goal()
    goal.params.odom_topic = ODOM_TOPIC
    goal.params.pose_stamped_topics = [POSE_TOPIC]
    goal.params.clustered_child_frame_ids = ["test/clustered_action"]
    goal.params.sync_tolerance = 0.05
    goal.params.min_poses = 5
    goal.params.min_cluster_size = 3
    goal.params.min_samples = 2
    goal.params.cluster_selection_epsilon = 0.0
    goal.params.top_k = 1
    goal.params.sort_key = ClusterPosesRequest.SORT_BY_NUM_CLUSTER_POSES
    goal.collection_duration = 0.6

    rig["publish_timer"].reset()
    try:
        send_future = rig["action_client"].send_goal_async(goal)
        assert spin_until_done(rig["executor"], send_future, 3.0), "send_goal timed out"
        goal_handle = send_future.result()
        assert goal_handle.accepted, "goal was rejected"

        result_future = goal_handle.get_result_async()
        assert spin_until_done(rig["executor"], result_future, 5.0), "result wait timed out"
        result = result_future.result().result
    finally:
        rig["publish_timer"].cancel()

    cluster_results = result.cluster_results
    assert cluster_results.sort_key == ClusterPosesRequest.SORT_BY_NUM_CLUSTER_POSES
    assert len(cluster_results.results) >= 1
    top = cluster_results.results[0]
    assert top.num_input_poses > 0
    assert top.num_cluster_poses > 0
    assert top.num_cluster_poses <= top.num_input_poses
    assert abs(top.clustered_pose.position.x - EXPECTED_CLUSTER_X) < 0.05


def streaming_goal(duration=1.2):
    request = ClusterPosesStreamAction.Goal()
    request.cluster_interval = .3
    request.collection_duration = duration
    p = request.params
    p.odom_topic, p.pose_stamped_topics = ODOM_TOPIC, [POSE_TOPIC]
    p.clustered_child_frame_ids = ['test/spiral_accumulation']
    p.sync_tolerance = .05
    p.min_poses = p.min_cluster_size = 5
    p.min_samples = 1
    p.cluster_selection_epsilon = .1
    return request


def assert_clean(rig):
    server = rig['server']
    assert not server._action_running
    assert server._odom_relay is None and server._num_collected() == 0
    assert server._action_timer.is_canceled()


@pytest.mark.parametrize('next_streaming', [False, True])
def test_streaming_feedback_final_result_and_next_goal_are_fresh(rig, next_streaming):
    rig['tf_pub'].publish(identity_tf_static())
    spin_for(rig['executor'], .2)
    snapshots = []
    rig['publish_timer'].reset()
    sent = rig['stream_client'].send_goal_async(
        streaming_goal(), feedback_callback=lambda msg: snapshots.append(msg.feedback))
    assert spin_until_done(rig['executor'], sent, 3.)
    result = sent.result().get_result_async()
    assert spin_until_done(rig['executor'], result, 5.)
    rig['publish_timer'].cancel()
    counts = [s.cluster_results.results[0].num_input_poses
              for s in snapshots if s.cluster_results.results]
    assert len(counts) >= 2 and counts[-1] > counts[0]
    assert result.result().result.cluster_results.results
    assert_clean(rig)
    spin_for(rig['executor'], .2)  # Drain in-flight publications before next goal.
    next_goal = streaming_goal(.5)
    next_client = rig['stream_client'] if next_streaming else rig['action_client']
    if not next_streaming:
        next_goal = ClusterPosesAction.Goal(
            params=next_goal.params, collection_duration=next_goal.collection_duration)
    sent = next_client.send_goal_async(next_goal)
    assert spin_until_done(rig['executor'], sent, 3.)
    result = sent.result().get_result_async()
    assert spin_until_done(rig['executor'], result, 3.)
    assert not result.result().result.cluster_results.results
    assert_clean(rig)


def test_cancel_returns_final_snapshot_before_first_periodic_evaluation(rig):
    from action_msgs.msg import GoalStatus
    request = streaming_goal(10.)
    request.cluster_interval = 5.
    rig['tf_pub'].publish(identity_tf_static())
    spin_for(rig['executor'], .2)
    rig['publish_timer'].reset()
    sent = rig['stream_client'].send_goal_async(request)
    assert spin_until_done(rig['executor'], sent, 3.)
    handle = sent.result()
    result = handle.get_result_async()
    spin_for(rig['executor'], .6)
    cancel = handle.cancel_goal_async()
    assert spin_until_done(rig['executor'], cancel, 3.)
    assert spin_until_done(rig['executor'], result, 3.)
    assert result.result().status == GoalStatus.STATUS_CANCELED
    assert result.result().result.cluster_results.results
    assert_clean(rig)


@pytest.mark.parametrize('failure', ['lost_acceptance_reply', 'client_disappears'])
def test_orphaned_collection_expires_even_without_client_cleanup(rig, failure):
    server = rig['server']
    server._start_goal = Mock(wraps=server._start_goal)
    sent = rig['stream_client'].send_goal_async(streaming_goal(.6))
    if failure == 'lost_acceptance_reply':
        # Server still sees the request, but the client never receives acceptance.
        rig['stream_client']._remove_pending_goal_request(sent)
    else:
        assert spin_until_done(rig['executor'], sent, 3.)
        assert sent.result().accepted
        rig['stream_client'].destroy()
    spin_for(rig['executor'], 1.5)
    server._start_goal.assert_called_once()
    assert_clean(rig)


@pytest.mark.parametrize('first_streaming', [False, True])
def test_endpoints_share_one_exclusive_goal_slot_and_release_after_cancel(rig, first_streaming):
    stream = streaming_goal(10.)
    ordinary = ClusterPosesAction.Goal(params=stream.params, collection_duration=10.)
    first_client, first_goal = ((rig['stream_client'], stream) if first_streaming
                                else (rig['action_client'], ordinary))
    next_client, next_goal = ((rig['action_client'], ordinary) if first_streaming
                              else (rig['stream_client'], stream))
    first = first_client.send_goal_async(first_goal)
    assert spin_until_done(rig['executor'], first, 3.)
    assert first.result().accepted
    # An accepted action owns the collector until its terminal result.
    blocked = next_client.send_goal_async(next_goal)
    assert spin_until_done(rig['executor'], blocked, 3.)
    assert not blocked.result().accepted
    terminal = first.result().get_result_async()
    cancel = first.result().cancel_goal_async()
    assert spin_until_done(rig['executor'], cancel, 3.)
    assert spin_until_done(rig['executor'], terminal, 3.)
    assert_clean(rig)
    next_goal.collection_duration = .2
    following = next_client.send_goal_async(next_goal)
    assert spin_until_done(rig['executor'], following, 3.)
    assert following.result().accepted
    result = following.result().get_result_async()
    assert spin_until_done(rig['executor'], result, 3.)
    assert_clean(rig)


@pytest.mark.parametrize('finish', ['deadline', 'cancel'])
def test_streaming_cleanup_runs_when_ros_time_is_paused(rig, finish):
    from action_msgs.msg import GoalStatus
    from rclpy.parameter import Parameter
    server = rig['server']
    server.set_parameters([Parameter('use_sim_time', value=True)])
    assert server.get_clock().ros_time_is_active
    assert server.get_clock().now().nanoseconds == 0  # No /clock publisher.
    snapshots = []
    sent = rig['stream_client'].send_goal_async(
        streaming_goal(.4 if finish == 'deadline' else 10.),
        feedback_callback=lambda msg: snapshots.append(msg.feedback))
    assert spin_until_done(rig['executor'], sent, 3.)
    assert sent.result().accepted
    result = sent.result().get_result_async()
    if finish == 'cancel':
        cancel = sent.result().cancel_goal_async()
        assert spin_until_done(rig['executor'], cancel, 3.)
    assert spin_until_done(rig['executor'], result, 2.)
    assert result.result().status == (GoalStatus.STATUS_ABORTED if finish == 'deadline'
                                     else GoalStatus.STATUS_CANCELED)
    assert server.get_clock().now().nanoseconds == 0
    if finish == 'deadline':
        assert any(0. < f.collection_progress < 1. for f in snapshots)
    assert_clean(rig)
