"""In-process integration test for ClusterPosesActionNode."""

import pytest
import rclpy
from bb_perception_msgs.action import ClusterPosesAction
from bb_perception_msgs.msg import ClusterPosesRequest
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor

from cluster_test_utils import (
    EXPECTED_CLUSTER_X,
    add_scripts_to_path,
    attach_synthetic_publishers,
    identity_tf_static,
    spin_for,
    spin_until_done,
)

add_scripts_to_path()
from cluster_poses_action_node import ClusterPosesActionNode  # noqa: E402

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

    spin_for(executor, 0.05)
    assert action_client.server_is_ready(), "action server should be ready intraprocess"

    yield {
        "executor": executor,
        "tf_pub": publishers.tf,
        "publish_timer": publishers.publish_timer,
        "action_client": action_client,
    }

    executor.shutdown()
    action_client.destroy()
    client_node.destroy_node()
    action_node.destroy_node()


def test_action_succeeds(rig):
    rig["tf_pub"].publish(identity_tf_static())
    spin_for(rig["executor"], 0.2)

    goal = ClusterPosesAction.Goal()
    goal.params.odom_topic = ODOM_TOPIC
    goal.params.pose_stamped_topics = [POSE_TOPIC]
    goal.params.clustered_child_frame_id = "test/clustered_action"
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
