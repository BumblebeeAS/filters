"""In-process integration test for ClusterPosesServiceNode."""

import pytest
import rclpy
from bb_perception_msgs.msg import ClusterPoseResultArray, ClusterPosesRequest
from bb_perception_msgs.srv import ClusterPosesSrv
from cluster_test_utils import (
    EXPECTED_CLUSTER_X,
    add_scripts_to_path,
    attach_synthetic_publishers,
    identity_tf_static,
    spin_for,
    spin_until_done,
)
from rclpy.executors import SingleThreadedExecutor

add_scripts_to_path()
from cluster_poses_service_node import ClusterPosesServiceNode  # noqa: E402

ODOM_TOPIC = "/test_odom"
POSE_TOPIC = "/test_pose"
POSE_TOPIC_A = "/test_pose_a"
POSE_TOPIC_B = "/test_pose_b"
SECONDARY_CLUSTER_X = 3.0  # second cluster center for the multi-topic test
RESULT_TOPIC = "/cluster_pose_results"
SERVICE_NAME = "/cluster_poses_srv"


def _build_rig(pose_topics, cluster_xs=None):
    """Spin up a service node + client + synthetic publishers.

    Returns (rig_dict, teardown_callable). Caller is responsible for invoking
    teardown_callable, typically from a fixture's finalize block.
    """
    service_node = ClusterPosesServiceNode()
    client_node = rclpy.create_node("cluster_service_test_client")

    executor = SingleThreadedExecutor()
    executor.add_node(service_node)
    executor.add_node(client_node)

    publishers = attach_synthetic_publishers(
        client_node, ODOM_TOPIC, pose_topics, cluster_xs
    )
    received: list[ClusterPoseResultArray] = []
    client_node.create_subscription(
        ClusterPoseResultArray, RESULT_TOPIC, received.append, 10
    )
    client = client_node.create_client(ClusterPosesSrv, SERVICE_NAME)

    spin_for(executor, 0.05)
    assert client.service_is_ready(), "service should be ready intraprocess"

    def call_service(req, timeout_sec=3.0):
        future = client.call_async(req)
        assert spin_until_done(executor, future, timeout_sec), "service call timed out"
        return future.result()

    rig = {
        "executor": executor,
        "tf_pub": publishers.tf,
        "publish_timer": publishers.publish_timer,
        "received": received,
        "call_service": call_service,
    }

    def teardown():
        executor.shutdown()
        client_node.destroy_node()
        service_node.destroy_node()

    return rig, teardown


@pytest.fixture
def rig(rclpy_context):
    rig, teardown = _build_rig([POSE_TOPIC])
    yield rig
    teardown()


@pytest.fixture
def multi_rig(rclpy_context):
    rig, teardown = _build_rig(
        [POSE_TOPIC_A, POSE_TOPIC_B],
        cluster_xs=[EXPECTED_CLUSTER_X, SECONDARY_CLUSTER_X],
    )
    yield rig
    teardown()


def _base_enable_request(pose_topics, child_frame_id):
    req = ClusterPosesSrv.Request()
    req.enabled = True
    req.params.odom_topic = ODOM_TOPIC
    req.params.pose_stamped_topics = pose_topics
    req.params.clustered_child_frame_id = child_frame_id
    req.params.sync_queue_size = 100
    req.params.sync_tolerance = 0.05
    req.params.min_poses = 5
    req.params.min_cluster_size = 3
    req.params.min_samples = 2
    req.params.cluster_selection_epsilon = 0.0
    req.params.top_k = 1
    req.params.sort_key = ClusterPosesRequest.SORT_BY_NUM_CLUSTER_POSES
    return req


def test_disable_without_enable_is_noop(rig):
    req = ClusterPosesSrv.Request()
    req.enabled = False
    response = rig["call_service"](req)
    assert response.is_enabled is False
    assert response.is_cluster_success is False


def test_enable_periodic_then_disable(rig):
    rig["tf_pub"].publish(identity_tf_static())
    spin_for(rig["executor"], 0.2)

    req = _base_enable_request([POSE_TOPIC], "test/clustered")
    req.cluster_interval = 0.3

    enable_response = rig["call_service"](req)
    assert enable_response.is_enabled is True
    assert enable_response.is_cluster_success is False

    rig["publish_timer"].reset()
    spin_for(rig["executor"], 1.5)
    rig["publish_timer"].cancel()

    periodic_before_disable = len(rig["received"])
    assert periodic_before_disable >= 1, (
        "periodic ClusterPoseResultArray was never published"
    )

    disable_req = ClusterPosesSrv.Request()
    disable_req.enabled = False
    disable_response = rig["call_service"](disable_req)

    assert disable_response.is_enabled is False
    assert disable_response.is_cluster_success is True, "final cluster did not succeed"
    final = disable_response.cluster_results
    assert final.sort_key == ClusterPosesRequest.SORT_BY_NUM_CLUSTER_POSES
    assert len(final.results) >= 1
    top = final.results[0]
    assert top.num_input_poses > 0
    assert top.num_cluster_poses > 0
    assert top.num_cluster_poses <= top.num_input_poses
    assert abs(top.clustered_pose.position.x - EXPECTED_CLUSTER_X) < 0.05

    spin_for(rig["executor"], 0.2)
    assert len(rig["received"]) > periodic_before_disable, (
        "final ClusterPoseResultArray was not republished on the topic"
    )


def test_disable_with_no_poses_returns_empty_array(rig):
    """Enable without ever publishing poses; disable should return an empty
    results array, sort_key preserved, and is_cluster_success=False."""
    rig["tf_pub"].publish(identity_tf_static())
    spin_for(rig["executor"], 0.2)

    req = _base_enable_request([POSE_TOPIC], "test/clustered_empty")
    # Force "not enough poses" by demanding more than we'll ever publish.
    req.params.min_poses = 100
    req.params.sort_key = ClusterPosesRequest.SORT_BY_MEAN_PROBABILITY

    enable_response = rig["call_service"](req)
    assert enable_response.is_enabled is True

    # Briefly spin without starting the publish timer. No (odom, pose) pairs
    # accumulate.
    spin_for(rig["executor"], 0.2)

    disable_req = ClusterPosesSrv.Request()
    disable_req.enabled = False
    disable_response = rig["call_service"](disable_req)

    assert disable_response.is_enabled is False
    assert disable_response.is_cluster_success is False
    final = disable_response.cluster_results
    assert final.results == [], "expected empty results when no poses collected"
    assert final.sort_key == ClusterPosesRequest.SORT_BY_MEAN_PROBABILITY


def test_multiple_pose_topics_feed_same_buffer(multi_rig):
    """Two pose topics publish around different X values into the same
    cluster buffer; with top_k=2 we should see one cluster per topic."""
    multi_rig["tf_pub"].publish(identity_tf_static())
    spin_for(multi_rig["executor"], 0.2)

    req = _base_enable_request([POSE_TOPIC_A, POSE_TOPIC_B], "test/clustered_multi")
    req.params.top_k = 2

    enable_response = multi_rig["call_service"](req)
    assert enable_response.is_enabled is True

    multi_rig["publish_timer"].reset()
    spin_for(multi_rig["executor"], 1.5)
    multi_rig["publish_timer"].cancel()

    disable_req = ClusterPosesSrv.Request()
    disable_req.enabled = False
    response = multi_rig["call_service"](disable_req)

    assert response.is_cluster_success is True
    final = response.cluster_results
    assert len(final.results) == 2, (
        f"expected two clusters (one per pose topic), got {len(final.results)}"
    )

    cluster_xs = sorted(r.clustered_pose.position.x for r in final.results)
    assert abs(cluster_xs[0] - EXPECTED_CLUSTER_X) < 0.05
    assert abs(cluster_xs[1] - SECONDARY_CLUSTER_X) < 0.05

    # Each cluster should hold a number of cluster poses both
    # above the minimum and below the total input count
    total_input = final.results[0].num_input_poses
    assert total_input == final.results[1].num_input_poses
    for cluster in final.results:
        assert cluster.num_cluster_poses >= 3
        assert cluster.num_cluster_poses < total_input
