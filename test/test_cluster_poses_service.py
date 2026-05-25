"""In-process integration test for ClusterPosesServiceNode."""

import pytest
import rclpy
from bb_perception_msgs.msg import ClusterPoseResult
from bb_perception_msgs.srv import ClusterPosesSrv
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
from cluster_poses_service_node import ClusterPosesServiceNode  # noqa: E402

ODOM_TOPIC = "/test_odom"
POSE_TOPIC = "/test_pose"
RESULT_TOPIC = "/cluster_pose_result"
SERVICE_NAME = "/cluster_poses_srv"


@pytest.fixture
def rig(rclpy_context):
    service_node = ClusterPosesServiceNode()
    client_node = rclpy.create_node("cluster_service_test_client")

    executor = SingleThreadedExecutor()
    executor.add_node(service_node)
    executor.add_node(client_node)

    publishers = attach_synthetic_publishers(client_node, ODOM_TOPIC, POSE_TOPIC)
    received: list[ClusterPoseResult] = []
    client_node.create_subscription(
        ClusterPoseResult, RESULT_TOPIC, received.append, 10
    )
    client = client_node.create_client(ClusterPosesSrv, SERVICE_NAME)

    spin_for(executor, 0.05)
    assert client.service_is_ready(), "service should be ready intraprocess"

    def call_service(req, timeout_sec=3.0):
        future = client.call_async(req)
        assert spin_until_done(executor, future, timeout_sec), "service call timed out"
        return future.result()

    yield {
        "executor": executor,
        "tf_pub": publishers.tf,
        "publish_timer": publishers.publish_timer,
        "received": received,
        "call_service": call_service,
    }

    executor.shutdown()
    client_node.destroy_node()
    service_node.destroy_node()


def test_disable_without_enable_is_noop(rig):
    req = ClusterPosesSrv.Request()
    req.enabled = False
    response = rig["call_service"](req)
    assert response.is_enabled is False
    assert response.is_cluster_success is False


def test_enable_periodic_then_disable(rig):
    rig["tf_pub"].publish(identity_tf_static())
    spin_for(rig["executor"], 0.2)

    req = ClusterPosesSrv.Request()
    req.enabled = True
    req.odom_topic = ODOM_TOPIC
    req.pose_stamped_topic = POSE_TOPIC
    req.clustered_child_frame_id = "test/clustered"
    req.sync_queue_size = 100
    req.sync_tolerance = 0.05
    req.min_poses = 5
    req.min_cluster_size = 3
    req.min_samples = 2
    req.cluster_selection_epsilon = 0.0
    req.cluster_interval = 0.3

    enable_response = rig["call_service"](req)
    assert enable_response.is_enabled is True
    assert enable_response.is_cluster_success is False

    rig["publish_timer"].reset()
    spin_for(rig["executor"], 1.5)
    rig["publish_timer"].cancel()

    periodic_before_disable = len(rig["received"])
    assert periodic_before_disable >= 1, (
        "periodic ClusterPoseResult was never published"
    )

    disable_req = ClusterPosesSrv.Request()
    disable_req.enabled = False
    disable_response = rig["call_service"](disable_req)

    assert disable_response.is_enabled is False
    assert disable_response.is_cluster_success is True, "final cluster did not succeed"
    final = disable_response.cluster_result
    assert final.num_input_poses > 0
    assert final.num_cluster_poses > 0
    assert final.num_cluster_poses <= final.num_input_poses
    assert abs(final.clustered_pose.pose.position.x - EXPECTED_CLUSTER_X) < 0.05

    spin_for(rig["executor"], 0.2)
    assert len(rig["received"]) > periodic_before_disable, (
        "final ClusterPoseResult was not republished on the topic"
    )
