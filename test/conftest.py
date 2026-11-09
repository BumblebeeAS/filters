"""Pytest fixtures shared across the filters/test/ suite."""

import pytest
import rclpy


@pytest.fixture(scope="module")
def rclpy_context():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def cluster_node(rclpy_context):
    from bb_filters.nodes.cluster.cluster_poses_action_node import ClusterPosesActionNode
    node = ClusterPosesActionNode()
    yield node
    node.destroy_node()
