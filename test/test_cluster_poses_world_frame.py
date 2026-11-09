"""Merged camera positions keep their world pose, orientation and newest timestamp."""
import numpy as np
from rclpy.time import Time
from tf_transformations import quaternion_from_euler

from bb_filters.nodes.cluster.cluster_poses_node import ClusterParams
from cluster_test_utils import make_odom, make_pose


def test_merged_world_positions_ignore_vehicle_motion_and_use_newest_stamp(cluster_node):
    node = cluster_node
    rows, rng, q = [], np.random.default_rng(3), quaternion_from_euler(0, 0, .9)
    for i in range(40):
        # The second camera's samples are older, despite being appended last.
        stamp = Time(seconds=(100. if i < 20 else 98.) + (i % 20)*.01).to_msg()
        odom = make_odom(stamp)
        odom.header.frame_id = 'odom_ned'
        odom.pose.pose.position.x = 10. + i  # Vehicle motion must not be applied twice.
        pose = make_pose(stamp, *(np.array([1., 2., 3.]) + rng.normal(0, .003, 3)))
        pose.header = odom.header
        pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = q
        rows.append((odom, pose))
    node._stream_data = [rows[:20], rows[20:]]
    merged, _ = node._resolve_stream_frames(['pad'])[0]
    params = ClusterParams(min_cluster_size=5, min_samples=3, cluster_selection_epsilon=.03, min_poses=5)
    clusters, poses, total = node._run_clustering(merged, params)
    assert total == 40 and clusters
    avg, _ = clusters[0]
    np.testing.assert_allclose([avg.position.x, avg.position.y, avg.position.z], [1., 2., 3.], atol=.01)
    actual = np.array([avg.orientation.x, avg.orientation.y, avg.orientation.z, avg.orientation.w])
    assert abs(np.dot(actual, q)) > 1. - 1e-12
    assert Time.from_msg(poses[-1].header.stamp).nanoseconds == Time(seconds=100.19).nanoseconds
