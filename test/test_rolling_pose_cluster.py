"""Rolling fusion must merge camera extrinsics and expire stale observations."""

from unittest.mock import Mock

import numpy as np
import pytest
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.time import Time

from bb_filters.nodes.cluster.rolling_pose_cluster_node import RollingPoseClusterNode


def pair(stamp, camera, x):
    pose = PoseStamped()
    pose.header.frame_id = camera
    pose.header.stamp = Time(nanoseconds=int(stamp * 1e9)).to_msg()
    pose.pose.position.x = float(x)
    pose.pose.orientation.w = 1.
    odom = Odometry()
    odom.header.stamp = pose.header.stamp
    odom.header.frame_id = 'odom_ned'
    odom.child_frame_id = 'body'
    odom.pose.pose.position.x = 10.
    odom.pose.pose.orientation.w = 1.
    return odom, pose


def rig():
    node = RollingPoseClusterNode()
    node.get_clock = Mock(return_value=Mock(now=Mock(return_value=Time(seconds=100))))
    node.pose_pub = Mock()
    node.result_pub = Mock()
    node.tf_pub = Mock()
    for camera, x in [('wide', 1.), ('narrow', -1.)]:
        tf = TransformStamped()
        tf.header.frame_id = 'body'
        tf.child_frame_id = camera
        tf.transform.translation.x = x
        tf.transform.rotation.w = 1.
        node._tf_buffer.set_transform_static(tf, 'test')
    return node


@pytest.mark.parametrize('cameras', [[('wide', 2.)], [('wide', 2.), ('narrow', 4.)]])
def test_camera_streams_merge_into_one_cluster_and_expire(rclpy_context, cameras):
    node = rig()
    try:
        rng = np.random.default_rng(3)
        for index, (camera, x) in enumerate(cameras):
            for i in range(20):
                odom, pose = pair(99.5 + i*.01, camera, x + rng.normal(0, .002))
                node._collect_pose(odom, pose, index)
        node._tick()
        node.pose_pub.publish.assert_called_once()
        pose = node.pose_pub.publish.call_args.args[0]
        assert pose.header.frame_id == 'odom_ned'
        assert abs(pose.pose.position.x - 13.) < .01
        assert node.result_pub.publish.call_args.args[0].results[0].num_input_poses == 20 * len(cameras)
        assert node.tf_pub.sendTransform.call_args.args[0].child_frame_id == 'asv/clustered'
        # Repeated timers must not freshen or republish an old observation.
        node._tick()
        node.pose_pub.publish.assert_called_once()
        node.get_clock.return_value.now.return_value = Time(seconds=110)
        node._tick()
        assert not any(node._stream_data)
        node.pose_pub.publish.assert_called_once()
    finally:
        node.destroy_node()


def test_buffer_is_bounded_deduplicated_and_rejects_bad_times_or_frames(rclpy_context):
    node = rig()
    try:
        for i in range(1100):
            odom, pose = pair(99. + i*.0005, 'wide', 2.)
            node._collect_pose(odom, pose, 0)
        assert len(node._stream_data[0]) == 1000
        node._collect_pose(odom, pose, 0)
        assert len(node._stream_data[0]) == 1000
        for stamp in (0., 90., 101.):
            node._collect_pose(*pair(stamp, 'narrow', 4.), 1)
        odom, pose = pair(99., 'narrow', 4.)
        odom.header.frame_id = 'home_ned'
        node._collect_pose(odom, pose, 1)
        assert not node._stream_data[1]
    finally:
        node.destroy_node()
