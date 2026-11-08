"""World-frame recheck positions must not be transformed by vehicle motion again."""
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.time import Time
from tf_transformations import quaternion_from_euler
from bb_filters.nodes.cluster.cluster_poses_action_node import ClusterPosesActionNode
from bb_filters.nodes.cluster.cluster_poses_node import ClusterParams


def test_world_positions_cluster_without_body_transform_and_retain_orientation(rclpy_context):
    node = ClusterPosesActionNode()
    try:
        rows=[];rng=np.random.default_rng(3);q=quaternion_from_euler(0,0,.9)
        for i in range(40):
            odom=Odometry();odom.header.frame_id='odom_ned';odom.child_frame_id='body'
            odom.header.stamp=Time(seconds=100+i*.01).to_msg()
            odom.pose.pose.position.x=10.+i  # Deliberately large vehicle motion.
            odom.pose.pose.orientation.w=1.
            pose=PoseStamped();pose.header=odom.header
            pose.pose.position.x,pose.pose.position.y,pose.pose.position.z=np.array([1.,2.,3.])+rng.normal(0,.003,3)
            pose.pose.orientation.x,pose.pose.orientation.y,pose.pose.orientation.z,pose.pose.orientation.w=q
            rows.append((odom,pose))
        params=ClusterParams(min_cluster_size=5,min_samples=3,cluster_selection_epsilon=.03,min_poses=5)
        clusters,poses,total=node._run_clustering(rows,params)
        assert total==40 and clusters
        avg,_=clusters[0]
        np.testing.assert_allclose([avg.position.x,avg.position.y,avg.position.z],[1.,2.,3.],atol=.01)
        actual=np.array([avg.orientation.x,avg.orientation.y,avg.orientation.z,avg.orientation.w])
        assert abs(np.dot(actual,q))>1.-1e-12
        assert not node._camera_to_odom_transforms
    finally:
        node.destroy_node()


def test_merged_camera_stream_timestamp_uses_newest_observation(rclpy_context):
    node=ClusterPosesActionNode()
    try:
        streams=[]
        for start in (100.,98.):  # The last camera stream contains older samples.
            stream=[]
            for i in range(20):
                odom=Odometry();odom.header.frame_id='odom_ned';odom.child_frame_id='body'
                odom.header.stamp=Time(seconds=start+i*.01).to_msg();odom.pose.pose.orientation.w=1.
                pose=PoseStamped();pose.header=odom.header;pose.pose.orientation.w=1.
                pose.pose.position.x=1.+i*.0001
                stream.append((odom,pose))
            streams.append(stream)
        node._stream_data=streams
        merged,_=node._resolve_stream_frames(['pad'])[0]
        params=ClusterParams(min_cluster_size=3,min_samples=2,cluster_selection_epsilon=.03,min_poses=5)
        clusters,poses,_=node._run_clustering(merged,params)
        assert clusters
        assert Time.from_msg(poses[-1].header.stamp).nanoseconds==Time(seconds=100.19).nanoseconds
    finally:
        node.destroy_node()
