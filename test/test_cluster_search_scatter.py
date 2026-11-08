"""Regression from a sparse nine-pose burst; search must tolerate its 6 cm spread."""
from geometry_msgs.msg import PoseStamped
from bb_filters.nodes.cluster.cluster_poses_action_node import ClusterPosesActionNode
from bb_filters.nodes.cluster.cluster_poses_node import ClusterParams


def test_search_selection_accepts_recorded_scatter_without_reducing_support(rclpy_context):
    # Recorded world poses from 2026-11-08; approximate action input because the
    # monitor used timestamped TF rather than the action's matched odometry.
    points = [[-33.539672961736485, -0.705456466891605, 11.417506106204462], [-33.56512922685429, -0.6917185019060605, 11.474487220593296], [-33.51699682683786, -0.6964778274892616, 11.418142626172209], [-33.51963721307496, -0.6940039411728485, 11.417563330238734], [-33.54180444288512, -0.6634370818883009, 11.472302097465617], [-33.46169675320631, -0.7080001221157983, 11.3252632957951], [-33.4905580455184, -0.6976542518412221, 11.367882415467506], [-33.50724659037174, -0.6778664051530594, 11.42318836955951], [-33.434897592143685, -0.7128871600227531, 11.300917539772957]]
    node = ClusterPosesActionNode()
    try:
        poses = []
        for x, y, z in points:
            p = PoseStamped()
            p.pose.position.x, p.pose.position.y, p.pose.position.z = x, y, z
            p.pose.orientation.w = 1.
            poses.append(p)
        strict = ClusterParams(min_cluster_size=5, min_samples=1,
                               min_poses=5, cluster_selection_epsilon=.05)
        assert not node._cluster_poses(poses, strict)
        search = ClusterParams(min_cluster_size=5, min_samples=1,
                               min_poses=5, cluster_selection_epsilon=.10)
        clusters = node._cluster_poses(poses, search)
        assert len(clusters) == 1
        _, result = clusters[0]
        assert result.num_cluster_poses == 9
        assert .05 < result.clustered_position_std < .07
    finally:
        node.destroy_node()
