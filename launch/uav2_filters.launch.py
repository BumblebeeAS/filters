from launch import LaunchDescription
from launch_ros.actions import Node, PushRosNamespace


def generate_launch_description():
    return LaunchDescription(
        [
            PushRosNamespace("uav2"),
            Node(
                package="bb_filters",
                executable="cluster_poses_node.py",
                name="cluster_poses_node",
                output="screen",
            ),
        ]
    )
