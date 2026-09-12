from launch import LaunchDescription
from launch_ros.actions import Node, PushRosNamespace


def generate_launch_description():
    return LaunchDescription(
        [
            PushRosNamespace("uav"),
            Node(
                package="bb_filters",
                executable="cluster_poses_action_node.py",
                name="cluster_poses_action_node",
                output="screen",
            ),
        ]
    )
