from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import Command

# Launch File for fixed obstacles tfs e.g. from dock to placards

def generate_launch_description():
    obstacles_urdf = os.path.join(
        get_package_share_directory("bb_filters"), "urdf", "obstacles.urdf.xacro"
    )
    return LaunchDescription(
        [
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                # namespace=namespace,
                parameters=[
                    {
                        "robot_description": Command(["xacro ", obstacles_urdf]),
                    }
                ],
            )
        ]
    )
