from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

# Launch File for running the lidar segmentation pipeline on the BBASV4 Simulation
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="bb_filters",
                executable="detected_object_3d_array_vis.py",
                name="raw_gt_dets_vis",
                parameters=[
                    {
                        "input_detections_topics": [
                            "/robotx/detections",
                        ],
                        "output_markers_topic": "/robotx/detections/marker",
                        "objects_config": "robotx.yaml",
                    }
                ],
            ),
            Node(
                package="bb_filters",
                executable="detected_object_3d_array_vis.py",
                name="filtered_gt_dets_vis",
                parameters=[
                    {
                        "input_detections_topics": [
                            "/robotx/detections/filtered",
                        ],
                        "output_markers_topic": "/robotx/detections/filtered/marker",
                        "objects_config": "robotx.yaml",
                    }
                ],
            ),
            Node(
                package="bb_filters",
                executable="detected_object_3d_filter.py",
                # executable="detected_object_3d_composite_filter.py",
                name="det_3d_sort_filter",
                parameters=[
                    {
                        "dets_3d_topic": "/asv4/vision/lidar_small_objects/dets_3d",
                        "filtered_topic": "/asv4/vision/lidar_small_objects/dets_3d/filtered",
                        "objects_config": "robotx.yaml",
                        "max_lost": 100,
                        "dist_threshold": 4.0,
                    }
                ],
            ),
            Node(
                package="bb_filters",
                executable="detected_object_3d_filter.py",
                # executable="detected_object_3d_composite_filter.py",
                name="large_det_3d_sort_filter",
                parameters=[
                    {
                        "dets_3d_topic": "/asv4/vision/lidar_large_objects/dets_3d",
                        "filtered_topic": "/asv4/vision/lidar_large_objects/dets_3d/filtered",
                        "objects_config": "robotx.yaml",
                        "max_lost": 150,
                        "dist_threshold": 10.0,
                    }
                ],
            ),
            Node(
                package="bb_filters",
                executable="detected_object_3d_filter.py",
                # executable="detected_object_3d_composite_filter.py",
                name="bev_det_3d_sort_filter",
                parameters=[
                    {
                        "dets_3d_topic": "/asv4/bev_detections",
                        "filtered_topic": "/asv4/bev_detections/filtered",
                        "objects_config": "robotx.yaml",
                    }
                ],
            ),
            Node(
                package="bb_filters",
                executable="robotx_sim_obstacles_converter.py",
                name="obstacles_converter",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [
                        PathJoinSubstitution(
                            [
                                FindPackageShare("ml_detector"),
                                "launch",
                                "label_publisher.launch.py",
                            ]
                        )
                    ]
                ),
                launch_arguments={"competition_name": "robotx"}.items(),
            ),
        ]
    )
