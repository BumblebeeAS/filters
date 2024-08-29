from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import os
from ament_index_python.packages import get_package_share_directory

# Launch File for running the lidar segmentation pipeline on the BBASV4 Simulation


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="bb_filters",
                executable="detected_object_3d_array_vis.py",
                name="raw_dets_vis",
                parameters=[
                    {
                        "input_detections_topics": [
                            "/asv4/vision/lidar_small_objects/dets_3d",
                        ],
                        "output_markers_topic": "/asv4/vision/lidar_small_objects/dets_3d/marker",
                        "objects_config": "robotx.yaml",
                    }
                ],
            ),
            Node(
                package="bb_filters",
                executable="detected_object_3d_array_vis.py",
                name="large_raw_dets_vis",
                parameters=[
                    {
                        "input_detections_topics": [
                            "/asv4/vision/lidar_large_objects/dets_3d",
                        ],
                        "output_markers_topic": "/asv4/vision/lidar_large_objects/dets_3d/marker",
                        "objects_config": "robotx.yaml",
                    }
                ],
            ),
            # Node(
            #     package="bb_filters",
            #     executable="detected_object_3d_array_vis.py",
            #     name="raw_dets_vis",
            #     parameters=[{
            #         "input_detections_topics": [
            #             "/asv4/vision/lidar_small_objects/dets_3d/filtered",
            #         ],
            #         "output_markers_topic": "/asv4/vision/lidar_small_objects/dets_3d/filtered/marker",
            #         "objects_config": "robotx.yaml"
            #     }]
            # ),
            Node(
                package="bb_filters",
                executable="detected_object_2d_array_vis.py",
                name="ml_dets_vis",
                parameters=[
                    {
                        "input_detections_topics": [
                            "/asv4/vision/detections_2d",
                        ],
                        "camera_info_topics": [
                            "/asv4/left_cam/camera_info",
                            "/asv4/right_cam/camera_info",
                            "/asv4/zed2i/zed_node/left/camera_info",
                        ],
                        "output_markers_topic": "/asv4/vision/detections_2d/marker",
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
                executable="detected_object_3d_array_vis.py",
                name="bev_labelled_dets_vis",
                parameters=[
                    {
                        "input_detections_topics": [
                            "/asv4/bev_detections/filtered",
                        ],
                        "output_markers_topic": "/asv4/bev_detections/filtered/marker",
                        "objects_config": "robotx.yaml",
                        "publish_tf": True,
                        "publish_tf_unique": True,
                    }
                ],
            ),
            Node(
                package="bb_filters",
                executable="detected_object_3d_labelling.py",
                name="det_3d_labeller",
                parameters=[
                    {
                        "detection_2d_topic": "/asv4/vision/detections_2d",
                        "detection_3d_topic": "/asv4/vision/lidar_small_objects/dets_3d/filtered",
                        "camera_info_topics": [
                            "/asv4/left_cam/camera_info",
                            "/asv4/right_cam/camera_info",
                            "/asv4/zed2i/zed_node/left/camera_info",
                        ],
                        "output_labeled_topic": "/asv4/vision/lidar_small_objects/dets_3d/labelled",
                        "objects_config": "robotx.yaml",
                    }
                ],
            ),
            Node(
                package="bb_filters",
                executable="detected_object_3d_labelling.py",
                name="large_det_3d_labeller",
                parameters=[
                    {
                        "detection_2d_topic": "/asv4/vision/detections_2d",
                        "detection_3d_topic": "/asv4/vision/lidar_large_objects/dets_3d/filtered",
                        "camera_info_topics": [
                            "/asv4/left_cam/camera_info",
                            "/asv4/right_cam/camera_info",
                            "/asv4/zed2i/zed_node/left/camera_info",
                        ],
                        "output_labeled_topic": "/asv4/vision/lidar_large_objects/dets_3d/labelled",
                        "objects_config": "robotx.yaml",
                    }
                ],
            ),
            Node(
                package="bb_filters",
                executable="detected_object_3d_array_vis.py",
                name="filtered_dets_vis",
                parameters=[
                    {
                        "input_detections_topics": [
                            "/asv4/vision/lidar_small_objects/dets_3d/filtered",
                        ],
                        "output_markers_topic": "/asv4/vision/lidar_small_objects/dets_3d/filtered/marker",
                        "objects_config": "robotx.yaml",
                        "publish_tf": False,
                    }
                ],
            ),
            Node(
                package="bb_filters",
                executable="detected_object_3d_array_vis.py",
                name="labelled_dets_vis",
                parameters=[
                    {
                        "input_detections_topics": [
                            "/asv4/vision/lidar_small_objects/dets_3d/labelled",
                        ],
                        "output_markers_topic": "/asv4/vision/lidar_small_objects/dets_3d/labelled/marker",
                        "objects_config": "robotx.yaml",
                        "publish_tf": False,
                    }
                ],
            ),
            Node(
                package="bb_filters",
                executable="detected_object_3d_array_vis.py",
                name="large_filtered_dets_vis",
                parameters=[
                    {
                        "input_detections_topics": [
                            "/asv4/vision/lidar_large_objects/dets_3d/labelled",
                        ],
                        "output_markers_topic": "/asv4/vision/lidar_large_objects/dets_3d/filtered/marker",
                        "objects_config": "robotx.yaml",
                        "publish_tf": False,
                    }
                ],
            ),
            Node(
                package="bb_filters",
                executable="detected_object_3d_array_vis.py",
                name="large_labelled_dets_vis",
                parameters=[
                    {
                        "input_detections_topics": [
                            "/asv4/vision/lidar_large_objects/dets_3d/labelled",
                        ],
                        "output_markers_topic": "/asv4/vision/lidar_large_objects/dets_3d/labelled/marker",
                        "objects_config": "robotx.yaml",
                        "publish_tf": False,
                    }
                ],
            ),
            Node(
                package="bb_filters",
                executable="detected_object_3d_labelling.py",
                name="large_det_3d_labeller",
                parameters=[
                    {
                        "detection_2d_topic": "/asv4/vision/detections_2d",
                        "detection_3d_topic": "/asv4/vision/lidar_large_objects/dets_3d/filtered",
                        "camera_info_topics": [
                            "/asv4/left_cam/camera_info",
                            "/asv4/right_cam/camera_info",
                            "/asv4/zed2i/zed_node/left/camera_info",
                        ],
                        "output_labeled_topic": "/asv4/vision/lidar_large_objects/dets_3d/labelled",
                        "objects_config": "robotx.yaml",
                    }
                ],
            ),
            Node(
                package="bb_filters",
                executable="detected_object_3d_array_vis.py",
                name="large_labelled_dets_vis",
                parameters=[
                    {
                        "input_detections_topics": [
                            "/asv4/vision/lidar_large_objects/dets_3d/labelled",
                        ],
                        "output_markers_topic": "/asv4/vision/lidar_large_objects/dets_3d/labelled/marker",
                        "objects_config": "robotx.yaml",
                        "publish_tf": True,
                    }
                ],
            ),
        ]
    )
