from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import os
from ament_index_python.packages import get_package_share_directory

# Launch File for running the lidar segmentation pipeline on the BBASV4 Simulation

detected_object_3d_vis = [
    "/asv4/vision/lidar_small_objects/dets_3d",
    "/asv4/vision/lidar_small_objects/dets_3d/filtered",
    "/asv4/vision/lidar_small_objects/dets_3d/labelled",
    # "/asv4/vision/lidar_large_objects/dets_3d",
    # "/asv4/vision/lidar_large_objects/dets_3d/filtered",
    # "/asv4/vision/lidar_large_objects/dets_3d/labelled",
    "/asv4/bev_detections/filtered",
    "/asv4/tasks/scan_dock_deliver/placard/detections_3d",
    "/asv4/vision/gate_detections",
    "/asv4/vision/detections_2d/projected",
    "/asv4/vision/red_green_gate_detections"
]
detected_object_3d_vis_tf = [
    "/asv4/robotx/filtered_detections",
]
detected_object_2d_vis = [
    "/asv4/vision/detections_2d",
    "/asv4/vision/detections_2d/fixed"
]


def generate_launch_description():
    return LaunchDescription(
        [
            # ComposableNodeContainer(),

            *[
                Node(
                    package="bb_filters",
                    executable="detected_object_3d_vis",
                    name=f"{topic.replace('/', '_')}_vis",
                    parameters=[
                        {
                            "input_detections_topics": [topic],
                            "output_markers_topic": f"{topic}/marker",
                            "objects_config": "robotx.yaml",
                        }
                    ],
                ) for topic in detected_object_3d_vis
            ],
            *[
                Node(
                    package="bb_filters",
                    executable="detected_object_3d_vis",
                    name=f"{topic.replace('/', '_')}_vis",
                    parameters=[
                        {
                            "input_detections_topics": [topic],
                            "output_markers_topic": f"{topic}/marker",
                            "objects_config": "robotx.yaml",
                            "publish_tf": True,
                            "use_sim_time": True,
                        }
                    ],
                ) for topic in detected_object_3d_vis_tf
            ],
            *[
                Node(
                    package="bb_filters",
                    executable="detected_object_2d_vis",
                    name=f"{topic.replace('/', '_')}_vis",
                    parameters=[
                        {
                            "input_detections_topics": [topic],
                            "camera_info_topics": [
                                "/asv4/left_cam/camera_info",
                                "/asv4/right_cam/camera_info",
                                "/asv4/front_cam/camera_info",
                            ],
                            "output_markers_topic": f"{topic}/marker",
                            "objects_config": "robotx.yaml",
                        }
                    ],
                ) for topic in detected_object_2d_vis
            ],
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
                executable="detected_object_3d_labelling.py",
                name="det_3d_labeller",
                parameters=[
                    {
                        "detection_2d_topic": "/asv4/vision/detections_2d",
                        "detection_3d_topic": "/asv4/vision/lidar_small_objects/dets_3d/filtered",
                        "camera_info_topics": [
                            "/asv4/left_cam/camera_info",
                            "/asv4/right_cam/camera_info",
                            "/asv4/front_cam/camera_info",
                        ],
                        "output_labeled_topic": "/asv4/vision/lidar_small_objects/dets_3d/labelled",
                        "objects_config": "robotx.yaml",
                    }
                ],
            ),
            # Node(
            #     package="bb_filters",
            #     executable="detected_object_3d_labelling.py",
            #     name="large_det_3d_labeller",
            #     parameters=[
            #         {
            #             "detection_2d_topic": "/asv4/vision/detections_2d",
            #             "detection_3d_topic": "/asv4/vision/lidar_large_objects/dets_3d/filtered",
            #             "camera_info_topics": [
            #                 "/asv4/left_cam/camera_info",
            #                 "/asv4/right_cam/camera_info",
            #                 "/asv4/front_cam/camera_info",
            #             ],
            #             "output_labeled_topic": "/asv4/vision/lidar_large_objects/dets_3d/labelled",
            #             "objects_config": "robotx.yaml",
            #         }
            #     ],
            # ),
            # Node(
            #     package="bb_filters",
            #     executable="detected_object_3d_filter.py",
            #     # executable="detected_object_3d_composite_filter.py",
            #     name="det_2d_proj_filter",
            #     parameters=[
            #         {
            #             "dets_3d_topic": "/asv4/vision/detections_2d/projected",
            #             "filtered_topic": "/asv4/vision/detections_2d/projected/filtered",
            #             "objects_config": "robotx.yaml",
            #             "max_lost": 10,
            #             "dist_threshold": 5.0,
            #         }
            #     ],
            # ),
            Node(
                package="bb_filters",
                executable="detected_object_2d_filter_projection.py",
                name="dets_2d_projection",
                parameters=[
                    "input_detections_topics": [
                        "/asv4/vision/detections_2d/fixed",
                    ],
                    "output_detections_topic": "/asv4/vision/detections_2d/projected",
                    "objects_config": "robotx.yaml",
                    "inflate_height": 0.1,
                    "ground_z": -0.2
                ],
            ),
        ]
    )
