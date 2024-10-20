from launch import LaunchDescription
from launch_ros.actions import Node

# Launch File for running the lidar segmentation pipeline on the BBASV4 Simulation


detected_object_3d_vis = [
    "/uav2/projected_3d"
]
detected_object_3d_vis_tf = [
    # "/asv4/robotx/filtered_detections",
]
detected_object_2d_vis = [
    "/sim/detections_2d"
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
                            "objects_config": "drone.yaml",
                        }
                    ],
                )
                for topic in detected_object_3d_vis
            ],
            *[
                Node(
                    package="bb_filters",
                    executable="detected_object_2d_vis",
                    name=f"{topic.replace('/', '_')}_vis",
                    parameters=[
                        {
                            "input_detections_topics": [topic],
                            "camera_info_topics": ["/camera_info"],
                            "output_markers_topic": f"{topic}/marker",
                            "objects_config": "drone.yaml",
                        }
                    ],
                )
                for topic in detected_object_2d_vis
            ],
            Node(
                package="bb_filters",
                executable="detected_object_2d_filter_projection_bottom_facing.py",
                name="dets_2d_projection",
                parameters=[
                    {
                        "input_detections_topics": [
                            "/sim/detections_2d",
                        ],
                        "detection_frame": "odom_ned",
                        "height_offset_topic": "/uav2/height_offset",
                        "output_detections_topic": "/uav2/projected_3d",
                        "objects_config": "drone.yaml",
                        "ground_z": -0.2,
                    }
                ],
            ),
            Node(
                package="bb_filters",
                executable="cluster_detected_objects_3d.py",
                name="cluster_det_obj_3d",
                parameters=[
                    {
                        "objects_config": "drone.yaml",
                        "pose_frame": "odom_ned",
                        "detected_objects_3d_topic": "/uav2/projected_3d",
                        "cluster_interval": 2.0,
                        "queue_size": 10,
                        "min_cluster_size": 2,
                        "min_samples": 1,
                    }
                ],
            ),
        ]
    )
