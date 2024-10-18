from launch import LaunchDescription
from launch_ros.actions import Node


detected_object_3d_vis = [
    "/uav2/projected_3d"
]
detected_object_3d_vis_tf = [
    # "/asv4/robotx/filtered_detections",
]
detected_object_2d_vis = [
    "/actual/detections_2d"
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
                            "camera_info_topics": [
                                "/wide/left/camera_info"
                            ],
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
                            "/actual/detections_2d",
                        ],
                        "output_detections_topic": "/uav2/projected_3d",
                        "objects_config": "drone.yaml",
                        "inflate_height": 0.1,
                        "ground_z": -0.2,
                    }
                ],
            ),
        ]
    )
