from launch import LaunchDescription
from launch_ros.actions import Node

#TODO clean up launch file
detected_object_3d_vis = ["/uav2/bottom_cam/projected_3d"]
detected_object_2d_vis = ["/rn"]


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
                            "camera_info_topics": ["/wide/left/camera_info"],
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
                            "/rn",
                        ],
                        "camera_info_topics": ["/wide/left/camera_info"],
                        "detection_frame": "odom_ned",
                        "height_offset_topic": "/uav2/height_offset",
                        "output_detections_topic": "/uav2/bottom_cam/projected_3d",
                        "objects_config": "drone.yaml",
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
                        "detected_objects_3d_topic": "/uav2/bottom_cam/projected_3d",
                        "cluster_interval": 2.0,
                        "queue_size": 100,
                        "min_cluster_size": 2,
                        "min_samples": 1,
                    }
                ],
            ),
        ]
    )
