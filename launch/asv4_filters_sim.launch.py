from launch import LaunchDescription
from launch_ros.actions import Node

# Launch File for running the lidar segmentation pipeline on the BBASV4 Simulation


def generate_launch_description():
    return LaunchDescription(
        [
            # ComposableNodeContainer(),
            Node(
                package="bb_filters",
                executable="detected_object_2d_filter_projection.py",
                name="dets_2d_projection",
                parameters=[
                    {
                        "input_detections_topics": [
                            "/asv4/vision/detections_2d/fixed",
                        ],
                        "output_detections_topic": "/asv4/vision/detections_2d/projected",
                        "objects_config": "robotx.yaml",
                        "inflate_height": 0.1,
                        "ground_z": -0.2
                    }
                ],
            ),
            Node(
                package="bb_filters",
                executable="detected_object_3d_labelling.py",
                name="det_3d_labeller",
                parameters=[
                    {
                        "detection_2d_topic": "/asv4/vision/detections_2d/fixed",
                        "detection_3d_topic": "/asv4/vision/lidar_small_objects/dets_3d",
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
        ]
    )
