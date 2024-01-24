from setuptools import setup
import os
from glob import glob

package_name = "bb_filters"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.launch.py') + glob('launch/*.launch.xml')),
        (os.path.join('share', package_name, 'urdf'),
         glob('urdf/*.urdf')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz'))
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="islabella",
    maintainer_email="bumblebeeauv@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "task_launcher = bb_vrx_2023.task_launcher_node:main",
            "stationkeep = bb_vrx_2023.basic_stationkeeping_node:main",
            "wayfinding = bb_vrx_2023.basic_wayfinding_node:main",
            "wildlife_encounter = bb_vrx_2023.basic_wildlife_encounter_node:main",
            "pinger_converter = bb_vrx_2023.pinger_local_converter:main",
            "acoustics_perception = bb_vrx_2023.basic_acoustics_perception_node:main",
            "acoustics_tracking_with_nav2 = bb_vrx_2023.acoustics_tracking_with_nav2_node:main",
            "global_path_following = bb_vrx_2023.global_path_following_node:main",
            "position_2d_filter = bb_vrx_2023.position_filter_node:main",
            "basic_perception = bb_vrx_2023.basic_perception_node:main",
            "stereo_detect = bb_vrx_2023.stereo_detected_object_fuse:main",
            "detect_track_filter = bb_vrx_2023.detected_object_3d_filter:main",
            "lidar_track_filter = bb_vrx_2023.lidar_detected_object_2d_filter:main",
            "basic_follow_the_path = bb_vrx_2023.basic_follow_the_path_node:main",
            "basic_scan_dock_deliver = bb_vrx_2023.basic_scan_dock_deliver_node:main",
            "lidar_backproject = bb_vrx_2023.lidar_backproject:main",
        ],
    },
)
