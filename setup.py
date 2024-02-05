from setuptools import setup
import os
from glob import glob

package_name = "bb_filters"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
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
            "lidar_detected_object_2d_filter = bb_filters.lidar_detected_object_2d_filter:main",
            "detected_object_3d_filter = bb_filters.detected_object_3d_filter:main",
            "lidar_backproject = bb_filters.lidar_backproject:main",
            "lidar_backproject_ml = bb_filters.lidar_backproject_ml:main",
            "ukf_filter = bb_filters.UKF_Tracker:main",
        ],
    },
)
