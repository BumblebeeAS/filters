from glob import glob

from setuptools import find_packages
from setuptools import setup

package_name = "bb_filters"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test", "bb_filters.archive*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml", "LICENSE"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/urdf", glob("urdf/*.urdf.xacro")),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="Samuel Foo",
    maintainer_email="fooenzesamuel@gmail.com",
    description="TODO: Package description",
    license="MIT",
    scripts=[
        "bb_filters/nodes/detected_object_3d_labelling.py",
        "bb_filters/nodes/detected_object_2d_filter_projection.py",
        "bb_filters/nodes/cluster/cluster_poses_action_node.py",
        "bb_filters/nodes/cluster/cluster_poses_node.py",
        "bb_filters/nodes/cluster/cluster_poses_service_node.py",
        "bb_filters/nodes/cluster/cluster_tf_action.py",
        "bb_filters/nodes/cluster/cluster_tf_action_server.py",
        "bb_filters/nodes/cluster/cluster_tf_service_server.py",
        "bb_filters/nodes/cluster/cluster_tf_multi_action.py",
        "bb_filters/nodes/cluster/cluster_tf_multi_action_server.py",
        "bb_filters/nodes/cluster/cluster_tf_multi_service_server.py",
        "bb_filters/nodes/cluster/cluster_slalom_tfs.py",
        "bb_filters/nodes/cluster/cluster_slalom_tfs_node.py",
    ],
)
