"""Shared helpers for the cluster_poses_* in-process integration tests.

Constants, message builders, sys.path setup, and the synthetic-publisher
rig used by both the service and action tests.
"""

from __future__ import annotations

import pathlib
import sys
import time
from collections import namedtuple

from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from tf2_msgs.msg import TFMessage

CAMERA_FRAME = "camera"
BASE_FRAME = "base"
# Position the synthetic poses cluster around. Assertions in the tests use
# this same constant so they don't drift apart from the publisher.
EXPECTED_CLUSTER_X = 1.0
# Small spread so HDBSCAN classifies everything as one cluster.
POSE_NOISE = 0.005

TF_STATIC_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)

# Tuple returned by attach_synthetic_publishers. publish_timer fires every
# 20ms while reset(); calls publish a synchronized odom+pose pair.
PublisherRig = namedtuple("PublisherRig", ["tf", "odom", "pose", "publish_timer"])


def add_scripts_to_path() -> None:
    """Add filters/scripts/cluster/ to sys.path so node modules import."""
    scripts_dir = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "cluster"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


def identity_tf_static() -> TFMessage:
    """Static TF base -> camera at the identity (so camera_to_odom == identity)."""
    tf = TransformStamped()
    tf.header.frame_id = BASE_FRAME
    tf.child_frame_id = CAMERA_FRAME
    tf.transform.rotation.w = 1.0
    return TFMessage(transforms=[tf])


def make_odom(stamp) -> Odometry:
    """Odometry message: base sitting at the odom origin (identity pose)."""
    msg = Odometry()
    msg.header.stamp = stamp
    msg.header.frame_id = "odom"
    msg.child_frame_id = BASE_FRAME
    msg.pose.pose.orientation.w = 1.0
    return msg


def make_pose(stamp, x: float, y: float = 0.0, z: float = 0.0) -> PoseStamped:
    msg = PoseStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = CAMERA_FRAME
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = z
    msg.pose.orientation.w = 1.0
    return msg


def spin_for(executor, duration_sec: float) -> None:
    end = time.monotonic() + duration_sec
    while time.monotonic() < end:
        executor.spin_once(timeout_sec=0.05)


def spin_until_done(executor, future, timeout_sec: float) -> bool:
    end = time.monotonic() + timeout_sec
    while not future.done() and time.monotonic() < end:
        executor.spin_once(timeout_sec=0.05)
    return future.done()


def attach_synthetic_publishers(
    client_node, odom_topic: str, pose_topic: str
) -> PublisherRig:
    """Create /tf_static + odom + pose publishers and a periodic-publish timer.

    The timer is created in cancelled state; call ``rig.publish_timer.reset()``
    to start streaming synchronized odom+pose pairs clustered around
    (``EXPECTED_CLUSTER_X``, 0, 0), and ``rig.publish_timer.cancel()`` to stop.
    """
    tf_pub = client_node.create_publisher(TFMessage, "/tf_static", TF_STATIC_QOS)
    odom_pub = client_node.create_publisher(
        Odometry, odom_topic, qos_profile_sensor_data
    )
    pose_pub = client_node.create_publisher(
        PoseStamped, pose_topic, qos_profile_sensor_data
    )

    counter = [0]

    def publish_once():
        stamp = client_node.get_clock().now().to_msg()
        x = EXPECTED_CLUSTER_X + POSE_NOISE * ((counter[0] % 5) - 2)
        odom_pub.publish(make_odom(stamp))
        pose_pub.publish(make_pose(stamp, x=x))
        counter[0] += 1

    publish_timer = client_node.create_timer(0.02, publish_once)
    publish_timer.cancel()
    return PublisherRig(tf=tf_pub, odom=odom_pub, pose=pose_pub, publish_timer=publish_timer)
