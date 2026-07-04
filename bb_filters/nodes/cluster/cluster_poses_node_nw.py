#!/usr/bin/env python3
# _nw variant of the pose-clustering base node.
#
# Difference from ClusterPosesNode: each pose topic is subscribed as a
# PoseArray instead of a PoseStamped. The matching _nw bin pose estimator
# publishes all (<= 2) same-class center poses for a frame in ONE PoseArray
# that carries a single header/timestamp. That array is paired with odom by the
# ApproximateTimeSynchronizer as a single message, then expanded here into
# individual (odom, PoseStamped) pairs so the entire inherited clustering
# pipeline runs unchanged.
#
# This avoids the same-timestamp drop the original suffered: two PoseStamped
# sharing a header.stamp would compete for one synchronizer slot and one would
# be discarded before clustering. The shared base node is NOT modified, so every
# task that publishes plain PoseStamped keeps using ClusterPosesNode untouched.
from __future__ import annotations

from bb_filters.nodes.cluster.cluster_poses_node import ClusterPosesNode
from geometry_msgs.msg import PoseArray, PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data

# Each class publishes at most 2 bins per frame (matches the estimator's top_k).
MAX_POSES_PER_ARRAY = 2


class ClusterPosesNodeNw(ClusterPosesNode):
    """Base node that collects odom + PoseArray pairs and clusters them.

    Only the subscriber message type and collection callback differ from
    ClusterPosesNode; clustering, transforms, and publishing are inherited.
    """

    def _start_subscribers(
        self,
        *,
        odom_topic: str,
        pose_topics: list[str],
        sync_tolerance: float,
        sync_queue_size: int | None = None,
        max_detection_age_s: float = 0.0,
    ) -> None:
        """Subscribe to one odom topic and N PoseArray pose topics.

        Mirrors ClusterPosesNode._start_subscribers but each pose topic is a
        PoseArray; the per-topic accumulation buffer / stream-layout semantics
        are identical.
        """
        if not pose_topics:
            raise ValueError("pose_topics must contain at least one topic")
        self._max_detection_age_s = float(max_detection_age_s)
        self._stream_data = [[] for _ in pose_topics]
        qsize = int(sync_queue_size or self._sync_queue_size)
        self._odom_subscriber = Subscriber(
            self,
            Odometry,
            odom_topic,
            qos_profile=qos_profile_sensor_data,
        )
        for stream_index, topic in enumerate(pose_topics):
            pose_sub = Subscriber(
                self,
                PoseArray,
                topic,
                qos_profile=qos_profile_sensor_data,
            )
            self._pose_subscribers.append(pose_sub)
            time_synchronizer = ApproximateTimeSynchronizer(
                [self._odom_subscriber, pose_sub],
                queue_size=qsize,
                slop=float(sync_tolerance),
            )
            time_synchronizer.registerCallback(self._collect_pose_array, stream_index)
            self._time_synchronizers.append(time_synchronizer)

    def _collect_pose_array(
        self, odom_msg: Odometry, pose_array_msg: PoseArray, stream_index: int
    ) -> None:
        """Expand a synchronized PoseArray into per-pose (odom, PoseStamped) pairs.

        Every expanded PoseStamped reuses the array's header so the downstream
        camera->odom lookup and age filtering behave exactly as before.
        """
        for pose in pose_array_msg.poses[:MAX_POSES_PER_ARRAY]:
            pose_stamped = PoseStamped()
            pose_stamped.header = pose_array_msg.header
            pose_stamped.pose = pose
            self._collect_pose(odom_msg, pose_stamped, stream_index)
