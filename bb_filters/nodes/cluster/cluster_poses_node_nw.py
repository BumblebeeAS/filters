#!/usr/bin/env python3
# _nw variant of the pose-clustering base node -- WORLD-FRAME PoseArray inputs.
#
# The bin-structure estimator publishes its landmark / measurement-view
# arrays already in world_ned with a fixed labeled pose order
# ([blood_1, blood_2, fire_1, fire_2, box_1, box_2]). That kills the two
# jobs the original pipeline did per pose: there is no camera->odom
# transform to apply and therefore no odom to time-synchronize against.
# What remains is trivial: subscribe each PoseArray topic directly, buffer
# every array INDEX as its own sub-stream, cluster each sub-stream alone,
# and broadcast one frame per index (`<frame_ids[i]>_<k>` per top-k
# cluster; top_k=1 -> `<frame_ids[i]>_0`).
#
# `clustered_child_frame_ids` layouts: 1 entry merges every sub-stream into
# one cloud (presence/spike checks), or one entry per (topic, pose index)
# for the per-landmark output. The old one-per-topic layout is GONE --
# the PoseStamped-era consumers (bins.py / bin_close.py) are not served by
# this node anymore.
#
# The service/action wrappers still pass odom_topic / sync_* from the
# request; they are accepted and ignored so the request API stays put.
from __future__ import annotations

import math

from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from bb_filters.nodes.cluster.cluster_poses_node import ClusterPosesNode
from bb_filters.utils.cluster.cluster import ClusterResult

# Sub-stream cap per topic, sized for the 6-pose bin-structure arrays;
# shorter arrays leave the tail sub-streams empty.
MAX_POSES_PER_ARRAY = 6


def validate_stream_frame_ids_nw(frame_ids: list[str], num_pose_topics: int) -> None:
    """_nw layout rule: empty, 1 entry (merge everything), or one per
    (topic, pose index) pair."""
    allowed = (1, num_pose_topics * MAX_POSES_PER_ARRAY)
    if frame_ids and len(frame_ids) not in allowed:
        raise ValueError(
            "clustered_child_frame_ids must be empty, have 1 entry, or one "
            f"per pose index ({num_pose_topics * MAX_POSES_PER_ARRAY}); "
            f"got {len(frame_ids)}"
        )


class ClusterPosesNodeNw(ClusterPosesNode):
    """Collects world-frame PoseArrays and clusters each array index alone.

    Clustering, TF/pose publishing, and the service/action wrappers are
    inherited from ClusterPosesNode; subscription (plain rclpy, no odom
    synchronizer) and the transform-free clustering path live here.
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
        """Subscribe to N PoseArray topics (already world-frame).

        `odom_topic`, `sync_tolerance`, and `sync_queue_size` are accepted
        for request compatibility and ignored -- there is nothing to sync.
        """
        if not pose_topics:
            raise ValueError("pose_topics must contain at least one topic")
        self._max_detection_age_s = float(max_detection_age_s)
        self._num_pose_topics = len(pose_topics)
        self._stream_data = [[] for _ in range(len(pose_topics) * MAX_POSES_PER_ARRAY)]
        self._array_subscriptions = []
        for topic_index, topic in enumerate(pose_topics):
            self._array_subscriptions.append(
                self.create_subscription(
                    PoseArray,
                    topic,
                    lambda msg, ti=topic_index: self._collect_pose_array(msg, ti),
                    qos_profile_sensor_data,
                )
            )

    def _cleanup_subscribers(self) -> None:
        for sub in getattr(self, "_array_subscriptions", []):
            try:
                self.destroy_subscription(sub)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warning(f"Subscriber cleanup failed: {exc}")
        self._array_subscriptions = []

    def _collect_pose_array(self, pose_array_msg: PoseArray, topic_index: int) -> None:
        """Buffer pose i of topic t into sub-stream t * MAX_POSES_PER_ARRAY + i.

        Each buffered PoseStamped reuses the array's header, so the age
        filter and the published headers carry the DATA stamp/frame.
        """
        if not self._stream_data:
            return
        base = topic_index * MAX_POSES_PER_ARRAY
        for pose_index, pose in enumerate(pose_array_msg.poses[:MAX_POSES_PER_ARRAY]):
            # NaN marks "no measurement for this slot this frame"
            # (detections-only publishers) -- skip, never cluster it.
            p = pose.position
            if not (math.isfinite(p.x) and math.isfinite(p.y) and math.isfinite(p.z)):
                continue
            pose_stamped = PoseStamped()
            pose_stamped.header = pose_array_msg.header
            pose_stamped.pose = pose
            self._stream_data[base + pose_index].append(pose_stamped)

    def _resolve_stream_frames(
        self, frame_ids: list[str]
    ) -> list[tuple[list[PoseStamped], str]]:
        """1 frame: merge every sub-stream. One per (topic, pose index):
        sub-streams map 1:1 to frames, topic-major."""
        frame_ids = list(frame_ids) or ["clustered_object"]
        num_topics = getattr(self, "_num_pose_topics", None) or max(
            len(self._stream_data) // MAX_POSES_PER_ARRAY, 1
        )
        validate_stream_frame_ids_nw(frame_ids, num_topics)
        if len(frame_ids) == 1:
            merged = [pose for stream in self._stream_data for pose in stream]
            return [(merged, frame_ids[0])]
        return list(zip(self._stream_data, frame_ids))

    def _run_clustering(
        self,
        collected: list[PoseStamped],
        params,
    ) -> tuple[list[tuple[Pose, ClusterResult]], list[PoseStamped], int]:
        """Cluster one sub-stream. Poses are already world-frame, so this is
        the base pipeline minus the camera->odom lookup and transform."""
        if not collected:
            self.get_logger().error(
                f"Not enough poses collected. Got 0, need {int(params.min_poses)}"
            )
            return [], [], 0

        clustering_time = Time.from_msg(collected[-1].header.stamp)
        collected = [
            pose_msg
            for pose_msg in collected
            if not self._is_detection_too_old(clustering_time, pose_msg)
        ]
        total_collected = len(collected)

        if total_collected < int(params.min_poses) or total_collected < 2:
            self.get_logger().error(
                "Not enough poses collected. "
                f"Got {total_collected}, need {int(params.min_poses)}"
            )
            return [], [], total_collected

        clustered = self._cluster_poses(collected, params)
        return clustered, collected, total_collected
