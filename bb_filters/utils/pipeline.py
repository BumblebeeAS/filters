"""Shared pose-clustering pipeline pieces used by cluster_poses_node and
cluster_poses_service_node.

This module wraps the HDBSCAN clustering call, the camera->odom transform
lookup, and the per-snapshot transform-then-cluster operation. The nodes
themselves stay focused on subscriber lifecycle, action/service plumbing, and
publishing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from operator import attrgetter
from typing import Optional

import numpy as np
import tf2_ros
from bb_perception_msgs.msg import ClusterSpikeStatus
from frames.utils.transform_ros_msgs import transform_pose_to_odom
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.time import Time
from sklearn.cluster import HDBSCAN  # type: ignore

from bb_filters.utils.cluster import (
    get_idxs_and_confidence_in_largest_cluster,
)
from bb_filters.utils.pose import get_average_pose

# Confidence-metric selector values; mirrors the constants on
# ClusterPosesAction.Goal so callers can pass them through unchanged.
CONFIDENCE_MEAN_PROBABILITY = 0
CONFIDENCE_CLUSTER_PERSISTENCE = 1
CONFIDENCE_INLIER_RATIO = 2
CONFIDENCE_POSITION_STD = 3

_CONFIDENCE_KEY_BY_METRIC = {
    CONFIDENCE_MEAN_PROBABILITY: "mean_probability",
    CONFIDENCE_CLUSTER_PERSISTENCE: "cluster_persistence",
    CONFIDENCE_INLIER_RATIO: "inlier_ratio",
    CONFIDENCE_POSITION_STD: "position_std",
}


def select_primary_confidence(confidence: dict[str, float], metric: int) -> float:
    return confidence.get(
        _CONFIDENCE_KEY_BY_METRIC.get(metric, "mean_probability"), 0.0
    )


@dataclass(frozen=True)
class ClusterParams:
    min_cluster_size: int
    min_samples: int
    cluster_selection_epsilon: float


@dataclass
class ClusterOutcome:
    avg_pose: PoseStamped
    num_in_cluster: int
    confidence: dict[str, float] = field(default_factory=dict)


def cluster_transformed_poses(
    transformed_poses: list[PoseStamped], params: ClusterParams
) -> Optional[ClusterOutcome]:
    """Cluster a list of poses already in the target frame. Returns None if
    there are insufficient poses or HDBSCAN finds no non-noise cluster.
    """
    if len(transformed_poses) < max(params.min_cluster_size, params.min_samples):
        return None

    hdbscan = HDBSCAN(
        min_cluster_size=params.min_cluster_size,
        min_samples=params.min_samples,
        cluster_selection_epsilon=params.cluster_selection_epsilon,
        allow_single_cluster=True,
        store_centers="centroid",
    )

    positions = np.array(
        [attrgetter("x", "y", "z")(pose.pose.position) for pose in transformed_poses]
    )
    idxs, confidence = get_idxs_and_confidence_in_largest_cluster(hdbscan, positions)
    if len(idxs) == 0:
        return None

    filtered_pose_msgs = [transformed_poses[i].pose for i in idxs]
    avg_pose = get_average_pose(filtered_pose_msgs)
    avg_stamped = PoseStamped()
    avg_stamped.pose = avg_pose
    avg_stamped.header = transformed_poses[idxs[0]].header

    return ClusterOutcome(
        avg_pose=avg_stamped,
        num_in_cluster=len(idxs),
        confidence=confidence,
    )


def lookup_camera_to_odom(
    tf_buffer: tf2_ros.Buffer,
    snapshot: list[tuple[Odometry, PoseStamped]],
    timeout_sec: float = 5.0,
) -> Optional[TransformStamped]:
    """Lookup the static transform from the pose's camera frame to the odom
    child frame using the first sample of the snapshot. Returns None if the
    snapshot is empty or the transform isn't available.
    """
    if not snapshot:
        return None
    odom_child_frame = snapshot[0][0].child_frame_id
    camera_frame_id = snapshot[0][1].header.frame_id
    try:
        return tf_buffer.lookup_transform(
            odom_child_frame,
            camera_frame_id,
            Time(),
            timeout=Duration(seconds=int(timeout_sec)),
        )
    except (
        tf2_ros.LookupException,  # type: ignore
        tf2_ros.ConnectivityException,  # type: ignore
        tf2_ros.ExtrapolationException,  # type: ignore
    ):
        return None


def transform_and_cluster(
    snapshot: list[tuple[Odometry, PoseStamped]],
    camera_to_odom: TransformStamped,
    params: ClusterParams,
) -> tuple[Optional[ClusterOutcome], list[PoseStamped]]:
    """Transform a snapshot into the odom frame and cluster it.

    Returns (outcome_or_None, transformed_poses). The transformed list is
    always returned (possibly empty) so callers can publish it as a PoseArray.
    """
    if not snapshot:
        return None, []
    transformed = [
        transform_pose_to_odom(odom, pose, camera_to_odom) for odom, pose in snapshot
    ]
    return cluster_transformed_poses(transformed, params), transformed


def fill_spike_status(
    msg: ClusterSpikeStatus,
    *,
    spike_detected: bool,
    detection_rate: float,
    outcome: Optional[ClusterOutcome],
    total_poses: int,
    primary_metric: int,
) -> None:
    """Populate a ClusterSpikeStatus message in-place. Caller sets the header."""
    msg.spike_detected = spike_detected
    msg.current_detection_rate = float(detection_rate)
    if outcome is None:
        msg.partial_cluster_available = False
        msg.partial_clustered_pose = PoseStamped()
        msg.partial_poses_in_cluster = 0
        msg.partial_total_poses = int(total_poses)
        msg.partial_mean_probability = 0.0
        msg.partial_cluster_persistence = 0.0
        msg.partial_inlier_ratio = 0.0
        msg.partial_position_std = 0.0
        msg.partial_primary_confidence = 0.0
        return
    msg.partial_cluster_available = True
    msg.partial_clustered_pose = outcome.avg_pose
    msg.partial_poses_in_cluster = int(outcome.num_in_cluster)
    msg.partial_total_poses = int(total_poses)
    msg.partial_mean_probability = outcome.confidence.get("mean_probability", 0.0)
    msg.partial_cluster_persistence = outcome.confidence.get("cluster_persistence", 0.0)
    msg.partial_inlier_ratio = outcome.confidence.get("inlier_ratio", 0.0)
    msg.partial_position_std = outcome.confidence.get("position_std", 0.0)
    msg.partial_primary_confidence = select_primary_confidence(
        outcome.confidence, primary_metric
    )
