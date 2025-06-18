from operator import attrgetter
from typing import List, Tuple

import numpy as np
from geometry_msgs.msg import (
    PoseStamped,
    PoseWithCovarianceStamped,
    Quaternion,
    TransformStamped,
    Vector3,
)
from rclpy.impl.rcutils_logger import RcutilsLogger
from sklearn.cluster import HDBSCAN


def get_position_from_transform(tf: TransformStamped) -> Tuple[float, float, float]:
    """Get the position tuple from a TransformStamped message.

    Args:
        tf (TransformStamped): The transform message.

    Returns:
        tuple(float, float, float): The position tuple.
    """
    return (
        tf.transform.translation.x,
        tf.transform.translation.y,
        tf.transform.translation.z,
    )


def get_orientation_from_transform(
    tf: TransformStamped,
) -> Tuple[float, float, float, float]:
    """Get the orientation tuple from a TransformStamped message.

    Args:
        tf (TransformStamped): The transform message.

    Returns:
        tuple(float, float, float, float): The orientation tuple xyzw.
    """
    return (
        tf.transform.rotation.x,
        tf.transform.rotation.y,
        tf.transform.rotation.z,
        tf.transform.rotation.w,
    )


def get_position_tuple_from_pose(
    pose: PoseWithCovarianceStamped,
) -> Tuple[float, float, float]:
    """Get the position tuple from a PoseWithCovarianceStamped message.

    Args:
        pose (PoseWithCovarianceStamped): The pose message.

    Returns:
        tuple(float, float, float): The position tuple.
    """
    return attrgetter("x", "y", "z")(pose.pose.pose.position)


def get_orientation_tuple_from_pose(
    pose: PoseWithCovarianceStamped,
) -> Tuple[float, float, float, float]:
    """Get the orientation tuple from a PoseWithCovarianceStamped message.

    Args:
        pose (PoseWithCovarianceStamped): The pose message.

    Returns:
        tuple(float, float, float, float): The orientation tuple.
    """
    return attrgetter("x", "y", "z", "w")(pose.pose.pose.orientation)


def get_covariance_from_pose(
    pose: PoseWithCovarianceStamped,
) -> np.ndarray:
    """Get the covariance matrix from a PoseWithCovarianceStamped message.

    Args:
        pose (PoseWithCovarianceStamped): The pose message.

    Returns:
        np.ndarray: The covariance matrix.
    """
    return np.array(pose.pose.covariance).reshape(6, 6)


def get_average_pose(
    pose_msgs: List[PoseWithCovarianceStamped],
    logger: RcutilsLogger = None,
) -> PoseWithCovarianceStamped:
    """Get the average pose from a list of PoseWithCovarianceStamped messages.

    Warning! The quaternion of the last pose is returned if orientation averaging fails.

    Args:
        pose_msgs (List[PoseWithCovarianceStamped]): The list of pose messages.

    Returns:
        PoseWithCovarianceStamped: The average pose message.
    """
    avg_pose = PoseWithCovarianceStamped()

    positions = np.array([get_position_tuple_from_pose(pose) for pose in pose_msgs])
    centroid = positions.mean(axis=0)
    avg_pose.pose.pose.position.x = centroid[0]
    avg_pose.pose.pose.position.y = centroid[1]
    avg_pose.pose.pose.position.z = centroid[2]

    try:
        quats = np.array([get_orientation_tuple_from_pose(pose) for pose in pose_msgs])
        quat_matrix = np.dot(quats.T, quats)
        eigvals, eigvecs = np.linalg.eigh(quat_matrix)
        avg_quat = eigvecs[:, np.argmax(eigvals)]  # eigenvector with largest eigenvalue
    except np.linalg.LinAlgError:
        avg_quat = pose_msgs[-1].pose.pose.orientation
        if logger:
            logger.warning(
                "Quaternion averaging failed, using the last pose's quaternion."
            )

    avg_pose.pose.pose.orientation.x = avg_quat[0]
    avg_pose.pose.pose.orientation.y = avg_quat[1]
    avg_pose.pose.pose.orientation.z = avg_quat[2]
    avg_pose.pose.pose.orientation.w = avg_quat[3]

    covariances = np.array([get_covariance_from_pose(pose) for pose in pose_msgs])
    avg_covariance = np.mean(covariances, axis=0)
    avg_pose.pose.covariance = avg_covariance.flatten().tolist()

    return avg_pose


def get_idxs_in_largest_cluster(
    hdbscan: HDBSCAN,
    positions: np.ndarray,
) -> np.ndarray:
    """Returns an array of indices belonging to the largest, non-noise cluster.

    If no clusters are found, an empty array is returned.
    """
    hdbscan.fit(positions)

    labels = np.array(hdbscan.labels_)
    non_noise_labels = labels[labels >= 0]

    if len(non_noise_labels) == 0:
        return np.array([])

    unique_labels, unique_label_counts = np.unique(non_noise_labels, return_counts=True)
    largest_cluster_label = unique_labels[np.argmax(unique_label_counts)]
    largest_cluster_idxs = np.where(labels == largest_cluster_label)[0]

    return largest_cluster_idxs


def tf_to_pose_stamped(tf: TransformStamped) -> PoseStamped:
    """Convert a TransformStamped message to a PoseStamped message.

    Args:
        tf (TransformStamped): The TransformStamped message.

    Returns:
        PoseStamped: The converted PoseStamped message.
    """
    pose_msg = PoseStamped()
    pose_msg.header = tf.header
    pose_msg.pose.position.x = tf.transform.translation.x
    pose_msg.pose.position.y = tf.transform.translation.y
    pose_msg.pose.position.z = tf.transform.translation.z
    pose_msg.pose.orientation = tf.transform.rotation

    return pose_msg


def average_transforms(tfs: List[TransformStamped]) -> PoseWithCovarianceStamped:
    """Average a list of TransformStamped messages into a PoseWithCovarianceStamped message.

    Args:
        tfs (List[TransformStamped]): The list of TransformStamped messages.

    Returns:
        PoseWithCovarianceStamped: The averaged pose message.
    """
    translations = np.array([get_position_from_transform(tf) for tf in tfs])
    avg_translation = translations.mean(axis=0)

    try:  # Average quaternions using eigenvector method
        quats = np.array([get_orientation_from_transform(tf) for tf in tfs])
        quat_matrix = np.dot(quats.T, quats)
        eigvals, eigvecs = np.linalg.eigh(quat_matrix)
        avg_quat = eigvecs[:, np.argmax(eigvals)]  # eigenvector with largest eigenvalue
    except np.linalg.LinAlgError:
        avg_quat = get_orientation_from_transform(
            tfs[-1]
        )  # fallback to last quaternion

    return (
        Vector3(x=avg_translation[0], y=avg_translation[1], z=avg_translation[2]),
        Quaternion(x=avg_quat[0], y=avg_quat[1], z=avg_quat[2], w=avg_quat[3]),
    )
