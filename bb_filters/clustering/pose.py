from operator import attrgetter

import numpy as np
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped


def get_position_tuple_from_pose(pose: Pose) -> tuple[float, float, float]:
    """Get the position tuple from a Pose message.

    Args:
        pose (Pose): The pose message.

    Returns:
        tuple(float, float, float): The position tuple.
    """
    return attrgetter("x", "y", "z")(pose.position)


def get_orientation_tuple_from_pose(pose: Pose) -> tuple[float, float, float, float]:
    """Get the orientation tuple from a Pose message.

    Args:
        pose (Pose): The pose message.

    Returns:
        tuple(float, float, float, float): The orientation tuple.
    """
    return attrgetter("x", "y", "z", "w")(pose.orientation)


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


def get_average_pose(pose_msgs: list[Pose]) -> Pose:
    """Get the average pose from a list of Pose messages.

    Warning! The quaternion of the last pose is returned if orientation averaging fails.

    Args:
        pose_msgs (List[Pose]): The list of Pose messages.

    Returns:
        Pose: The average Pose message.
    """
    # TODO: Return a flag indicating if quat averaging was successful

    avg_pose = Pose()
    positions = np.array([get_position_tuple_from_pose(pose) for pose in pose_msgs])
    centroid = positions.mean(axis=0)
    avg_pose.position.x = centroid[0]
    avg_pose.position.y = centroid[1]
    avg_pose.position.z = centroid[2]

    try:
        quats = np.array([get_orientation_tuple_from_pose(pose) for pose in pose_msgs])
        quat_matrix = np.dot(quats.T, quats)
        eigvals, eigvecs = np.linalg.eigh(quat_matrix)
        avg_quat = eigvecs[:, np.argmax(eigvals)]  # eigenvector with largest eigenvalue
    except np.linalg.LinAlgError:
        avg_quat = get_orientation_tuple_from_pose(pose_msgs[-1])

    avg_pose.orientation.x = avg_quat[0]
    avg_pose.orientation.y = avg_quat[1]
    avg_pose.orientation.z = avg_quat[2]
    avg_pose.orientation.w = avg_quat[3]

    return avg_pose


def get_average_pose_with_cov(
    pose_with_cov_msgs: list[PoseWithCovarianceStamped],
) -> PoseWithCovarianceStamped:
    """Get the average pose from a list of PoseWithCovarianceStamped messages.

    Warning! The quaternion of the last pose is returned if orientation averaging fails.

    Args:
        pose_with_cov_msgs (List[PoseWithCovarianceStamped]): The list of PoseWithCovarianceStamped messages.

    Returns:
        PoseWithCovarianceStamped: The average PoseWithCovarianceStamped message.
    """
    avg_pose_with_cov = PoseWithCovarianceStamped()
    avg_pose_with_cov.header = pose_with_cov_msgs[-1].header

    pose_msgs = [pose.pose.pose for pose in pose_with_cov_msgs]
    avg_pose_with_cov.pose.pose = get_average_pose(pose_msgs)

    covariances = np.array(
        [get_covariance_from_pose(pose) for pose in pose_with_cov_msgs]
    )
    avg_covariance = np.mean(covariances, axis=0)
    avg_pose_with_cov.pose.covariance = avg_covariance.flatten().tolist()

    return avg_pose_with_cov
