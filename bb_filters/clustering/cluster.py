import copy

import numpy as np
from geometry_msgs.msg import Pose, PoseStamped, Quaternion, TransformStamped, Vector3
from numpy.typing import ArrayLike
from sklearn.cluster import HDBSCAN


def euclidean_metric(v: tuple[Vector3, Quaternion], w: tuple[Vector3, Quaternion]):
    v_t = v[0]
    w_t = w[0]

    return ((v_t.x - w_t.x) ** 2) + ((v_t.y - w_t.y) ** 2) + ((v_t.z - w_t.z) ** 2)


def get_position_from_transform(tf: TransformStamped) -> tuple[float, float, float]:
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
) -> tuple[float, float, float, float]:
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


def get_top_k_clusters(
    hdbscan: HDBSCAN, tfs: list[TransformStamped], k: int
) -> list[list[TransformStamped]]:
    positions = np.array([get_position_from_transform(tf) for tf in tfs])

    labels = hdbscan.fit_predict(positions)
    non_noise_labels = labels[labels >= 0]

    cluster_sizes = np.bincount(non_noise_labels)

    top_k_labels = np.argsort(-cluster_sizes)[:k]

    return [
        [tfs[i] for i in np.asarray(labels == label).nonzero()[0]]
        for label in top_k_labels
    ]


def get_idxs_in_largest_cluster(
    hdbscan: HDBSCAN,
    positions: np.ndarray,
) -> np.ndarray:
    """
    Returns an array of indices belonging to the largest, non-noise cluster.

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
    tf_copy = copy.deepcopy(tf)
    pose_msg.header = tf_copy.header
    pose_msg.pose.position.x = tf_copy.transform.translation.x
    pose_msg.pose.position.y = tf_copy.transform.translation.y
    pose_msg.pose.position.z = tf_copy.transform.translation.z
    pose_msg.pose.orientation = tf_copy.transform.rotation

    return pose_msg


def tf_to_pose(tf: TransformStamped) -> Pose:
    """
    Convert a TransformStamped message to a Pose message.

    Args:
        tf (TransformStamped): The TransformStamped message.

    Returns:
        Pose: The converted Pose message.
    """
    pose = Pose()
    tf_copy = copy.deepcopy(tf)
    pose.position.x = tf_copy.transform.translation.x
    pose.position.y = tf_copy.transform.translation.y
    pose.position.z = tf_copy.transform.translation.z
    pose.orientation = tf_copy.transform.rotation
    return pose


def get_tfs_spread(tfs: list[TransformStamped]):
    translations = np.array(
        [get_position_from_transform(tf) for tf in tfs]
    )  # np array of (float, float, float)
    avg_translation = translations.mean(axis=0)

    distances = np.linalg.norm(translations - avg_translation, axis=1)
    return np.mean(distances)


def average_transforms(tfs: list[TransformStamped]) -> tuple[Vector3, Quaternion]:
    """Average a list of TransformStamped messages into a PoseWithCovarianceStamped message.

    Args:
        tfs (list[TransformStamped]): The list of TransformStamped messages.

    Returns:
        tuple[Vector3, Quaternion]: The averaged position and orientation.
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


def assign_to_centroids(data: ArrayLike, centroids: ArrayLike) -> np.ndarray:
    """Assign each data point to the nearest centroid.

    Args:
        data (ArrayLike): (n_samples, n_features)
        centroids (ArrayLike): (k, n_features)

    Returns:
        np.ndarray: (n_samples,) array of cluster indices
    """
    data = np.asarray(data)
    centroids = np.asarray(centroids)
    dists = np.linalg.norm(data[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    return np.argmin(dists, axis=1)
