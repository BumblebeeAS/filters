#!/usr/bin/env python3
import copy
import traceback

import numpy as np
import rclpy
from bb_robosub_msgs.srv import ClusterSlalomTfsStart, ClusterSlalomTfsStop
from cluster_slalom_tfs import get_slalom_centroids
from geometry_msgs.msg import PoseArray, Quaternion, TransformStamped, Vector3
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time
from transforms3d.euler import euler2quat

from bb_filters.utils.cluster import (
    assign_to_centroids,
    get_position_from_transform,
    tf_to_pose_stamped,
)
from bb_filters.utils.tf_lookup_node import TFLookUpSrvNode
from bb_filters.utils.tf_lru_cache import TfLruCache


def get_slalom_tfs(
    theta_enu: float, centroids: np.ndarray, frame_id: str, child_frame_ids: list[str]
) -> list[TransformStamped]:
    """Generate TransformStamped messages for slalom centroids."""
    tfs = []

    # Convert yaw from enu to ned and normalize to [-pi, pi)
    theta_ned = np.pi / 2 - theta_enu
    theta_ned = (theta_ned + np.pi) % (2 * np.pi) - np.pi
    qw, qx, qy, qz = euler2quat(0, 0, theta_ned)
    quat = Quaternion(x=qx, y=qy, z=qz, w=qw)

    for i, (x, y) in enumerate(centroids):
        tf = TransformStamped()
        tf.header.frame_id = frame_id
        tf.child_frame_id = child_frame_ids[i]

        # Convert ENU to NED
        tf.transform.translation.x = y
        tf.transform.translation.y = x
        tf.transform.translation.z = 0.0
        tf.transform.rotation = quat

        tfs.append(tf)

    return tfs


class ClusterSlalomTfsNode(TFLookUpSrvNode):
    # TODO: Make slalom tfs general to any frame

    def __init__(self):
        super().__init__("cluster_slalom_tfs_node")

        self.start_srv_server = self.create_service(
            ClusterSlalomTfsStart,
            "/auv4/cluster_slalom_tfs/start",
            self.start_srv_callback,
        )
        self.stop_srv_server = self.create_service(
            ClusterSlalomTfsStop,
            "/auv4/cluster_slalom_tfs/stop",
            self.stop_srv_callback,
        )

        self.pose_array_publisher_all = self.create_publisher(
            PoseArray, "/auv4/cluster_slalom_tfs/poses", 10
        )

    def start_srv_callback(
        self,
        request: ClusterSlalomTfsStart.Request,
        response: ClusterSlalomTfsStart.Response,
    ) -> ClusterSlalomTfsStart.Response:
        self.get_logger().info(
            "Recevied start request for slalom TFs clustering."
            f"input_child_frame_ids: {self.tf_list_in}, "
            f"reset_cache: {request.reset_cache}"
        )

        self.enabled = True
        self.num_duplicated_tfs = 0
        self.num_old_tfs = 0
        self.start_time = self.get_clock().now()

        self.tf_list_in = request.input_child_frame_ids
        if request.reset_cache:
            self.cache = TfLruCache(size=self.cache.size, logger=self.get_logger())
        response.success = True
        return response

    def stop_srv_callback(
        self,
        request: ClusterSlalomTfsStop.Request,
        response: ClusterSlalomTfsStop.Response,
    ) -> ClusterSlalomTfsStop.Response:
        self.get_logger().info(
            "Recevied stop request for slalom TFs clustering."
            f"output_parent_frame_ids: {request.output_parent_frame_ids}, "
            f"output_child_frame_ids: {request.output_child_frame_ids}, "
            f"min_cluster_size: {request.min_cluster_size}, "
            f"num_layers: {request.num_layers}"
        )

        self.enabled = False
        tfs, latest_time = self.cache.get_all()

        self.get_logger().info(
            f"Collected {len(tfs)} transforms from cache."
            f"{self.num_old_tfs}, {self.num_duplicated_tfs} old and duplicated TFs collected."
        )

        if len(tfs) == 0:
            self.get_logger().warn("No transforms available in cache.")
            return response

        # Get x, y positions from transforms and flip them to go from NED to ENU
        positions = np.array([get_position_from_transform(tf)[:2][::-1] for tf in tfs])
        theta_enu, centroids = get_slalom_centroids(
            data=positions, num_centroids=len(request.output_child_frame_ids)
        )

        # Filter centroids with number of positions less than the minimum cluster size
        assigned = assign_to_centroids(positions, centroids)
        counts = np.bincount(assigned, minlength=len(centroids))
        self.get_logger().info(
            f"Assigned positions of shape {positions.shape} to centroids of shape {counts.shape}"
        )
        self.get_logger().info(
            f"Centroids found: {len(centroids)}, "
            f"Counts: {counts}, "
            f"Min cluster size: {request.min_cluster_size}"
        )
        valid_centroids = centroids[counts >= request.min_cluster_size]

        # Get slalom Tfs in NED frame
        slalom_tfs = get_slalom_tfs(
            theta_enu=theta_enu,
            centroids=valid_centroids,
            frame_id=self.output_parent_frame,
            child_frame_ids=list(request.output_child_frame_ids),
        )

        # Sort transforms by increasing distance from the current position
        try:
            curr_pos = self.tf_buffer.lookup_transform(
                target_frame=self.output_parent_frame,
                source_frame=self.base_link_frame,
                time=Time(),
            ).transform.translation
        except Exception as e:
            self.get_logger().error(
                f"Failed to lookup transform from world to base link frame: {e}"
                f"Traceback: {traceback.format_exc()}"
            )
            return response

        slalom_tfs.sort(key=self._comparator(curr_pos))

        for tf, child_frame_id in zip(slalom_tfs, request.output_child_frame_ids):
            self.publish_transform(tf, latest_time, child_frame_id)

        # Publish debug poses
        pose_stamped_msgs = map(tf_to_pose_stamped, tfs)
        pose_array_msg = PoseArray()
        pose_array_msg.header = tfs[-1].header
        pose_array_msg.poses = [
            pose_stamped_msg.pose for pose_stamped_msg in pose_stamped_msgs
        ]
        self.pose_array_publisher_all.publish(pose_array_msg)

        if len(valid_centroids) < len(centroids):
            self.get_logger().warn("Some clusters are smaller than the minimum size.")
            response.success = False
            return response
        else:
            response.success = True
            return response

    def publish_transform(
        self, transform: TransformStamped, latest_time: Time, output_child: str
    ):
        _transform = copy.deepcopy(transform)
        _transform.header.stamp = latest_time.to_msg()
        _transform.header.frame_id = self.output_parent_frame
        _transform.child_frame_id = output_child

        translation = _transform.transform.translation
        self.get_logger().info(
            f"Transform from {self.output_parent_frame} to {output_child} "
            f"with average position: {translation.x:.3f}, {translation.y:.3f}, {translation.z:.3f}"
        )

        self.static_tf_broadcaster.sendTransform(_transform)

    @staticmethod
    def _comparator(origin: Vector3):
        def d(transform: TransformStamped) -> float:
            translation = transform.transform.translation
            return (
                ((translation.x - origin.x) ** 2)
                + ((translation.y - origin.y) ** 2)
                + ((translation.z - origin.z) ** 2)
            )

        return d


def main(args=None):
    rclpy.init(args=args)
    node = ClusterSlalomTfsNode()
    try:
        rclpy.spin(node, executor=MultiThreadedExecutor())
    except KeyboardInterrupt:
        pass
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
