#!/usr/bin/env python3
from __future__ import annotations

import rclpy
from bb_filters.clustering.cluster import ClusterResult, ClusterSortKey
from bb_perception_msgs.msg import ClusterPoseResultArray
from bb_perception_msgs.srv import ClusterPosesSrv
from cluster_poses_node import (
    ClusterParams,
    ClusterPosesNode,
    build_cluster_pose_result,
)
from frames.utils.transform_ros_msgs import transform_pose_to_odom
from geometry_msgs.msg import PoseStamped
from rclpy.timer import Timer


class ClusterPosesServiceNode(ClusterPosesNode):
    """Service-triggered pose clustering.

    Per-call configuration (subscriber topics, clustering params, frame
    IDs) is delivered in the `ClusterPosesSrv` request on every `enabled=True`
    call. Only the publisher topic names are launch-time ROS parameters;
    everything else flows through the service.
    """

    def __init__(self) -> None:
        super().__init__("cluster_poses_service_node")

        self._service_server = self.create_service(
            ClusterPosesSrv,
            "cluster_poses_srv",
            self.cluster_srv_callback,
        )
        cluster_pose_result_topic = (
            self.declare_parameter("cluster_pose_result_topic", "cluster_pose_results")
            .get_parameter_value()
            .string_value
        )
        self._cluster_pose_result_publisher = self.create_publisher(
            ClusterPoseResultArray, cluster_pose_result_topic, 10
        )

        # Per-call configuration. Sentinel values; overwritten on every
        # enable=True call via `_apply_request_config`.
        self.enabled = False
        self.odom_topic = ""
        self.pose_stamped_topics: list[str] = []
        self.sync_tolerance = 0.05
        self.min_poses = 10
        self.min_cluster_size = 5
        self.min_samples = 5
        self.cluster_selection_epsilon = 0.0
        self.top_k = 1
        self.sort_key = int(ClusterSortKey.NUM_CLUSTER_POSES)
        self.cluster_interval = 0.0
        self._clustered_child_frame_id = ""
        self._cluster_timer: Timer | None = None

        self.get_logger().info("Cluster Poses Service Node initialized")

    def _apply_request_config(self, request: ClusterPosesSrv.Request) -> None:
        """Apply per-call configuration from the service request.

        Called on every enable=True. Sentinel-empty strings/zero values are
        not special-cased — `.srv` defaults are the only fallback.
        """
        params = request.params
        self.odom_topic = params.odom_topic
        self.pose_stamped_topics = list(params.pose_stamped_topics)
        self._sync_queue_size = int(params.sync_queue_size)
        self.sync_tolerance = float(params.sync_tolerance)
        self.min_poses = int(params.min_poses)
        self.min_cluster_size = int(params.min_cluster_size)
        self.min_samples = int(params.min_samples)
        self.cluster_selection_epsilon = float(params.cluster_selection_epsilon)
        self.top_k = int(params.top_k)
        self.sort_key = int(params.sort_key)
        self.cluster_interval = float(request.cluster_interval)
        self._clustered_child_frame_id = (
            params.clustered_child_frame_id or "unknown/clustered"
        )

    def _cluster_params(self) -> ClusterParams:
        return ClusterParams(
            min_cluster_size=int(self.min_cluster_size),
            min_samples=int(self.min_samples),
            cluster_selection_epsilon=float(self.cluster_selection_epsilon),
            top_k=int(self.top_k),
            sort_key=ClusterSortKey(int(self.sort_key)),
        )

    def cluster_srv_callback(
        self,
        request: ClusterPosesSrv.Request,
        response: ClusterPosesSrv.Response,
    ) -> ClusterPosesSrv.Response:
        """Service callback.

        - request.enabled == True: apply config, start accumulating
        - request.enabled == False: stop + cluster + publish
        """
        if request.enabled:
            return self._start_collection(request, response)
        return self._stop_and_cluster(response)

    def _start_collection(
        self,
        request: ClusterPosesSrv.Request,
        response: ClusterPosesSrv.Response,
    ) -> ClusterPosesSrv.Response:
        if self.enabled:
            self._cleanup_subscribers()
        self._cancel_cluster_timer()

        # Apply request configuration, then start accumulating.
        self._apply_request_config(request)
        self._reset_collection()
        self._start_subscribers(
            odom_topic=self.odom_topic,
            pose_topics=self.pose_stamped_topics,
            sync_tolerance=self.sync_tolerance,
            sync_queue_size=self._sync_queue_size,
        )
        if self.cluster_interval > 0.0:
            self._cluster_timer = self.create_timer(
                self.cluster_interval, self._periodic_cluster_tick
            )

        self.enabled = True
        response.is_enabled = True
        response.is_cluster_success = False
        response.cluster_results.sort_key = int(self.sort_key)
        return response

    def _stop_and_cluster(
        self,
        response: ClusterPosesSrv.Response,
    ) -> ClusterPosesSrv.Response:
        if not self.enabled:
            response.is_enabled = False
            response.is_cluster_success = False
            return response

        try:
            self.enabled = False
            self._cancel_cluster_timer()
            self._cleanup_subscribers()
            synchronized_data = self._synchronized_data
            total_collected = len(synchronized_data)
            response.cluster_results.sort_key = int(self.sort_key)

            if total_collected < int(self.min_poses):
                self.get_logger().error(
                    "Not enough synchronized poses collected. "
                    f"Got {total_collected}, need {int(self.min_poses)}"
                )
                response.is_enabled = False
                response.is_cluster_success = False
                return response

            if not self._ensure_camera_to_odom(synchronized_data):
                response.is_enabled = False
                response.is_cluster_success = False
                return response

            transformed_poses = [
                transform_pose_to_odom(
                    odom_msg,
                    pose_msg,
                    self._camera_to_odom_transforms[pose_msg.header.frame_id],
                )
                for odom_msg, pose_msg in synchronized_data
            ]
            clustered = self._cluster_poses(
                transformed_poses, self._cluster_params()
            )
            if not clustered:
                response.is_enabled = False
                response.is_cluster_success = False
                return response

            last_header = transformed_poses[-1].header
            for avg_pose, _ in clustered:
                avg_pose.header = last_header
            self._publish_results(
                clustered, transformed_poses, self._clustered_child_frame_id
            )

            self._fill_cluster_result_array(
                response.cluster_results, clustered, total_collected, last_header
            )
            self._publish_cluster_pose_result_array(
                clustered, total_collected, last_header
            )

            response.is_enabled = False
            response.is_cluster_success = True
            return response
        finally:
            self._reset_collection()
            self._clustered_child_frame_id = ""

    def _periodic_cluster_tick(self) -> None:
        """Cluster the poses collected so far and publish a ClusterPoseResultArray."""
        if not self.enabled:
            return
        synchronized_data = self._synchronized_data
        total_collected = len(synchronized_data)
        if total_collected < int(self.min_poses):
            return
        if not self._ensure_camera_to_odom(synchronized_data):
            return
        transformed_poses = [
            transform_pose_to_odom(
                odom_msg,
                pose_msg,
                self._camera_to_odom_transforms[pose_msg.header.frame_id],
            )
            for odom_msg, pose_msg in synchronized_data
        ]
        clustered = self._cluster_poses(transformed_poses, self._cluster_params())
        if not clustered:
            return
        last_header = transformed_poses[-1].header
        for avg_pose, _ in clustered:
            avg_pose.header = last_header
        self._publish_cluster_pose_result_array(
            clustered, total_collected, last_header
        )

    def _publish_cluster_pose_result_array(
        self,
        clustered: list[tuple[PoseStamped, ClusterResult]],
        num_input_poses: int,
        header,
    ) -> None:
        msg = ClusterPoseResultArray()
        self._fill_cluster_result_array(msg, clustered, num_input_poses, header)
        self._cluster_pose_result_publisher.publish(msg)

    def _fill_cluster_result_array(
        self,
        msg: ClusterPoseResultArray,
        clustered: list[tuple[PoseStamped, ClusterResult]],
        num_input_poses: int,
        header,
    ) -> None:
        msg.header = header
        msg.sort_key = int(self.sort_key)
        msg.results = [
            build_cluster_pose_result(avg_pose, cluster_result, num_input_poses)
            for avg_pose, cluster_result in clustered
        ]

    def _cancel_cluster_timer(self) -> None:
        if self._cluster_timer is None:
            return
        self._cluster_timer.cancel()
        self.destroy_timer(self._cluster_timer)
        self._cluster_timer = None


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ClusterPosesServiceNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
