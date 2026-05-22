#!/usr/bin/env python3
from __future__ import annotations

import threading
from typing import Optional

import rclpy
import tf2_ros
from bb_filters.utils.goal_sync import GoalSynchronizer
from bb_filters.utils.pipeline import (
    TF_STATIC_QOS,
    ClusterParams,
    SpikeClusterMonitor,
    fill_spike_status,
    lookup_camera_to_odom,
    pose_to_transform_stamped,
    publish_clustered_results,
    transform_and_cluster_top_k,
)
from bb_perception_msgs.msg import ClusterSpikeStatus
from bb_perception_msgs.srv import ClusterPosesSrv
from geometry_msgs.msg import PoseArray, PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from tf2_msgs.msg import TFMessage


class ClusterPosesServiceNode(Node):
    """Service-triggered pose clustering.

    Per-call configuration (subscriber topics, clustering params, spike
    params, frame IDs) is delivered in the `ClusterPosesSrv` request on
    every `enabled=True` call. Only the publisher topic names are
    launch-time ROS parameters; everything else flows through the service.
    """

    def __init__(self) -> None:
        super().__init__("cluster_poses_service_node")

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10))

        self._tf_static_sub = self.create_subscription(
            TFMessage,
            "/tf_static",
            self._handle_tf_static,
            qos_profile=TF_STATIC_QOS,
        )

        self._service_server = self.create_service(
            ClusterPosesSrv,
            "cluster_poses_srv",
            self.cluster_srv_callback,
        )

        # Launch-time ROS parameters: publisher topic names only.
        self.output_pose_array_topic = (
            self.declare_parameter("output_pose_array_topic", "clustered_poses")
            .get_parameter_value()
            .string_value
        )
        self.spike_status_topic = (
            self.declare_parameter("spike_status_topic", "cluster_spike_status")
            .get_parameter_value()
            .string_value
        )

        # Per-call configuration. Sentinel values; overwritten on every
        # enable=True call via `_apply_request_config`.
        self.odom_topic: str = ""
        self.pose_stamped_topic: str = ""
        self.sync_queue_size: int = 100
        self.sync_tolerance: float = 0.05
        self.min_poses: int = 10
        self.cluster_num: int = 1
        self.min_cluster_size: int = 5
        self.min_samples: int = 5
        self.cluster_selection_epsilon: float = 0.0
        self.spike_tick_hz: float = 10.0
        self.spike_window_sec: float = 1.0
        self.spike_rate_threshold: float = 3.0
        self.min_seconds_between_spike_clusters: float = 0.5
        self.spike_min_poses: int = 5
        self.primary_confidence_metric: int = 0
        self.partial_cluster_max_size: int = 100

        self.pose_array_publisher = self.create_publisher(
            PoseArray, self.output_pose_array_topic, 10
        )
        self.spike_status_publisher = self.create_publisher(
            ClusterSpikeStatus, self.spike_status_topic, 10
        )
        self._static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        # State
        self.enabled: bool = False
        self._synchronized_data: list[tuple[Odometry, PoseStamped]] = []
        self._data_lock = threading.Lock()
        self._camera_to_odom_transform: Optional[TransformStamped] = None
        self._channel: GoalSynchronizer | None = None
        self._spike_monitor: SpikeClusterMonitor | None = None
        self._prev_cached_outcome: object = None
        self._spike_timer = self.create_timer(
            1.0 / self.spike_tick_hz, self._spike_tick
        )
        self._spike_cluster_frame_id = ""
        self._clustered_child_frame_id = ""
        self._spike_timer.cancel()  # start disabled

        self.get_logger().info("Cluster Poses Service Node initialized")

    def _apply_request_config(self, request: ClusterPosesSrv.Request) -> None:
        """Apply per-call configuration from the service request.

        Called on every enable=True. Sentinel-empty strings/zero values are
        not special-cased — `.srv` defaults are the only fallback.
        """
        self.odom_topic = request.odom_topic
        self.pose_stamped_topic = request.pose_stamped_topic
        self.sync_queue_size = int(request.sync_queue_size)
        self.sync_tolerance = float(request.sync_tolerance)
        self.min_poses = int(request.min_poses)
        # Number of top (largest) clusters to extract and broadcast.
        self.cluster_num = max(1, int(request.cluster_num))
        self.min_cluster_size = int(request.min_cluster_size)
        self.min_samples = int(request.min_samples)
        self.cluster_selection_epsilon = float(request.cluster_selection_epsilon)
        self.spike_window_sec = float(request.spike_window_sec)
        self.spike_rate_threshold = float(request.spike_rate_threshold)
        self.min_seconds_between_spike_clusters = float(
            request.min_seconds_between_spike_clusters
        )
        self.spike_min_poses = int(request.spike_min_poses)
        self.primary_confidence_metric = int(request.primary_confidence_metric)
        # <= 0 means unbounded.
        partial_max = int(request.partial_cluster_max_size)
        self.partial_cluster_max_size = partial_max if partial_max > 0 else int(1e9)

        new_tick_hz = float(request.spike_tick_hz)
        if new_tick_hz > 0.0 and new_tick_hz != self.spike_tick_hz:
            self.spike_tick_hz = new_tick_hz
            self._spike_timer.timer_period_ns = int(1e9 / self.spike_tick_hz)

    def _snapshot(self) -> list[tuple[Odometry, PoseStamped]]:
        with self._data_lock:
            return list(self._synchronized_data[-self.partial_cluster_max_size :])

    def _get_camera_to_odom(
        self, snapshot: list[tuple[Odometry, PoseStamped]]
    ) -> Optional[TransformStamped]:
        return (
            self._camera_to_odom_transform
            if self._ensure_camera_to_odom(snapshot)
            else None
        )

    def _cluster_params(self) -> ClusterParams:
        return ClusterParams(
            min_cluster_size=int(self.min_cluster_size),
            min_samples=int(self.min_samples),
            cluster_selection_epsilon=float(self.cluster_selection_epsilon),
        )

    def _build_spike_monitor(self) -> SpikeClusterMonitor:
        return SpikeClusterMonitor(
            window_sec=float(self.spike_window_sec),
            rate_threshold=float(self.spike_rate_threshold),
            min_seconds_between_clusters=float(self.min_seconds_between_spike_clusters),
            spike_min_poses=int(self.spike_min_poses),
        )

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _ensure_camera_to_odom(
        self, snapshot: list[tuple[Odometry, PoseStamped]]
    ) -> bool:
        if self._camera_to_odom_transform is not None:
            return True
        tf = lookup_camera_to_odom(self.tf_buffer, snapshot, timeout_sec=5.0)
        if tf is None:
            return False
        self._camera_to_odom_transform = tf
        return True

    def cluster_srv_callback(
        self, request: ClusterPosesSrv.Request, response: ClusterPosesSrv.Response
    ) -> ClusterPosesSrv.Response:
        """Service callback.

        - request.enabled == True: apply config, start accumulating
        - request.enabled == False: stop + cluster + publish
        """

        if not request.enabled:
            if not self.enabled:
                response.is_enabled = False
                response.is_cluster_success = False
                response.poses_in_cluster = 0
                response.total_poses_collected = 0
                return response

            try:
                self.enabled = False

                # Stop accepting new tuples and take a final snapshot. The
                # channel itself is destroyed in `finally`.
                if self._channel is not None:
                    self._channel._accepting = False
                with self._data_lock:
                    final_snapshot = list(self._synchronized_data)
                total_collected = len(final_snapshot)
                response.total_poses_collected = total_collected

                if total_collected < int(self.min_poses):
                    response.is_enabled = False
                    response.is_cluster_success = False
                    response.poses_in_cluster = 0
                    return response

                if not self._ensure_camera_to_odom(final_snapshot):
                    self.get_logger().error("Failed to lookup camera->odom transform")
                    response.is_enabled = False
                    response.is_cluster_success = False
                    response.poses_in_cluster = 0
                    return response

                results, transformed_poses = transform_and_cluster_top_k(
                    final_snapshot,
                    self._camera_to_odom_transform,
                    self._cluster_params(),
                    int(self.cluster_num),
                )
                if not results:
                    self.get_logger().error("No clusters found")
                    response.is_enabled = False
                    response.is_cluster_success = False
                    response.poses_in_cluster = 0
                    return response

                # Existing behaviour: publish the pose array and broadcast the
                # largest cluster under the unsuffixed `clustered_child_frame_id`.
                primary_outcome, _ = results[0]
                primary_outcome.avg_pose.header = transformed_poses[-1].header
                publish_clustered_results(
                    self.pose_array_publisher,
                    self._static_tf_broadcaster,
                    primary_outcome.avg_pose,
                    transformed_poses,
                    self._clustered_child_frame_id,
                )

                # Add-on: also broadcast each of the top-K clusters under
                # `<clustered_child_frame_id>/<i>` (i=0 is the largest, mirroring
                # the unsuffixed transform above).
                self._static_tf_broadcaster.sendTransform(
                    [
                        pose_to_transform_stamped(
                            self._stamped_avg_pose(outcome, transformed_poses),
                            f"{self._clustered_child_frame_id}/{i}",
                        )
                        for i, (outcome, _members) in enumerate(results)
                    ]
                )

                response.is_enabled = False
                response.is_cluster_success = True
                response.poses_in_cluster = int(primary_outcome.num_in_cluster)
                return response
            finally:
                # Tear down the per-goal channel. shutdown() removes the child
                # node from the executor (fencing the wait loop) and destroys it.
                if self._channel is not None:
                    channel, self._channel = self._channel, None
                    channel.shutdown(join_timeout=2)
                with self._data_lock:
                    self._synchronized_data = []
                self._camera_to_odom_transform = None
                self._spike_timer.cancel()
                self._spike_cluster_frame_id = ""
                self._clustered_child_frame_id = ""

        if self._channel is not None:
            channel = self._channel
            self._channel = None
            channel.shutdown(join_timeout=2)

        # Apply request configuration, then start accumulating.
        self._apply_request_config(request)

        self._spike_timer.reset()
        with self._data_lock:
            self._synchronized_data = []
        self._camera_to_odom_transform = None
        self._spike_monitor = self._build_spike_monitor()

        self._spike_cluster_frame_id = request.spike_cluster_frame_id or "spike/cluster"
        self._clustered_child_frame_id = (
            request.clustered_child_frame_id or "unknown/clustered"
        )

        self._prev_cached_outcome = None

        self._channel = GoalSynchronizer(
            self,
            odom_topic=self.odom_topic,
            pose_topic=self.pose_stamped_topic,
            slop=float(self.sync_tolerance),
            queue_size=int(self.sync_queue_size),
            on_synchronized=self._synchronized_callback,
        )

        self.enabled = True
        response.is_enabled = True
        response.is_cluster_success = False
        response.poses_in_cluster = 0
        response.total_poses_collected = 0
        return response

    def _spike_tick(self) -> None:
        if not self.enabled or self._spike_monitor is None:
            return
        with self._data_lock:
            poses_now = len(self._synchronized_data)
        reading = self._spike_monitor.tick(
            now_sec=self._now_sec(),
            poses_now=poses_now,
            snapshot_fn=self._snapshot,
            get_tf_fn=self._get_camera_to_odom,
            params=self._cluster_params(),
        )
        self.get_logger().info(
            f"Spike tick: {poses_now} poses, rate={reading.rate:.2f}"
        )

        msg = ClusterSpikeStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        fill_spike_status(
            msg,
            spike_detected=reading.is_spike,
            detection_rate=reading.rate,
            outcome=self._spike_monitor.cached_outcome,
            total_poses=self._spike_monitor.cached_snapshot_len,
            primary_metric=int(self.primary_confidence_metric),
        )
        self.spike_status_publisher.publish(msg)
        fresh_outcome = self._spike_monitor.cached_outcome
        if fresh_outcome is not None and fresh_outcome is not self._prev_cached_outcome:
            self._broadcast_spike_tf(fresh_outcome.avg_pose)
        self._prev_cached_outcome = fresh_outcome

    @staticmethod
    def _stamped_avg_pose(outcome, transformed_poses: list[PoseStamped]) -> PoseStamped:
        """Stamp a cluster's average pose with the final transformed header so
        all broadcast transforms share a consistent stamp/frame.
        """
        outcome.avg_pose.header = transformed_poses[-1].header
        return outcome.avg_pose

    def _broadcast_spike_tf(self, pose: PoseStamped) -> None:
        self._static_tf_broadcaster.sendTransform(
            pose_to_transform_stamped(
                pose,
                self._spike_cluster_frame_id,
                stamp=self.get_clock().now().to_msg(),
            )
        )

    def _synchronized_callback(self, odom_msg: Odometry, pose_msg: PoseStamped) -> None:
        with self._data_lock:
            self._synchronized_data.append((odom_msg, pose_msg))

    def _handle_tf_static(self, msg: TFMessage) -> None:
        for transform in msg.transforms:
            self.tf_buffer.set_transform_static(transform, "default_authority")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ClusterPosesServiceNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"Unhandled exception in main: {e}")
    finally:
        if node._channel is not None:
            node._channel.shutdown(join_timeout=2.0)
        executor.shutdown()
        node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
