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
        # Parallel lists: one entry per stream to cluster concurrently.
        self.pose_stamped_topics: list[str] = []
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

        # State. One bucket of synchronized (odom, pose) tuples per stream;
        # one GoalSynchronizer channel per stream. Parallel with
        # `pose_stamped_topics` / `_clustered_child_frame_ids`.
        self.enabled: bool = False
        self._synchronized_data_buckets: list[list[tuple[Odometry, PoseStamped]]] = []
        self._data_lock = threading.Lock()
        self._camera_to_odom_transform: Optional[TransformStamped] = None
        self._channels: list[GoalSynchronizer] = []
        self._spike_monitor: SpikeClusterMonitor | None = None
        self._prev_cached_outcome: object = None
        self._spike_timer = self.create_timer(
            1.0 / self.spike_tick_hz, self._spike_tick
        )
        self._spike_cluster_frame_id = ""
        self._clustered_child_frame_ids: list[str] = []
        self._spike_timer.cancel()  # start disabled

        self.get_logger().info("Cluster Poses Service Node initialized")

    def _apply_request_config(self, request: ClusterPosesSrv.Request) -> None:
        """Apply per-call configuration from the service request.

        Called on every enable=True. Sentinel-empty strings/zero values are
        not special-cased — `.srv` defaults are the only fallback. Validates
        that pose_stamped_topic and clustered_child_frame_id have matching
        length; raises ValueError otherwise.
        """
        self.odom_topic = request.odom_topic
        self.pose_stamped_topics = list(request.pose_stamped_topic)
        if len(self.pose_stamped_topics) != len(request.clustered_child_frame_id):
            raise ValueError(
                "pose_stamped_topic and clustered_child_frame_id must have "
                f"matching length; got {len(self.pose_stamped_topics)} and "
                f"{len(request.clustered_child_frame_id)}"
            )
        if not self.pose_stamped_topics:
            raise ValueError("pose_stamped_topic must contain at least one entry")
        self.sync_queue_size = int(request.sync_queue_size)
        self.sync_tolerance = float(request.sync_tolerance)
        self.min_poses = int(request.min_poses)
        # Number of top (largest) clusters to extract and broadcast per stream.
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

    def _snapshot(self, stream_idx: int = 0) -> list[tuple[Odometry, PoseStamped]]:
        """Return a bounded copy of one stream's accumulated tuples.

        Used by spike detection (stream 0 only) and per-stream final
        clustering. Caller is responsible for picking a valid index.
        """
        with self._data_lock:
            if stream_idx >= len(self._synchronized_data_buckets):
                return []
            return list(
                self._synchronized_data_buckets[stream_idx][
                    -self.partial_cluster_max_size :
                ]
            )

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

                # Stop accepting new tuples on all channels and take a final
                # snapshot per stream. Channels themselves are destroyed in
                # `finally`.
                for channel in self._channels:
                    channel._accepting = False
                with self._data_lock:
                    final_snapshots = [
                        list(bucket) for bucket in self._synchronized_data_buckets
                    ]
                total_collected = sum(len(s) for s in final_snapshots)
                response.total_poses_collected = total_collected

                # Need at least min_poses across the union to bother resolving
                # the camera->odom transform.
                if total_collected < int(self.min_poses):
                    response.is_enabled = False
                    response.is_cluster_success = False
                    response.poses_in_cluster = 0
                    return response

                # camera->odom uses any non-empty snapshot for the lookup.
                tf_snapshot = next((s for s in final_snapshots if s), [])
                if not self._ensure_camera_to_odom(tf_snapshot):
                    self.get_logger().error("Failed to lookup camera->odom transform")
                    response.is_enabled = False
                    response.is_cluster_success = False
                    response.poses_in_cluster = 0
                    return response

                # Cluster + broadcast each stream independently. Streams that
                # didn't accumulate enough poses contribute nothing — they're
                # silently skipped rather than failing the whole call.
                cluster_params = self._cluster_params()
                primary_sizes_sum = 0
                any_success = False
                for stream_idx, snapshot in enumerate(final_snapshots):
                    prefix = self._clustered_child_frame_ids[stream_idx]
                    if len(snapshot) < int(self.min_poses):
                        self.get_logger().info(
                            f"Stream {stream_idx} ({prefix}): "
                            f"{len(snapshot)} < min_poses={self.min_poses}, skip"
                        )
                        continue

                    results, transformed_poses = transform_and_cluster_top_k(
                        snapshot,
                        self._camera_to_odom_transform,
                        cluster_params,
                        int(self.cluster_num),
                    )
                    if not results:
                        self.get_logger().error(
                            f"Stream {stream_idx} ({prefix}): no clusters found"
                        )
                        continue

                    # Mirror prior single-stream behaviour: publish the largest
                    # cluster under the unsuffixed prefix, and the top-K under
                    # `<prefix>/<i>`. PoseArray of cluster members goes out on
                    # the shared topic for every stream that succeeds.
                    primary_outcome, _ = results[0]
                    primary_outcome.avg_pose.header = transformed_poses[-1].header
                    publish_clustered_results(
                        self.pose_array_publisher,
                        self._static_tf_broadcaster,
                        primary_outcome.avg_pose,
                        transformed_poses,
                        prefix,
                    )
                    self._static_tf_broadcaster.sendTransform(
                        [
                            pose_to_transform_stamped(
                                self._stamped_avg_pose(outcome, transformed_poses),
                                f"{prefix}/{i}",
                            )
                            for i, (outcome, _members) in enumerate(results)
                        ]
                    )
                    primary_sizes_sum += int(primary_outcome.num_in_cluster)
                    any_success = True

                response.is_enabled = False
                response.is_cluster_success = any_success
                response.poses_in_cluster = primary_sizes_sum
                return response
            finally:
                # Tear down all per-goal channels.
                channels_to_close, self._channels = self._channels, []
                for channel in channels_to_close:
                    channel.shutdown(join_timeout=2)
                with self._data_lock:
                    self._synchronized_data_buckets = []
                self._camera_to_odom_transform = None
                self._spike_timer.cancel()
                self._spike_cluster_frame_id = ""
                self._clustered_child_frame_ids = []

        # Tear down any leftover channels from a prior partial start.
        if self._channels:
            for channel in self._channels:
                channel.shutdown(join_timeout=2)
            self._channels = []

        # Apply request configuration, then start accumulating.
        self._apply_request_config(request)

        self._spike_timer.reset()
        with self._data_lock:
            # One bucket per declared stream.
            self._synchronized_data_buckets = [[] for _ in self.pose_stamped_topics]
        self._camera_to_odom_transform = None
        self._spike_monitor = self._build_spike_monitor()

        self._spike_cluster_frame_id = request.spike_cluster_frame_id or "spike/cluster"
        # Apply the unknown/clustered fallback per entry.
        self._clustered_child_frame_ids = [
            f or "unknown/clustered" for f in request.clustered_child_frame_id
        ]

        self._prev_cached_outcome = None

        # One GoalSynchronizer per stream; the per-stream callback is bound
        # to the bucket index at creation time.
        self._channels = [
            GoalSynchronizer(
                self,
                odom_topic=self.odom_topic,
                pose_topic=topic,
                slop=float(self.sync_tolerance),
                queue_size=int(self.sync_queue_size),
                on_synchronized=self._make_stream_callback(stream_idx),
            )
            for stream_idx, topic in enumerate(self.pose_stamped_topics)
        ]

        self.enabled = True
        response.is_enabled = True
        response.is_cluster_success = False
        response.poses_in_cluster = 0
        response.total_poses_collected = 0
        return response

    def _spike_tick(self) -> None:
        # Spike detection runs on stream 0 only; multi-stream callers wanting
        # spike behaviour should put their primary topic first. The
        # `spike_cluster_frame_id` broadcast doesn't disambiguate streams,
        # so trying to spike-monitor multiple sources would mix sources.
        if not self.enabled or self._spike_monitor is None:
            return
        with self._data_lock:
            poses_now = (
                len(self._synchronized_data_buckets[0])
                if self._synchronized_data_buckets
                else 0
            )
        reading = self._spike_monitor.tick(
            now_sec=self._now_sec(),
            poses_now=poses_now,
            snapshot_fn=lambda: self._snapshot(0),
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

    def _make_stream_callback(self, stream_idx: int):
        """Return a GoalSynchronizer callback bound to a specific bucket."""

        def _on_sync(odom_msg: Odometry, pose_msg: PoseStamped) -> None:
            with self._data_lock:
                # Channels can outlive the buckets briefly during teardown;
                # guard the index so a late tuple doesn't crash.
                if stream_idx < len(self._synchronized_data_buckets):
                    self._synchronized_data_buckets[stream_idx].append(
                        (odom_msg, pose_msg)
                    )

        return _on_sync

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
        for channel in node._channels:
            channel.shutdown(join_timeout=2.0)
        node._channels = []
        executor.shutdown()
        node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
