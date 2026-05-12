#!/usr/bin/env python3
from __future__ import annotations

import threading
from typing import Optional

import rclpy
import tf2_ros
from bb_perception_msgs.msg import ClusterSpikeStatus
from bb_perception_msgs.srv import ClusterTfSrv
from geometry_msgs.msg import PoseArray, PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import SetParametersResult
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from tf2_msgs.msg import TFMessage

from bb_filters.utils.goal_sync import GoalSynchronizer
from bb_filters.utils.pipeline import (
    TF_STATIC_QOS,
    ClusterParams,
    SpikeClusterMonitor,
    fill_spike_status,
    lookup_camera_to_odom,
    pose_to_transform_stamped,
    publish_clustered_results,
    transform_and_cluster,
)


class ClusterPosesServiceNode(Node):
    _SPIKE_PARAM_NAMES = frozenset(
        {
            "spike_tick_hz",
            "spike_window_sec",
            "spike_rate_threshold",
            "min_seconds_between_spike_clusters",
            "spike_min_poses",
            "primary_confidence_metric",
            "partial_cluster_max_size",
        }
    )
    _CLUSTER_PARAM_NAMES = frozenset(
        {
            "min_poses",
            "min_cluster_size",
            "min_samples",
            "cluster_selection_epsilon",
        }
    )

    # Baked into subscribers at service-enable; can only change while idle.
    _SUB_PARAM_NAMES = frozenset(
        {
            "odom_topic",
            "pose_stamped_topic",
            "sync_queue_size",
            "sync_tolerance",
        }
    )

    _POSITIVE_FLOAT_PARAMS = frozenset(
        {"spike_window_sec", "min_seconds_between_spike_clusters", "spike_tick_hz"}
    )
    _NON_NEGATIVE_INT_PARAMS = frozenset({"spike_min_poses"})
    _POSITIVE_INT_PARAMS = frozenset(
        {"min_poses", "min_cluster_size", "min_samples", "sync_queue_size"}
    )
    _NON_NEGATIVE_FLOAT_PARAMS = frozenset({"cluster_selection_epsilon"})

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
            ClusterTfSrv,
            "cluster_poses_srv",
            self.cluster_srv_callback,
        )

        # --------------------- Topics and synchronization parameters --------------------
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
        self.odom_topic = (
            self.declare_parameter("odom_topic", "/odom")
            .get_parameter_value()
            .string_value
        )
        self.pose_stamped_topic = (
            self.declare_parameter("pose_stamped_topic", "/pose")
            .get_parameter_value()
            .string_value
        )
        self.sync_queue_size = (
            self.declare_parameter("sync_queue_size", 100)
            .get_parameter_value()
            .integer_value
        )
        self.sync_tolerance = (
            self.declare_parameter("sync_tolerance", 0.05)
            .get_parameter_value()
            .double_value
        )

        # -------------------- Clustering parameters --------------------
        self.min_poses = (
            self.declare_parameter("min_poses", 10).get_parameter_value().integer_value
        )
        self.min_cluster_size = (
            self.declare_parameter("min_cluster_size", 5)
            .get_parameter_value()
            .integer_value
        )
        self.min_samples = (
            self.declare_parameter("min_samples", 5).get_parameter_value().integer_value
        )
        self.cluster_selection_epsilon = (
            self.declare_parameter("cluster_selection_epsilon", 0.0)
            .get_parameter_value()
            .double_value
        )

        # -------------------- Spike detection parameters --------------------
        self.spike_tick_hz = (
            self.declare_parameter("spike_tick_hz", 10.0)
            .get_parameter_value()
            .double_value
        )
        self.spike_window_sec = (
            self.declare_parameter("spike_window_sec", 1.0)
            .get_parameter_value()
            .double_value
        )
        self.spike_rate_threshold = (
            self.declare_parameter("spike_rate_threshold", 3.0)
            .get_parameter_value()
            .double_value
        )
        self.min_seconds_between_spike_clusters = (
            self.declare_parameter("min_seconds_between_spike_clusters", 0.5)
            .get_parameter_value()
            .double_value
        )
        self.spike_min_poses = (
            self.declare_parameter("spike_min_poses", 5)
            .get_parameter_value()
            .integer_value
        )
        self.primary_confidence_metric = (
            self.declare_parameter("primary_confidence_metric", 0)
            .get_parameter_value()
            .integer_value
        )
        self.partial_cluster_max_size = (
            self.declare_parameter("partial_cluster_max_size", 100)
            .get_parameter_value()
            .integer_value
        )

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

        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.get_logger().info("Cluster Poses Service Node initialized")

    def _on_set_parameters(self, params: list[Parameter]) -> SetParametersResult:
        # Validate first so we either apply all or none.
        handled = (
            self._SPIKE_PARAM_NAMES | self._CLUSTER_PARAM_NAMES | self._SUB_PARAM_NAMES
        )
        pending: dict[str, object] = {}
        for p in params:
            if p.name not in handled:
                continue
            if p.name in self._SUB_PARAM_NAMES and self.enabled:
                return SetParametersResult(
                    successful=False,
                    reason=(
                        f"{p.name} is baked into subscribers; stop the service "
                        "(call with enabled=False) before changing it"
                    ),
                )
            value = p.value
            if p.name == "sync_tolerance" and float(value) <= 0.0:
                return SetParametersResult(
                    successful=False, reason="sync_tolerance must be > 0"
                )
            if p.name in self._POSITIVE_FLOAT_PARAMS and float(value) <= 0.0:
                return SetParametersResult(
                    successful=False, reason=f"{p.name} must be > 0"
                )
            if p.name in self._NON_NEGATIVE_INT_PARAMS and int(value) < 0:
                return SetParametersResult(
                    successful=False, reason=f"{p.name} must be >= 0"
                )
            if p.name in self._POSITIVE_INT_PARAMS and int(value) <= 0:
                return SetParametersResult(
                    successful=False, reason=f"{p.name} must be > 0"
                )
            if p.name in self._NON_NEGATIVE_FLOAT_PARAMS and float(value) < 0.0:
                return SetParametersResult(
                    successful=False, reason=f"{p.name} must be >= 0"
                )
            if p.name == "partial_cluster_max_size" and int(value) <= 0:
                pending[p.name] = int(1e9)
            pending[p.name] = value

        if not pending:
            return SetParametersResult(successful=True)

        for name, value in pending.items():
            setattr(self, name, value)

        if self.enabled:
            if {
                "spike_window_sec",
                "spike_rate_threshold",
                "min_seconds_between_spike_clusters",
                "spike_min_poses",
            } & pending.keys():
                self._spike_monitor = self._build_spike_monitor()
                self._prev_cached_outcome = None
            if "spike_tick_hz" in pending:
                self._spike_timer.timer_period_ns = int(1e9 / float(self.spike_tick_hz))
                self._spike_timer.reset()

        return SetParametersResult(successful=True)

    def _build_spike_monitor(self) -> SpikeClusterMonitor:
        return SpikeClusterMonitor(
            window_sec=float(self.spike_window_sec),
            rate_threshold=float(self.spike_rate_threshold),
            min_seconds_between_clusters=float(self.min_seconds_between_spike_clusters),
            spike_min_poses=int(self.spike_min_poses),
        )

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
        self, request: ClusterTfSrv.Request, response: ClusterTfSrv.Response
    ) -> ClusterTfSrv.Response:
        """Service callback.

        - request.enabled == True: start accumulating
        - request.enabled == False: stop + cluster + publish
        """

        if not request.enabled:
            if not self.enabled:
                response.is_enabled = False
                response.is_cluster_success = False
                response.cluster_spread = 0.0
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

                if total_collected < int(self.min_poses):
                    response.is_enabled = False
                    response.is_cluster_success = False
                    response.cluster_spread = 0.0
                    return response

                if not self._ensure_camera_to_odom(final_snapshot):
                    self.get_logger().error("Failed to lookup camera->odom transform")
                    response.is_enabled = False
                    response.is_cluster_success = False
                    response.cluster_spread = 0.0
                    return response

                outcome, transformed_poses = transform_and_cluster(
                    final_snapshot,
                    self._camera_to_odom_transform,
                    self._cluster_params(),
                )
                if outcome is None:
                    self.get_logger().error("No clusters found")
                    response.is_enabled = False
                    response.is_cluster_success = False
                    response.cluster_spread = 0.0
                    return response

                outcome.avg_pose.header = transformed_poses[-1].header
                publish_clustered_results(
                    self.pose_array_publisher,
                    self._static_tf_broadcaster,
                    outcome.avg_pose,
                    transformed_poses,
                    self._clustered_child_frame_id,
                )

                response.is_enabled = False
                response.is_cluster_success = True
                # Reuse cluster_spread to report the number of poses in the cluster.
                response.cluster_spread = float(outcome.num_in_cluster)
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
        # Start accumulating
        self._spike_timer.reset()
        with self._data_lock:
            self._synchronized_data = []
        self._camera_to_odom_transform = None
        self._spike_monitor = self._build_spike_monitor()

        val = next(iter(request.input_child_frame_ids), None)
        self._spike_cluster_frame_id = f"{val}/spike" if val else "spike/cluster"
        val = next(iter(request.output_child_frame_ids), None)
        self._clustered_child_frame_id = val if val else "unknown/clustered"

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
        response.cluster_spread = 0.0
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
