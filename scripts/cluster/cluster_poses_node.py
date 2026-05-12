#!/usr/bin/env python3
import threading
from operator import attrgetter

import rclpy
import tf2_ros
from bb_perception_msgs.action import ClusterPosesAction
from bb_perception_msgs.msg import ClusterSpikeStatus
from geometry_msgs.msg import (
    PoseArray,
    PoseStamped,
    Quaternion,
    TransformStamped,
    Vector3,
)
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.task import Future
from tf2_msgs.msg import TFMessage

from bb_filters.utils.goal_sync import GoalSynchronizer
from bb_filters.utils.pipeline import (
    ClusterParams,
    SpikeClusterMonitor,
    fill_spike_status,
    lookup_camera_to_odom,
    select_primary_confidence,
    transform_and_cluster,
)


def seconds_to_duration(seconds: float) -> Duration:
    """Convert float seconds to rclpy Duration."""
    sec_int, sec_frac = divmod(seconds, 1)
    return Duration(seconds=int(sec_int), nanoseconds=int(round(sec_frac * 1e9)))


class ClusterPosesNode(Node):
    def __init__(self):
        super().__init__("cluster_poses_node")

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10))

        # Subscribe only to /tf_static to avoid processing dynamic TF
        static_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._tf_static_sub = self.create_subscription(
            TFMessage,
            "/tf_static",
            self._handle_tf_static,
            qos_profile=static_qos,
        )

        # Callback group for action server
        self.action_callback_group = ReentrantCallbackGroup()

        # Action server
        self._action_server = ActionServer(
            self,
            ClusterPosesAction,
            "cluster_poses",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.action_callback_group,
        )

        # State for current goal execution
        self._current_goal_handle: ServerGoalHandle | None = None
        self._synchronized_data: list[tuple[Odometry, PoseStamped]] = []
        self._data_lock = threading.Lock()
        self._camera_to_odom_transform: TransformStamped | None = None
        self._channel: GoalSynchronizer | None = None
        self._spike_monitor: SpikeClusterMonitor | None = None

        # Declare parameters
        output_pose_array_topic = (
            self.declare_parameter(
                "output_pose_array_topic",
                "clustered_poses",
            )
            .get_parameter_value()
            .string_value
        )
        spike_status_topic = (
            self.declare_parameter("spike_status_topic", "cluster_spike_status")
            .get_parameter_value()
            .string_value
        )
        self.sync_queue_size = (
            self.declare_parameter("sync_queue_size", 100)
            .get_parameter_value()
            .integer_value
        )
        self.feedback_rate = (
            self.declare_parameter("feedback_rate_hz", 10)
            .get_parameter_value()
            .integer_value
        )

        # Publishers
        self.pose_array_publisher = self.create_publisher(
            PoseArray, output_pose_array_topic, 10
        )
        self.spike_status_publisher = self.create_publisher(
            ClusterSpikeStatus, spike_status_topic, 10
        )
        self._tick_fut: Future = Future()
        self._static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        self._tick_timer = self.create_timer(
            1.0 / self.feedback_rate,
            self._tick,
            callback_group=self.action_callback_group,
        )

        self._tick_timer.cancel()

        self.get_logger().info("Cluster Poses Action Server initialized")

    def _tick(self):
        fut, self._tick_fut = self._tick_fut, Future()
        if not fut.done():
            fut.set_result(None)

    async def async_sleep(self):
        await self._tick_fut

    def goal_callback(self, goal_request: ClusterPosesAction.Goal) -> GoalResponse:
        self.get_logger().info("Received new goal request")
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle: ServerGoalHandle) -> CancelResponse:
        self.get_logger().info("Received cancel request")
        return CancelResponse.ACCEPT

    def synchronized_callback(self, odom_msg: Odometry, pose_msg: PoseStamped):
        with self._data_lock:
            self._synchronized_data.append((odom_msg, pose_msg))

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _ensure_camera_to_odom(self, snapshot: list[tuple]) -> bool:
        if self._camera_to_odom_transform is not None:
            return True
        tf = lookup_camera_to_odom(self.tf_buffer, snapshot, timeout_sec=5.0)
        if tf is None:
            return False
        self._camera_to_odom_transform = tf
        self.get_logger().info(
            f"Found transform from {snapshot[0][1].header.frame_id} to "
            f"{snapshot[0][0].child_frame_id}"
        )
        return True

    def _publish_spike_status(
        self,
        *,
        spike: bool,
        rate: float,
        outcome,
        total_poses: int,
        primary_metric: int,
        stamp_header,
    ) -> None:
        msg = ClusterSpikeStatus()
        if stamp_header is not None:
            msg.header = stamp_header
        else:
            msg.header.stamp = self.get_clock().now().to_msg()
        fill_spike_status(
            msg,
            spike_detected=spike,
            detection_rate=rate,
            outcome=outcome,
            total_poses=total_poses,
            primary_metric=primary_metric,
        )
        self.spike_status_publisher.publish(msg)

    async def execute_callback(
        self, goal_handle: ServerGoalHandle
    ) -> ClusterPosesAction.Result:
        self.get_logger().info("Executing goal...")
        self._tick_timer.reset()
        self._current_goal_handle = goal_handle
        self._synchronized_data = []
        self._camera_to_odom_transform = None

        goal: ClusterPosesAction.Goal = goal_handle.request
        feedback_msg = ClusterPosesAction.Feedback()

        cluster_params = ClusterParams(
            min_cluster_size=int(goal.min_cluster_size),
            min_samples=int(goal.min_samples),
            cluster_selection_epsilon=float(goal.cluster_selection_epsilon),
        )
        self._spike_monitor = SpikeClusterMonitor(
            window_sec=float(goal.spike_window_sec),
            rate_threshold=float(goal.spike_rate_threshold),
            min_seconds_between_clusters=float(goal.min_seconds_between_spike_clusters),
            spike_min_poses=int(goal.spike_min_poses),
        )

        try:
            feedback_msg.current_status = "Setting up subscribers"
            feedback_msg.collection_progress = 0.0
            feedback_msg.poses_collected_so_far = 0
            goal_handle.publish_feedback(feedback_msg)

            self._channel = GoalSynchronizer(
                self,
                odom_topic=goal.odom_topic,
                pose_topic=goal.pose_stamped_topic,
                slop=goal.sync_tolerance,
                queue_size=self.sync_queue_size,
                on_synchronized=self.synchronized_callback,
            )

            feedback_msg.current_status = "Collecting synchronized messages"
            goal_handle.publish_feedback(feedback_msg)

            collection_start_time = self.get_clock().now()
            collection_duration = seconds_to_duration(goal.collection_duration)

            def _snapshot_fn():
                with self._data_lock:
                    return list(self._synchronized_data)

            def _get_tf_fn(snapshot):
                return (
                    self._camera_to_odom_transform
                    if self._ensure_camera_to_odom(snapshot)
                    else None
                )

            while rclpy.ok():
                elapsed_time = self.get_clock().now() - collection_start_time
                if elapsed_time >= collection_duration:
                    break

                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    self.get_logger().info("Goal canceled")
                    return ClusterPosesAction.Result()

                with self._data_lock:
                    poses_now = len(self._synchronized_data)
                feedback_msg.collection_progress = min(
                    elapsed_time.nanoseconds / collection_duration.nanoseconds, 1.0
                )
                feedback_msg.poses_collected_so_far = poses_now
                self.get_logger().info(
                    f"Collecting... {poses_now} poses collected, "
                    f"{feedback_msg.collection_progress * 100:.1f}% complete"
                )
                goal_handle.publish_feedback(feedback_msg)

                reading = self._spike_monitor.tick(
                    now_sec=self._now_sec(),
                    poses_now=poses_now,
                    snapshot_fn=_snapshot_fn,
                    get_tf_fn=_get_tf_fn,
                    params=cluster_params,
                )
                self._publish_spike_status(
                    spike=reading.is_spike,
                    rate=reading.rate,
                    outcome=self._spike_monitor.cached_outcome,
                    total_poses=self._spike_monitor.cached_snapshot_len,
                    primary_metric=int(goal.primary_confidence_metric),
                    stamp_header=None,
                )

                await self.async_sleep()

            # Stop accepting new tuples and take a final snapshot. The channel
            # itself is destroyed in `finally`.
            if self._channel is not None:
                self._channel._accepting = False
            with self._data_lock:
                final_snapshot = list(self._synchronized_data)
            total_collected = len(final_snapshot)
            self.get_logger().info(
                f"Collected {total_collected} synchronized pose pairs"
            )

            if total_collected < goal.min_poses:
                self.get_logger().error(
                    f"Not enough synchronized poses collected. Got {total_collected}, need {goal.min_poses}"
                )
                goal_handle.abort()
                return ClusterPosesAction.Result()

            feedback_msg.current_status = "Looking up static transform"
            goal_handle.publish_feedback(feedback_msg)

            if not self._ensure_camera_to_odom(final_snapshot):
                self.get_logger().error("Failed to lookup camera->odom transform")
                goal_handle.abort()
                return ClusterPosesAction.Result()

            assert self._camera_to_odom_transform is not None  # for mypy

            feedback_msg.current_status = "Transforming and clustering poses"
            feedback_msg.collection_progress = 1.0
            goal_handle.publish_feedback(feedback_msg)

            outcome, transformed_poses = transform_and_cluster(
                final_snapshot,
                self._camera_to_odom_transform,
                cluster_params,
            )
            if outcome is None:
                self.get_logger().error("No clusters found")
                goal_handle.abort()
                return ClusterPosesAction.Result()

            outcome.avg_pose.header = transformed_poses[-1].header

            self._publish_results(
                outcome.avg_pose, transformed_poses, goal.clustered_child_frame_id
            )

            result = ClusterPosesAction.Result()
            result.clustered_pose = outcome.avg_pose
            result.total_poses_collected = total_collected
            result.poses_in_cluster = outcome.num_in_cluster
            result.mean_probability = outcome.confidence["mean_probability"]
            result.cluster_persistence = outcome.confidence["cluster_persistence"]
            result.inlier_ratio = outcome.confidence["inlier_ratio"]
            result.position_std = outcome.confidence["position_std"]
            result.primary_confidence = select_primary_confidence(
                outcome.confidence, int(goal.primary_confidence_metric)
            )

            self.get_logger().info(
                f"Clustering complete: {result.poses_in_cluster}/{result.total_poses_collected} poses in cluster"
            )

            goal_handle.succeed()
            return result

        except Exception as e:
            self.get_logger().error(f"Error during execution: {e}")
            goal_handle.abort()
            return ClusterPosesAction.Result()
        finally:
            # Tear down the per-goal channel. shutdown() removes the child
            # node from the executor (fencing the wait loop) and destroys it.
            if self._channel is not None:
                channel, self._channel = self._channel, None
                channel.shutdown()
            with self._data_lock:
                self._synchronized_data = []
            self._camera_to_odom_transform = None
            self._current_goal_handle = None
            self._tick_timer.cancel()

    def _publish_results(
        self,
        avg_pose: PoseStamped,
        transformed_poses: list[PoseStamped],
        clustered_child_frame_id: str,
    ) -> None:
        pose_array_msg = PoseArray()
        pose_array_msg.header = avg_pose.header
        pose_array_msg.poses = [pose.pose for pose in transformed_poses]
        self.pose_array_publisher.publish(pose_array_msg)

        transform_stamped = TransformStamped()
        transform_stamped.header = avg_pose.header
        transform_stamped.child_frame_id = clustered_child_frame_id
        t = attrgetter("x", "y", "z")(avg_pose.pose.position)
        qx, qy, qz, qw = attrgetter("x", "y", "z", "w")(avg_pose.pose.orientation)
        transform_stamped.transform.translation = Vector3(x=t[0], y=t[1], z=t[2])
        transform_stamped.transform.rotation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        self._static_tf_broadcaster.sendTransform(transform_stamped)

    def _handle_tf_static(self, msg: TFMessage) -> None:
        for transform in msg.transforms:
            self.tf_buffer.set_transform_static(transform, "default_authority")


def main(args=None):
    rclpy.init(args=args)
    node = ClusterPosesNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        import traceback

        node.get_logger().error(f"Unhandled exception in main: {e}")
        traceback.print_exc()
    finally:
        executor.shutdown()
        node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
