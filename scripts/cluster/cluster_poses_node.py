#!/usr/bin/env python3
from __future__ import annotations

import threading

import rclpy
import tf2_ros
from bb_perception_msgs.action import ClusterPosesAction
from geometry_msgs.msg import PoseArray, PoseStamped, TransformStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.task import Future
from tf2_msgs.msg import TFMessage

from bb_filters.utils.pipeline import (
    TF_STATIC_QOS,
    ClusterParams,
    lookup_camera_to_odom,
    publish_clustered_results,
    seconds_to_duration,
    select_primary_confidence,
    transform_and_cluster,
)


class ClusterPosesNode(Node):
    """ROS node that runs pose collection and clustering as an action."""

    def __init__(self) -> None:
        super().__init__(
            "cluster_poses_node",
            automatically_declare_parameters_from_overrides=True,
        )
        log = self.get_logger()

        self._output_pose_array_topic = str(
            self._p("output_pose_array_topic", "clustered_poses")
        )
        self._sync_queue_size = int(self._p("sync_queue_size", 100))
        self._feedback_rate_hz = float(self._p("feedback_rate_hz", 10.0))
        if self._feedback_rate_hz <= 0.0:
            raise ValueError("feedback_rate_hz must be > 0")

        self._tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10))

        self._tf_static_sub = self.create_subscription(
            TFMessage,
            "/tf_static",
            self._handle_tf_static,
            qos_profile=TF_STATIC_QOS,
        )
        self._pose_array_publisher = self.create_publisher(
            PoseArray, self._output_pose_array_topic, 10
        )
        self._static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        self._action_running = False
        self._goal_handle: ServerGoalHandle | None = None
        self._result_future: Future | None = None
        self._goal_phase = "idle"

        self._synchronized_data: list[tuple[Odometry, PoseStamped]] = []
        self._data_lock = threading.Lock()
        self._camera_to_odom_transform: TransformStamped | None = None
        self._odom_sub: Subscriber | None = None
        self._pose_sub: Subscriber | None = None
        self._sync: ApproximateTimeSynchronizer | None = None

        self._collection_start_time = self.get_clock().now()
        self._collection_duration = Duration(seconds=0)
        self._cluster_params = ClusterParams(
            min_cluster_size=1,
            min_samples=1,
            cluster_selection_epsilon=0.0,
        )

        self._action_timer = self.create_timer(
            1.0 / self._feedback_rate_hz,
            self._step_action,
        )
        self._action_server = ActionServer(
            self,
            ClusterPosesAction,
            "cluster_poses",
            execute_callback=self._execute_goal,
            goal_callback=self._goal_callback,
        )

        log.info("ClusterPosesNode ready. " f"output={self._output_pose_array_topic}")

    def _p(self, name: str, default):
        value = self.get_parameter_or(name, default)
        return getattr(value, "value", value)

    def _snapshot(self) -> list[tuple[Odometry, PoseStamped]]:
        with self._data_lock:
            return self._synchronized_data

    def _handle_tf_static(self, msg: TFMessage) -> None:
        for transform in msg.transforms:
            self._tf_buffer.set_transform_static(transform, "default_authority")

    def _on_synchronized(self, odom_msg: Odometry, pose_msg: PoseStamped) -> None:
        with self._data_lock:
            self._synchronized_data.append((odom_msg, pose_msg))

    def _ensure_camera_to_odom(
        self, snapshot: list[tuple[Odometry, PoseStamped]]
    ) -> bool:
        if self._camera_to_odom_transform is not None:
            return True
        tf = lookup_camera_to_odom(self._tf_buffer, snapshot, timeout_sec=5.0)
        if tf is None:
            return False
        self._camera_to_odom_transform = tf
        self.get_logger().info(
            f"Found transform from {snapshot[0][1].header.frame_id} to "
            f"{snapshot[0][0].child_frame_id}"
        )
        return True

    @staticmethod
    def _empty_result(total_poses_collected: int = 0) -> ClusterPosesAction.Result:
        result = ClusterPosesAction.Result()
        result.total_poses_collected = int(total_poses_collected)
        result.poses_in_cluster = 0
        result.mean_probability = 0.0
        result.cluster_persistence = 0.0
        result.inlier_ratio = 0.0
        result.position_std = 0.0
        result.primary_confidence = 0.0
        return result

    def _reset_goal_state(self) -> None:
        self._goal_handle = None
        self._result_future = None
        self._goal_phase = "idle"
        self._camera_to_odom_transform = None
        with self._data_lock:
            self._synchronized_data = []

    def _start_subscribers(self, goal: ClusterPosesAction.Goal) -> None:
        self._odom_sub = Subscriber(
            self,
            Odometry,
            goal.odom_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self._pose_sub = Subscriber(
            self,
            PoseStamped,
            goal.pose_stamped_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self._sync = ApproximateTimeSynchronizer(
            [self._odom_sub, self._pose_sub],
            queue_size=self._sync_queue_size,
            slop=float(goal.sync_tolerance),
        )
        self._sync.registerCallback(self._on_synchronized)

    def _stop_subscribers(self) -> None:
        for sub in (self._odom_sub, self._pose_sub):
            if sub is None:
                continue
            try:
                self.destroy_subscription(sub.sub)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warning(f"Subscriber cleanup failed: {exc}")
        self._odom_sub = None
        self._pose_sub = None
        self._sync = None

    def _goal_callback(self, _goal_request: ClusterPosesAction.Goal) -> GoalResponse:
        if self._action_running:
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    async def _execute_goal(
        self, goal_handle: ServerGoalHandle
    ) -> ClusterPosesAction.Result:
        self._action_running = True
        self._goal_handle = goal_handle
        self._result_future = Future()
        try:
            self._start_goal(goal_handle)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed to start cluster goal: {exc}")
            goal_handle.abort()
            self._stop_subscribers()
            self._action_running = False
            self._reset_goal_state()
            return self._empty_result()
        return await self._result_future

    def _start_goal(self, goal_handle: ServerGoalHandle) -> None:
        goal: ClusterPosesAction.Goal = goal_handle.request
        self.get_logger().info(
            f"Accepted cluster goal for {float(goal.collection_duration):.2f}s"
        )

        self._camera_to_odom_transform = None
        with self._data_lock:
            self._synchronized_data = []

        self._collection_start_time = self.get_clock().now()
        self._collection_duration = seconds_to_duration(float(goal.collection_duration))
        self._cluster_params = ClusterParams(
            min_cluster_size=int(goal.min_cluster_size),
            min_samples=int(goal.min_samples),
            cluster_selection_epsilon=float(goal.cluster_selection_epsilon),
        )

        feedback = ClusterPosesAction.Feedback()
        feedback.current_status = "Setting up subscribers"
        feedback.collection_progress = 0.0
        feedback.poses_collected_so_far = 0
        goal_handle.publish_feedback(feedback)

        self._start_subscribers(goal)
        self._goal_phase = "collecting"

    def _step_action(self) -> None:
        if not self._action_running:
            return
        if self._goal_handle is None:
            return

        try:
            if self._goal_handle.is_cancel_requested:
                self._finish_action("canceled", self._empty_result())
                return
            if self._goal_phase == "collecting":
                self._step_collection(self._goal_handle)
                return
            if self._goal_phase == "finalizing":
                self._finalize_goal(self._goal_handle)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Error during cluster goal: {exc}")
            self._finish_action("aborted", self._empty_result())

    def _step_collection(self, goal_handle: ServerGoalHandle) -> None:
        elapsed = self.get_clock().now() - self._collection_start_time
        if elapsed >= self._collection_duration:
            self._goal_phase = "finalizing"
            return

        poses_now = len(self._snapshot())
        duration_ns = max(self._collection_duration.nanoseconds, 1)

        feedback = ClusterPosesAction.Feedback()
        feedback.current_status = "Collecting synchronized messages"
        feedback.collection_progress = min(elapsed.nanoseconds / duration_ns, 1.0)
        feedback.poses_collected_so_far = poses_now
        goal_handle.publish_feedback(feedback)

    def _finalize_goal(self, goal_handle: ServerGoalHandle) -> None:
        goal: ClusterPosesAction.Goal = goal_handle.request

        self._stop_subscribers()
        final_snapshot = self._snapshot()
        total_collected = len(final_snapshot)
        self.get_logger().info(f"Collected {total_collected} synchronized pose pairs")

        if total_collected < int(goal.min_poses):
            self.get_logger().error(
                "Not enough synchronized poses collected. "
                f"Got {total_collected}, need {int(goal.min_poses)}"
            )
            self._finish_action("aborted", self._empty_result(total_collected))
            return

        feedback = ClusterPosesAction.Feedback()
        feedback.current_status = "Looking up static transform"
        feedback.collection_progress = 1.0
        feedback.poses_collected_so_far = total_collected
        goal_handle.publish_feedback(feedback)

        if not self._ensure_camera_to_odom(final_snapshot):
            self.get_logger().error("Failed to lookup camera->odom transform")
            self._finish_action("aborted", self._empty_result(total_collected))
            return

        assert self._camera_to_odom_transform is not None
        feedback.current_status = "Transforming and clustering poses"
        goal_handle.publish_feedback(feedback)

        outcome, transformed_poses = transform_and_cluster(
            final_snapshot,
            self._camera_to_odom_transform,
            self._cluster_params,
        )
        if outcome is None:
            self.get_logger().error("No clusters found")
            self._finish_action("aborted", self._empty_result(total_collected))
            return

        outcome.avg_pose.header = transformed_poses[-1].header
        publish_clustered_results(
            self._pose_array_publisher,
            self._static_tf_broadcaster,
            outcome.avg_pose,
            transformed_poses,
            goal.clustered_child_frame_id,
        )

        result = ClusterPosesAction.Result()
        result.clustered_pose = outcome.avg_pose
        result.total_poses_collected = total_collected
        result.poses_in_cluster = int(outcome.num_in_cluster)
        result.mean_probability = outcome.confidence["mean_probability"]
        result.cluster_persistence = outcome.confidence["cluster_persistence"]
        result.inlier_ratio = outcome.confidence["inlier_ratio"]
        result.position_std = outcome.confidence["position_std"]
        result.primary_confidence = select_primary_confidence(
            outcome.confidence,
            int(goal.primary_confidence_metric),
        )

        self.get_logger().info(
            "Clustering complete: "
            f"{result.poses_in_cluster}/{result.total_poses_collected} poses in cluster"
        )
        self._finish_action("succeeded", result)

    def _finish_action(self, status: str, result: ClusterPosesAction.Result) -> None:
        goal_handle = self._goal_handle
        result_future = self._result_future
        if goal_handle is None or result_future is None or result_future.done():
            return

        if status == "canceled":
            goal_handle.canceled()
            self.get_logger().info("Cluster goal canceled")
        elif status == "succeeded":
            goal_handle.succeed()
        else:
            goal_handle.abort()

        result_future.set_result(result)
        self._stop_subscribers()
        self._action_running = False
        self._reset_goal_state()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ClusterPosesNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
