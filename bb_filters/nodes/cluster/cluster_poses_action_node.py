#!/usr/bin/env python3
from __future__ import annotations

import traceback
import time
from collections import deque

from message_filters import SimpleFilter
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from frames.utils.sync_diagnostics import stamp_ns

import rclpy
from bb_filters.utils.cluster.cluster import ClusterSortKey
from bb_perception_msgs.action import ClusterPosesAction
from bb_filters.nodes.cluster.cluster_poses_node import (
    ClusterParams,
    ClusterPosesNode,
    fill_cluster_result_array,
    seconds_to_duration,
    validate_stream_frame_ids,
)
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.duration import Duration
from rclpy.task import Future


class ClusterPosesActionNode(ClusterPosesNode):
    """ROS node that runs pose collection and clustering as an action."""

    def __init__(self, node_name="cluster_poses_action_node", action_name="cluster_poses") -> None:
        super().__init__(node_name)

        self._action_server = ActionServer(
            self,
            ClusterPosesAction,
            self.declare_parameter("action_name", action_name).value,
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
        )

        # One live odometry subscription survives consecutive action windows.
        # Per-goal relays prevent old synchronizers from receiving future data.
        self._history_capacity = 2000
        self._history_seconds = 5.
        self._odom_history = deque(maxlen=self._history_capacity)
        self._persistent_odom_subscription = None
        self._persistent_odom_topic = None
        self._odom_relay = None

        self._session_id = ''
        self._session_parameters = None
        self._session_idle_since = None
        self._accumulation_capacity = 10000  # Per topic; bounds long-running searches.
        self._session_seen = []
        self._session_timer = self.create_timer(1., self._expire_idle_session)

        # State for current goal execution
        self._action_running = False
        self._goal_handle: ServerGoalHandle | None = None
        self._result_future: Future | None = None
        self._goal_phase = "idle"
        self._collection_start_time = self.get_clock().now()
        self._collection_duration = Duration(seconds=0)

        feedback_rate = (
            self.declare_parameter("feedback_rate_hz", 10)
            .get_parameter_value()
            .integer_value
        )
        self._action_timer = self.create_timer(
            1.0 / feedback_rate,
            self._step_action,
        )
        # Idle until a goal starts; reset() re-enables it in _execute_callback.
        self._action_timer.cancel()

        self.get_logger().info("Cluster Poses Action Server initialized")

    def _remember_odometry(self, message):
        stamp = stamp_ns(message)
        if self._odom_history and stamp < stamp_ns(self._odom_history[-1]):
            self._odom_history.clear()  # Never replay across a clock/epoch reset.
        self._odom_history.append(message)
        cutoff = stamp - self._history_seconds * 1e9
        while self._odom_history and stamp_ns(self._odom_history[0]) < cutoff:
            self._odom_history.popleft()
        if self._odom_relay is not None:
            self._odom_relay.signalMessage(message)

    def _start_subscribers(self, *, odom_topic, pose_topics, sync_tolerance,
                           sync_queue_size=None, max_detection_age_s=0.):
        if self._persistent_odom_topic != odom_topic:
            if self._persistent_odom_subscription is not None:
                self.destroy_subscription(self._persistent_odom_subscription)
            self._odom_history.clear()
            self._persistent_odom_topic = odom_topic
            self._persistent_odom_subscription = self.create_subscription(
                Odometry, odom_topic, self._remember_odometry, qos_profile_sensor_data)
        self._odom_relay = SimpleFilter()
        super()._start_subscribers(
            odom_topic=odom_topic, pose_topics=pose_topics, sync_tolerance=sync_tolerance,
            sync_queue_size=max(int(sync_queue_size or self._sync_queue_size), self._history_capacity),
            max_detection_age_s=max_detection_age_s, odom_filter=self._odom_relay)
        # Replay odometry only. Detection poses and clustering samples remain
        # specific to this action; nothing from a previous goal can be reused.
        now = self.get_clock().now().nanoseconds
        for message in self._odom_history:
            if now - self._history_seconds * 1e9 <= stamp_ns(message) <= now + 1e8:
                self._odom_relay.signalMessage(message)

    def _cleanup_subscribers(self):
        self._odom_relay = None
        super()._cleanup_subscribers()

    @staticmethod
    def _empty_result(sort_key: int = 0) -> ClusterPosesAction.Result:
        result = ClusterPosesAction.Result()
        result.cluster_results.sort_key = int(sort_key)
        result.cluster_results.results = []
        return result

    @staticmethod
    def _cluster_params(goal: ClusterPosesAction.Goal) -> ClusterParams:
        params = goal.params
        return ClusterParams(
            min_cluster_size=int(params.min_cluster_size),
            min_samples=int(params.min_samples),
            cluster_selection_epsilon=float(params.cluster_selection_epsilon),
            min_poses=int(params.min_poses),
            top_k=int(params.top_k),
            sort_key=ClusterSortKey(int(params.sort_key)),
        )

    def _reset_goal_state(self, keep_collection=False) -> None:
        self._goal_handle = None
        self._result_future = None
        self._goal_phase = "idle"
        if keep_collection:
            self._session_idle_since = time.monotonic()
        else:
            self._session_id = ''
            self._session_parameters = None
            self._session_idle_since = None
            self._session_seen = []
            self._reset_collection()

    def _expire_idle_session(self):
        if (not self._action_running and self._session_idle_since is not None
                and time.monotonic() - self._session_idle_since >= 10.):
            self.get_logger().info('Closing inactive clustering accumulation session')
            self._cleanup_subscribers()
            self._reset_goal_state()

    def _collect_pose(self, odom_msg, pose_msg, stream_index):
        if self._session_id:
            # Re-evaluating an accumulator must never count duplicate messages
            # as new evidence. Keep timestamps independently for each camera.
            key = (pose_msg.header.frame_id, stamp_ns(pose_msg))
            seen = self._session_seen[stream_index]
            if key in seen:
                return
            stream = self._stream_data[stream_index]
            if len(stream) == self._accumulation_capacity:
                oldest = stream[0][1]
                seen.discard((oldest.header.frame_id, stamp_ns(oldest)))
            seen.add(key)
        super()._collect_pose(odom_msg, pose_msg, stream_index)

    def _goal_callback(self, _goal_request: ClusterPosesAction.Goal) -> GoalResponse:
        self.get_logger().info("Received new goal request")
        if self._action_running:
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    async def _execute_callback(
        self,
        goal_handle: ServerGoalHandle,
    ) -> ClusterPosesAction.Result:
        self.get_logger().info("Executing goal...")
        self._action_running = True
        self._goal_handle = goal_handle
        self._result_future = Future()
        self._action_timer.reset()
        try:
            self._start_goal(goal_handle)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(
                f"Failed to start cluster goal: {exc}\n{traceback.format_exc()}"
            )
            goal_handle.abort()
            self._action_timer.cancel()
            self._cleanup_subscribers()
            self._cleanup_cluster_pose_publishers()
            self._action_running = False
            self._reset_goal_state()
            return self._empty_result(int(goal_handle.request.params.sort_key))
        return await self._result_future

    def _start_goal(self, goal_handle: ServerGoalHandle) -> None:
        goal: ClusterPosesAction.Goal = goal_handle.request
        self.get_logger().info(
            f"Accepted cluster goal for {float(goal.collection_duration):.2f}s"
        )

        # Reuse one subscriber set and accumulated observations for this search.
        # Normal goals (empty ID), new searches, or changed params always reset.
        session = goal.accumulation_id
        signature = repr(goal.params)
        reuse = (bool(session) and session == self._session_id
                 and signature == self._session_parameters and self._odom_relay is not None)
        if not reuse:
            self._cleanup_subscribers()
            self._reset_collection()
            self._session_seen = []
        self._session_id = session
        self._session_parameters = signature if session else None
        self._session_idle_since = None
        self._collection_start_time = self.get_clock().now()
        self._collection_duration = seconds_to_duration(float(goal.collection_duration))

        feedback = ClusterPosesAction.Feedback()
        feedback.current_status = "Setting up subscribers"
        feedback.collection_progress = 0.0
        feedback.poses_collected_so_far = 0
        goal_handle.publish_feedback(feedback)

        validate_stream_frame_ids(
            list(goal.params.clustered_child_frame_ids),
            len(goal.params.pose_stamped_topics),
        )
        if reuse:
            self.sync_diagnostics.clear()
            self.get_logger().info(f'Re-evaluating {self._num_collected()} accumulated poses')
        else:
            self._start_subscribers(
                odom_topic=goal.params.odom_topic,
                pose_topics=list(goal.params.pose_stamped_topics),
                sync_tolerance=float(goal.params.sync_tolerance),
                sync_queue_size=int(goal.params.sync_queue_size),
                max_detection_age_s=float(goal.params.max_detection_age_s),
            )
            if session:
                self._stream_data = [deque(maxlen=self._accumulation_capacity)
                                     for _ in goal.params.pose_stamped_topics]
                self._session_seen = [set() for _ in goal.params.pose_stamped_topics]
        self._goal_phase = "collecting"

    def _step_action(self) -> None:
        if not self._action_running or self._goal_handle is None:
            return

        try:
            if self._goal_handle.is_cancel_requested:
                self._finish_action("canceled", self._empty_result())
                return
            if self._goal_phase == "collecting":
                self._step_collection(self._goal_handle)
            elif self._goal_phase == "finalizing":
                self._finalize_goal(self._goal_handle)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(
                f"Error during cluster goal: {exc}\n{traceback.format_exc()}"
            )
            sort_key = (
                int(self._goal_handle.request.params.sort_key)
                if self._goal_handle
                else 0
            )
            self._finish_action("aborted", self._empty_result(sort_key))

    def _step_collection(self, goal_handle: ServerGoalHandle) -> None:
        elapsed = self.get_clock().now() - self._collection_start_time
        if elapsed >= self._collection_duration:
            self._goal_phase = "finalizing"
            return

        duration_ns = max(self._collection_duration.nanoseconds, 1)
        feedback = ClusterPosesAction.Feedback()
        feedback.current_status = "Collecting synchronized messages"
        feedback.collection_progress = min(elapsed.nanoseconds / duration_ns, 1.0)
        feedback.poses_collected_so_far = self._num_collected()
        goal_handle.publish_feedback(feedback)

    def _finalize_goal(self, goal_handle: ServerGoalHandle) -> None:
        goal: ClusterPosesAction.Goal = goal_handle.request
        params = goal.params

        keep_collection = bool(goal.accumulation_id)
        if not keep_collection:
            self._cleanup_subscribers()
        self.sync_diagnostics.log_summary()

        feedback = ClusterPosesAction.Feedback()
        feedback.current_status = "Finalizing clustering"
        feedback.collection_progress = 1.0
        feedback.poses_collected_so_far = self._num_collected()
        goal_handle.publish_feedback(feedback)

        clustered, total_collected, last_header = self._cluster_streams(
            list(params.clustered_child_frame_ids),
            self._cluster_params(goal),
            publish_tfs=True,
            inlier_header=keep_collection,
        )
        if not clustered:
            self._finish_action("aborted", self._empty_result(int(params.sort_key)),
                                keep_collection=keep_collection)
            return

        result = ClusterPosesAction.Result()
        fill_cluster_result_array(
            result.cluster_results,
            clustered,
            total_collected,
            last_header,
            int(params.sort_key),
        )

        self.get_logger().info(
            f"Clustering complete: {len(result.cluster_results.results)} cluster(s) "
            f"from {total_collected} poses"
        )
        self._finish_action("succeeded", result, keep_collection=keep_collection)

    def _finish_action(self, status: str, result: ClusterPosesAction.Result,
                       keep_collection=False) -> None:
        goal_handle = self._goal_handle
        result_future = self._result_future
        if goal_handle is None or result_future is None or result_future.done():
            return

        if status == "succeeded":
            goal_handle.succeed()
        elif status == "canceled":
            goal_handle.canceled()
        else:
            goal_handle.abort()

        result_future.set_result(result)
        self._action_timer.cancel()
        if not keep_collection:
            self._cleanup_subscribers()
        self._cleanup_cluster_pose_publishers()
        self._action_running = False
        self._reset_goal_state(keep_collection=keep_collection)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ClusterPosesActionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
