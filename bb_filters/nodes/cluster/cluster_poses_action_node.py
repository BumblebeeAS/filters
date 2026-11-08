#!/usr/bin/env python3
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import time
import traceback

import rclpy
from frames.utils.sync_diagnostics import stamp_ns
from bb_filters.utils.cluster.cluster import ClusterSortKey
from bb_perception_msgs.action import ClusterPosesAction, ClusterPosesStreamAction
from bb_filters.nodes.cluster.cluster_poses_node import (
    ClusterParams,
    ClusterPosesNode,
    fill_cluster_result_array,
    validate_stream_frame_ids,
)
from bb_filters.nodes.cluster.odometry_history import OdometryHistory
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.clock import Clock, ClockType
from rclpy.task import Future
from rclpy.time import Time


@dataclass
class CollectionGoal:
    """The reservation, execution state, and result future for one goal."""

    request: object
    action_type: type
    interval: float = 0.
    handle: object = None
    future: Future = field(default_factory=Future)
    ros_started: Time | None = None
    started: float = 0.
    next_cluster: float = 0.
    finalizing: bool = False
    seen: list = field(default_factory=list)

    def elapsed(self, ros_now):
        # Legacy collection follows ROS time; streaming must expire even if
        # ROS time pauses or the client disappears.
        return (time.monotonic() - self.started if self.interval
                else (ros_now - self.ros_started).nanoseconds / 1e9)

    def feedback(self, status, count, progress=0.):
        return self.action_type.Feedback(
            current_status=status, poses_collected_so_far=count,
            collection_progress=min(max(progress, 0.), 1.))

    def empty_result(self):
        result = self.action_type.Result()
        result.cluster_results.sort_key = int(self.request.params.sort_key)
        return result


class ClusterPosesActionNode(ClusterPosesNode):
    """Legacy and streaming action endpoints sharing one exclusive collector."""

    def __init__(self, node_name="cluster_poses_action_node", action_name="cluster_poses",
                 stream_action_name="stream_cluster_poses") -> None:
        super().__init__(node_name)
        self._goal: CollectionGoal | None = None
        self._odometry = OdometryHistory(self)
        self._accumulation_capacity = 10000  # Per topic; bounds streaming searches.
        self._action_server = ActionServer(
            self, ClusterPosesAction,
            self.declare_parameter("action_name", action_name).value,
            execute_callback=self._execute_callback,
            goal_callback=self._accept_legacy_goal,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
        )
        self._stream_action_server = ActionServer(
            self, ClusterPosesStreamAction,
            self.declare_parameter("stream_action_name", stream_action_name).value,
            execute_callback=self._execute_callback,
            goal_callback=self._accept_stream_goal,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
        )
        feedback_rate = self.declare_parameter("feedback_rate_hz", 10).value
        self._action_timer = self.create_timer(
            1.0 / feedback_rate, self._step_action,
            # Deadline and cancellation checks must run while ROS time is paused.
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )
        self._action_timer.cancel()
        self.get_logger().info("Cluster Poses Action Server initialized")

    def _start_subscribers(self, *, odom_topic, pose_topics, sync_tolerance,
                           sync_queue_size=None, max_detection_age_s=0.):
        relay = self._odometry.connect(odom_topic)
        super()._start_subscribers(
            odom_topic=odom_topic, pose_topics=pose_topics, sync_tolerance=sync_tolerance,
            sync_queue_size=max(int(sync_queue_size or self._sync_queue_size), self._odometry.capacity),
            max_detection_age_s=max_detection_age_s, odom_filter=relay)
        self._odometry.replay()

    def _cleanup_subscribers(self):
        self._odometry.disconnect()
        super()._cleanup_subscribers()

    @staticmethod
    def _cluster_params(params) -> ClusterParams:
        return ClusterParams(
            min_cluster_size=int(params.min_cluster_size),
            min_samples=int(params.min_samples),
            cluster_selection_epsilon=float(params.cluster_selection_epsilon),
            min_poses=int(params.min_poses),
            top_k=int(params.top_k),
            sort_key=ClusterSortKey(int(params.sort_key)),
        )

    def _collect_pose(self, odom_msg, pose_msg, stream_index):
        goal = self._goal
        if goal is not None and goal.interval:
            # Duplicate camera timestamps cannot become new supporting evidence.
            key = (pose_msg.header.frame_id, stamp_ns(pose_msg))
            seen = goal.seen[stream_index]
            if key in seen:
                return
            stream = self._stream_data[stream_index]
            if len(stream) == self._accumulation_capacity:
                oldest = stream[0][1]
                seen.discard((oldest.header.frame_id, stamp_ns(oldest)))
            seen.add(key)
        super()._collect_pose(odom_msg, pose_msg, stream_index)

    def _accept_legacy_goal(self, request):
        return self._reserve_goal(request, ClusterPosesAction)

    def _accept_stream_goal(self, request):
        if (request.collection_duration <= 0 or not math.isfinite(request.cluster_interval)
                or request.cluster_interval <= 0):
            return GoalResponse.REJECT
        return self._reserve_goal(request, ClusterPosesStreamAction, request.cluster_interval)

    def _reserve_goal(self, request, action_type, interval=0.):
        if self._goal is not None or not math.isfinite(request.collection_duration):
            return GoalResponse.REJECT
        # Reserve before execute is scheduled, including across the two endpoints.
        self._goal = CollectionGoal(request, action_type, interval)
        return GoalResponse.ACCEPT

    async def _execute_callback(self, goal_handle):
        goal = self._goal
        goal.handle = goal_handle
        try:
            self._action_timer.reset()
            self._start_goal()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed to start cluster goal: {exc}\n{traceback.format_exc()}")
            self._finish_action("aborted", goal.empty_result())
        return await goal.future

    def _start_goal(self):
        goal = self._goal
        request = goal.request
        self.get_logger().info(f"Accepted cluster goal for {request.collection_duration:.2f}s")
        self._cleanup_subscribers()
        self._reset_collection()
        goal.ros_started = self.get_clock().now()
        goal.started = time.monotonic()
        goal.next_cluster = goal.started + goal.interval
        goal.handle.publish_feedback(goal.feedback("Setting up subscribers", 0))
        params = request.params
        validate_stream_frame_ids(list(params.clustered_child_frame_ids), len(params.pose_stamped_topics))
        self._start_subscribers(
            odom_topic=params.odom_topic, pose_topics=list(params.pose_stamped_topics),
            sync_tolerance=float(params.sync_tolerance), sync_queue_size=int(params.sync_queue_size),
            max_detection_age_s=float(params.max_detection_age_s),
        )
        if goal.interval:
            self._stream_data = [deque(maxlen=self._accumulation_capacity)
                                 for _ in params.pose_stamped_topics]
            goal.seen = [set() for _ in params.pose_stamped_topics]

    def _step_action(self):
        goal = self._goal
        if goal is None or goal.handle is None:
            return
        try:
            if goal.handle.is_cancel_requested:
                result = self._snapshot(publish_tfs=False) if goal.interval else goal.action_type.Result()
                self._finish_action("canceled", result)
            elif goal.finalizing:
                self._finalize_goal()
            else:
                self._step_collection()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Error during cluster goal: {exc}\n{traceback.format_exc()}")
            self._finish_action("aborted", goal.empty_result())

    def _step_collection(self):
        goal = self._goal
        elapsed = goal.elapsed(self.get_clock().now())
        if elapsed >= goal.request.collection_duration:
            goal.finalizing = True
            return
        feedback = goal.feedback(
            "Collecting synchronized messages", self._num_collected(),
            elapsed / max(goal.request.collection_duration, 1e-9))
        if goal.interval and time.monotonic() >= goal.next_cluster:
            feedback.cluster_results = self._snapshot(publish_tfs=False).cluster_results
            goal.next_cluster = time.monotonic() + goal.interval
        goal.handle.publish_feedback(feedback)

    def _snapshot(self, *, publish_tfs):
        goal = self._goal
        params = goal.request.params
        clustered, total_collected, last_header = self._cluster_streams(
            list(params.clustered_child_frame_ids), self._cluster_params(params),
            publish_tfs=publish_tfs, inlier_header=bool(goal.interval))
        result = goal.empty_result()
        if clustered:
            fill_cluster_result_array(result.cluster_results, clustered, total_collected,
                                      last_header, int(params.sort_key))
        return result

    def _finalize_goal(self):
        goal = self._goal
        self._cleanup_subscribers()
        self.sync_diagnostics.log_summary()
        goal.handle.publish_feedback(goal.feedback("Finalizing clustering", self._num_collected(), 1.))
        # Streaming clients publish their frozen target after cancellation drains.
        result = self._snapshot(publish_tfs=not goal.interval)
        if result.cluster_results.results:
            self.get_logger().info(
                f"Clustering complete: {len(result.cluster_results.results)} cluster(s) "
                f"from {self._num_collected()} poses")
        self._finish_action("succeeded" if result.cluster_results.results else "aborted", result)

    def _finish_action(self, status, result):
        goal = self._goal
        if goal is None or goal.future.done():
            return
        if status == "succeeded":
            goal.handle.succeed()
        elif status == "canceled":
            goal.handle.canceled()
        else:
            goal.handle.abort()
        self._action_timer.cancel()
        self._cleanup_subscribers()
        self._cleanup_cluster_pose_publishers()
        self._reset_collection()
        self._goal = None
        goal.future.set_result(result)


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
