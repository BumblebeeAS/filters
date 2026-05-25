#!/usr/bin/env python3
from __future__ import annotations

import rclpy
from bb_filters.clustering.cluster import ClusterSortKey
from bb_perception_msgs.action import ClusterPosesAction
from cluster_poses_node import (
    ClusterParams,
    ClusterPosesNode,
    build_cluster_pose_result,
    seconds_to_duration,
)
from frames.utils.transform_ros_msgs import transform_pose_to_odom
from rclpy.action import ActionServer, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.duration import Duration
from rclpy.task import Future


class ClusterPosesActionNode(ClusterPosesNode):
    """ROS node that runs pose collection and clustering as an action."""

    def __init__(self) -> None:
        super().__init__("cluster_poses_action_node")

        self._action_server = ActionServer(
            self,
            ClusterPosesAction,
            "cluster_poses",
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
        )

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

        self.get_logger().info("Cluster Poses Action Server initialized")

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
            top_k=int(params.top_k),
            sort_key=ClusterSortKey(int(params.sort_key)),
        )

    def _reset_goal_state(self) -> None:
        self._goal_handle = None
        self._result_future = None
        self._goal_phase = "idle"
        self._reset_collection()

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
        try:
            self._start_goal(goal_handle)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed to start cluster goal: {exc}")
            goal_handle.abort()
            self._cleanup_subscribers()
            self._action_running = False
            self._reset_goal_state()
            return self._empty_result(int(goal_handle.request.params.sort_key))
        return await self._result_future

    def _start_goal(self, goal_handle: ServerGoalHandle) -> None:
        goal: ClusterPosesAction.Goal = goal_handle.request
        self.get_logger().info(
            f"Accepted cluster goal for {float(goal.collection_duration):.2f}s"
        )

        self._reset_collection()
        self._collection_start_time = self.get_clock().now()
        self._collection_duration = seconds_to_duration(float(goal.collection_duration))

        feedback = ClusterPosesAction.Feedback()
        feedback.current_status = "Setting up subscribers"
        feedback.collection_progress = 0.0
        feedback.poses_collected_so_far = 0
        goal_handle.publish_feedback(feedback)

        self._start_subscribers(
            odom_topic=goal.params.odom_topic,
            pose_topics=list(goal.params.pose_stamped_topics),
            sync_tolerance=float(goal.params.sync_tolerance),
            sync_queue_size=int(goal.params.sync_queue_size),
        )
        self._goal_phase = "collecting"

    def _step_action(self) -> None:
        if not self._action_running or self._goal_handle is None:
            return

        try:
            if self._goal_phase == "collecting":
                self._step_collection(self._goal_handle)
            elif self._goal_phase == "finalizing":
                self._finalize_goal(self._goal_handle)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Error during cluster goal: {exc}")
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
        feedback.poses_collected_so_far = len(self._synchronized_data)
        goal_handle.publish_feedback(feedback)

    def _finalize_goal(self, goal_handle: ServerGoalHandle) -> None:
        goal: ClusterPosesAction.Goal = goal_handle.request
        params = goal.params

        self._cleanup_subscribers()
        synchronized_data = self._synchronized_data
        total_collected = len(synchronized_data)
        self.get_logger().info(f"Collected {total_collected} synchronized pose pairs")

        if total_collected < int(params.min_poses):
            self.get_logger().error(
                "Not enough synchronized poses collected. "
                f"Got {total_collected}, need {int(params.min_poses)}"
            )
            self._finish_action("aborted", self._empty_result(int(params.sort_key)))
            return

        feedback = ClusterPosesAction.Feedback()
        feedback.current_status = "Looking up static transform"
        feedback.collection_progress = 1.0
        feedback.poses_collected_so_far = total_collected
        goal_handle.publish_feedback(feedback)

        if not self._ensure_camera_to_odom(synchronized_data):
            self._finish_action("aborted", self._empty_result(int(params.sort_key)))
            return

        feedback.current_status = "Transforming and clustering poses"
        goal_handle.publish_feedback(feedback)

        transformed_poses = [
            transform_pose_to_odom(
                odom_msg,
                pose_msg,
                self._camera_to_odom_transforms[pose_msg.header.frame_id],
            )
            for odom_msg, pose_msg in synchronized_data
        ]
        clustered = self._cluster_poses(transformed_poses, self._cluster_params(goal))
        if not clustered:
            self._finish_action("aborted", self._empty_result(int(params.sort_key)))
            return

        last_header = transformed_poses[-1].header
        for avg_pose, _ in clustered:
            avg_pose.header = last_header
        self._publish_results(
            clustered, transformed_poses, params.clustered_child_frame_id
        )

        result = ClusterPosesAction.Result()
        result.cluster_results.header = last_header
        result.cluster_results.sort_key = int(params.sort_key)
        result.cluster_results.results = [
            build_cluster_pose_result(avg_pose, cluster_result, total_collected)
            for avg_pose, cluster_result in clustered
        ]

        self.get_logger().info(
            f"Clustering complete: {len(result.cluster_results.results)} cluster(s) "
            f"from {total_collected} poses"
        )
        self._finish_action("succeeded", result)

    def _finish_action(self, status: str, result: ClusterPosesAction.Result) -> None:
        goal_handle = self._goal_handle
        result_future = self._result_future
        if goal_handle is None or result_future is None or result_future.done():
            return

        if status == "succeeded":
            goal_handle.succeed()
        else:
            goal_handle.abort()

        result_future.set_result(result)
        self._cleanup_subscribers()
        self._action_running = False
        self._reset_goal_state()


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
