#!/usr/bin/env python3
from __future__ import annotations

from operator import attrgetter
import threading

import numpy as np
import rclpy
import tf2_ros
from bb_filters.clustering.cluster import get_idxs_and_confidence_in_largest_cluster
from bb_filters.clustering.pose import get_average_pose
from bb_perception_msgs.action import ClusterPosesAction
from frames.utils.transform_ros_msgs import transform_pose_to_odom
from geometry_msgs.msg import (
    PoseArray,
    PoseStamped,
    Quaternion,
    TransformStamped,
    Vector3,
)
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.task import Future
from rclpy.time import Time
from sklearn.cluster import HDBSCAN
from tf2_msgs.msg import TFMessage

CONFIDENCE_KEY_BY_METRIC = {
    0: "mean_probability",
    1: "cluster_persistence",
    2: "inlier_ratio",
    3: "position_std",
}

TF_STATIC_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


def seconds_to_duration(seconds: float) -> Duration:
    """Convert float seconds to rclpy Duration."""
    sec_int, sec_frac = divmod(seconds, 1)
    return Duration(seconds=int(sec_int), nanoseconds=int(round(sec_frac * 1e9)))


class ClusterPosesNode(Node):
    """ROS node that runs pose collection and clustering as an action."""

    def __init__(self) -> None:
        super().__init__("cluster_poses_node")

        self._tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10))
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

        self._action_server = ActionServer(
            self,
            ClusterPosesAction,
            "cluster_poses",
            execute_callback=self._execute_goal,
            goal_callback=self._goal_callback,
        )

        # Declare parameters
        output_pose_array_topic = (
            self.declare_parameter(
                "output_pose_array_topic",
                "clustered_poses",
            )
            .get_parameter_value()
            .string_value
        )
        self._sync_queue_size = (
            self.declare_parameter("sync_queue_size", 100)
            .get_parameter_value()
            .integer_value
        )
        feedback_rate = (
            self.declare_parameter("feedback_rate_hz", 10)
            .get_parameter_value()
            .integer_value
        )

        # Publishers
        self._pose_array_publisher = self.create_publisher(
            PoseArray,
            output_pose_array_topic,
            10,
        )
        self._static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        self._action_timer = self.create_timer(
            1.0 / feedback_rate,
            self._step_action,
        )

        self.get_logger().info("Cluster Poses Action Server initialized")

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
        self,
        snapshot: list[tuple[Odometry, PoseStamped]],
    ) -> bool:
        if self._camera_to_odom_transform is not None:
            return True
        if not snapshot:
            return False

        odom_child_frame = snapshot[0][0].child_frame_id
        camera_frame_id = snapshot[0][1].header.frame_id
        try:
            self._camera_to_odom_transform = self._tf_buffer.lookup_transform(
                odom_child_frame,
                camera_frame_id,
                Time(),
                timeout=Duration(seconds=5),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            self.get_logger().error(f"Failed to lookup transform: {exc}")
            return False

        self.get_logger().info(
            f"Found transform from {camera_frame_id} to {odom_child_frame}"
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
        self,
        goal_handle: ServerGoalHandle,
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

        feedback = ClusterPosesAction.Feedback()
        feedback.current_status = "Setting up subscribers"
        feedback.collection_progress = 0.0
        feedback.poses_collected_so_far = 0
        goal_handle.publish_feedback(feedback)

        self._start_subscribers(goal)
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
            self._finish_action("aborted", self._empty_result(total_collected))
            return

        assert self._camera_to_odom_transform is not None
        feedback.current_status = "Transforming and clustering poses"
        goal_handle.publish_feedback(feedback)

        transformed_poses = [
            transform_pose_to_odom(odom_msg, pose_msg, self._camera_to_odom_transform)
            for odom_msg, pose_msg in final_snapshot
        ]
        avg_pose, poses_in_cluster, confidence = self._cluster_poses(
            transformed_poses,
            goal,
        )
        if avg_pose is None:
            self._finish_action("aborted", self._empty_result(total_collected))
            return

        avg_pose.header = transformed_poses[-1].header
        self._publish_results(
            avg_pose,
            transformed_poses,
            goal.clustered_child_frame_id,
        )

        result = ClusterPosesAction.Result()
        result.clustered_pose = avg_pose
        result.total_poses_collected = total_collected
        result.poses_in_cluster = int(poses_in_cluster)
        result.mean_probability = confidence["mean_probability"]
        result.cluster_persistence = confidence["cluster_persistence"]
        result.inlier_ratio = confidence["inlier_ratio"]
        result.position_std = confidence["position_std"]
        result.primary_confidence = confidence.get(
            CONFIDENCE_KEY_BY_METRIC.get(int(goal.primary_confidence_metric), ""),
            0.0,
        )

        self.get_logger().info(
            "Clustering complete: "
            f"{result.poses_in_cluster}/{result.total_poses_collected} poses in cluster"
        )
        self._finish_action("succeeded", result)

    def _cluster_poses(
        self,
        transformed_poses: list[PoseStamped],
        goal: ClusterPosesAction.Goal,
    ) -> tuple[PoseStamped | None, int, dict[str, float]]:
        if len(transformed_poses) < max(
            int(goal.min_cluster_size), int(goal.min_samples)
        ):
            self.get_logger().error("Not enough poses for clustering")
            return None, 0, self._empty_confidence()

        hdbscan = HDBSCAN(
            min_cluster_size=int(goal.min_cluster_size),
            min_samples=int(goal.min_samples),
            cluster_selection_epsilon=float(goal.cluster_selection_epsilon),
            allow_single_cluster=True,
            store_centers="centroid",
        )
        positions = np.array(
            [
                attrgetter("x", "y", "z")(pose.pose.position)
                for pose in transformed_poses
            ]
        )
        idxs, confidence = get_idxs_and_confidence_in_largest_cluster(
            hdbscan,
            positions,
        )
        if len(idxs) == 0:
            self.get_logger().error("No clusters found")
            return None, 0, confidence

        filtered_pose_msgs = [transformed_poses[i].pose for i in idxs]
        avg_pose = PoseStamped()
        avg_pose.pose = get_average_pose(filtered_pose_msgs)
        avg_pose.header = transformed_poses[idxs[0]].header
        return avg_pose, len(idxs), confidence

    @staticmethod
    def _empty_confidence() -> dict[str, float]:
        return {
            "mean_probability": 0.0,
            "cluster_persistence": 0.0,
            "inlier_ratio": 0.0,
            "position_std": 0.0,
        }

    def _publish_results(
        self,
        avg_pose: PoseStamped,
        transformed_poses: list[PoseStamped],
        clustered_child_frame_id: str,
    ) -> None:
        pose_array_msg = PoseArray()
        pose_array_msg.header = avg_pose.header
        pose_array_msg.poses = [pose.pose for pose in transformed_poses]
        self._pose_array_publisher.publish(pose_array_msg)

        transform_stamped = TransformStamped()
        transform_stamped.header = avg_pose.header
        transform_stamped.child_frame_id = clustered_child_frame_id
        t = attrgetter("x", "y", "z")(avg_pose.pose.position)
        qx, qy, qz, qw = attrgetter("x", "y", "z", "w")(avg_pose.pose.orientation)
        transform_stamped.transform.translation = Vector3(x=t[0], y=t[1], z=t[2])
        transform_stamped.transform.rotation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        self._static_tf_broadcaster.sendTransform(transform_stamped)

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
