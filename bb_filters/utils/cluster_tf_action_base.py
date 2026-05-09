#!/usr/bin/env python3

import traceback
from abc import ABC, abstractmethod

import tf2_ros
from action_msgs.msg import GoalStatus
from bb_perception_msgs.action import ClusterTfAction
from geometry_msgs.msg import PoseArray, TransformStamped
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.time import Time
from std_srvs.srv import Trigger

from bb_filters.utils.tf_lru_cache import TfCacheDict


class ClusterTfActionBase(Node, ABC):
    def __init__(self, node_name: str):
        super().__init__(node_name=node_name)

        self.declare_parameter(name="cache_size", value=10000)
        self.cache_size = (
            self.get_parameter("cache_size").get_parameter_value().integer_value
        )

        # TF components
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=False
        )

        # Action server
        self._action_server = ActionServer(
            self,
            ClusterTfAction,
            node_name,
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

        # Reset cache service
        self.reset_cache_srv = self.create_service(
            srv_type=Trigger,
            srv_name=f"{node_name}/reset_caches",
            callback=self.reset_callback,
        )

        # Caches
        self.caches: TfCacheDict = dict()

        # Pose Publishers
        self.debug_pose_publishers: dict[str, Publisher] = dict()

    ########################################
    # Service client methods
    ########################################

    def reset_callback(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """
        Service callback to reset all caches and destroy debug publishers.

        Args:
            request (Trigger.Request): Empty service request
            response (Trigger.Response): Service response object

        Returns:
            Trigger.Response: Response indicating success/failure with message
        """
        response = Trigger.Response()

        self.caches = dict()

        for pub in self.debug_pose_publishers.values():
            if not self.destroy_publisher(publisher=pub):
                self.get_logger().warn(f"Failed to destroy publisher: {pub}")
        self.debug_pose_publishers: dict[str, Publisher] = dict()

        response.success = True
        response.message = "Caches resetted and pose publishers destroyed"
        self.get_logger().info(response.message)

        return response

    ########################################
    # Action client methods
    ########################################

    def goal_callback(self, goal_request) -> GoalResponse:
        self.get_logger().info("Received goal request, accepting")
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle) -> CancelResponse:
        self.get_logger().info("Received cancel request, accepting")
        return CancelResponse.ACCEPT

    def handle_accepted(self, goal_handle) -> None:
        self.get_logger().info("Goal accepted, executing callback")
        goal_handle.execute()

    async def execute_callback(self, goal_handle) -> ClusterTfAction.Result:
        goal: ClusterTfAction.Goal = goal_handle.request
        result = ClusterTfAction.Result()

        try:
            # Setup caches
            caches = self.setup_caches(goal)

            # Collect transforms
            filled_caches, collection_result = self.collect_transforms(
                goal_handle, goal, caches
            )

            # If goal was cancelled during collection, return
            if collection_result == GoalStatus.STATUS_CANCELED:
                goal_handle.canceled()
                return result

            # Process transforms
            output_tfs, debug_poses, processing_result = self.process_transforms(
                goal_handle, goal, filled_caches
            )

            # If any error encountered during processing, abort
            if processing_result == GoalStatus.STATUS_ABORTED:
                goal_handle.abort()
                return result

            # Publish transforms
            self.static_tf_broadcaster.sendTransform(output_tfs)

            # Publish debug poses
            self.publish_debug_poses(debug_poses)

            # Cleanup resources
            self.cleanup_resources(goal, caches)

            goal_handle.succeed()
            return result
        except Exception as e:
            # This is fairly aggressive pokemon catching, but we cannot afford this action to fail and crash in the middle of a competition run. Recovery should be done through fallbacks in behaviour trees.

            self.get_logger().error(f"Error in {self.get_name()}: {e}")
            self.get_logger().warn(f"Traceback: {traceback.format_exc()}")
            goal_handle.abort()
            return result

    ########################################
    # Abstract methods
    ########################################

    @abstractmethod
    def setup_caches(self, goal: ClusterTfAction.Goal) -> TfCacheDict:
        """
        Setup transform caches based on goal parameters.

        This abstract method must be implemented by subclasses to initialise
        the appropriate caches.

        If the goal is not persistent, the caches *should not* be added to
        the persistent `self.caches` dictionary. Doing so will cause incorrect
        cleanup.

        Args:
            goal (ClusterTfAction.Goal): The clustering goal

        Returns:
            TfCacheDict: Dictionary of initialized transform caches
        """
        pass

    @abstractmethod
    def collect_once(
        self,
        goal: ClusterTfAction.Goal,
        caches: TfCacheDict,
        start_time: Time,
    ) -> None:
        """
        Collect transforms for a single iteration.

        This abstract method must be implemented by subclasses to define
        how transforms are collected and stored in caches during each
        iteration of the collection loop.

        Args:
            goal (ClusterTfAction.Goal): The clustering goal
            caches (TfCacheDict): Dictionary of transform caches to populate
            start_time (Time): Time when collection started
        """
        pass

    @abstractmethod
    def process_transforms(
        self,
        goal_handle,
        goal: ClusterTfAction.Goal,
        filled_caches: TfCacheDict,
    ) -> tuple[list[TransformStamped], dict[str, PoseArray], int]:
        """
        Process collected transforms through clustering algorithm.

        This abstract method must be implemented by subclasses to define
        the specific clustering algorithm and processing logic.

        Args:
            goal_handle: Handle to the current goal (for cancellation checking)
            goal (ClusterTfAction.Goal): The clustering goal containing configuration
            filled_caches (TfCacheDict): Dictionary of populated transform caches

        Returns:
            tuple containing:
                - list[TransformStamped]: Computed static transforms to publish
                - dict[str, PoseArray]: Debug poses for visualization
                - int: Processing result status (GoalStatus.STATUS_ABORTED / GoalStatus.STATUS_CANCELED / GoalStatus.STATUS_SUCCEEDED)
        """
        pass

    ########################################
    # Concrete methods
    ########################################

    def collect_transforms(
        self, goal_handle, goal: ClusterTfAction.Goal, caches: TfCacheDict
    ) -> tuple[TfCacheDict, int]:
        """
        Collect transforms over the specified clustering duration.

        This method runs a collection loop for the duration specified in the goal,
        calling `collect_once()` at regular intervals.

        Args:
            goal_handle: Handle to the current goal (for cancellation checking)
            goal (ClusterTfAction.Goal): Goal containing clustering_duration and tf_lookup_interval
            caches (TfCacheDict): Transform caches to populate

        Returns:
            tuple containing:
                - TfCacheDict: The populated caches
                - int: Collection result status (GoalStatus.STATUS_CANCELED / GoalStatus.STATUS_SUCCEEDED)
        """
        clustering_duration = goal.clustering_duration  # seconds
        tf_lookup_interval = goal.tf_lookup_interval

        rate = self.create_rate(1.0 / tf_lookup_interval)

        self.get_logger().info(f"Collecting TFs for {clustering_duration} seconds")
        start_time = self.get_clock().now()
        end_time = start_time + Duration(seconds=clustering_duration)

        while self.get_clock().now() < end_time:
            if goal_handle.is_cancel_requested:
                self.get_logger().warn("Goal canceled during TF collection.")
                self.destroy_rate(rate)
                return caches, GoalStatus.STATUS_CANCELED

            self.collect_once(
                goal=goal,
                caches=caches,
                start_time=start_time,
            )

            rate.sleep()

        self.destroy_rate(rate)

        return caches, GoalStatus.STATUS_SUCCEEDED

    def publish_debug_poses(self, debug_poses: dict[str, PoseArray]) -> None:
        """
        Publish debug poses for visualization.

        Creates publishers as needed for each debug topic and publishes
        the corresponding pose arrays. Publishers are cached to avoid
        repeated creation/destruction.

        Args:
            debug_poses (dict[str, PoseArray]): Dictionary mapping topic names to pose arrays
        """
        for topic_name, pose_array in debug_poses.items():
            if topic_name not in self.debug_pose_publishers:
                self.debug_pose_publishers[topic_name] = self.create_publisher(
                    PoseArray, f"{self.get_name()}/{topic_name}/poses", 10
                )

            self.debug_pose_publishers[topic_name].publish(pose_array)

    def cleanup_resources(
        self, goal: ClusterTfAction.Goal, caches: TfCacheDict
    ) -> None:
        """
        Cleanup resources based on goal persistence settings.

        If the goal is marked as persistent, caches are stored for future use.
        Otherwise, caches are cleared to free memory.

        Args:
            goal (ClusterTfAction.Goal): Goal containing persistence settings
            caches (TfCacheDict): Transform caches to manage
        """
        if goal.persistent:
            for key, cache in caches.items():
                self.caches[key] = cache
        else:
            for key, cache in caches.items():
                if key in self.caches:
                    del self.caches[key]
