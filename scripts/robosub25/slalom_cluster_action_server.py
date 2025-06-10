#!/usr/bin/env python3
import traceback

import numpy as np
import rclpy
import tf2_ros
from bb_perception_msgs.action import ClusterTf
from geometry_msgs.msg import Pose, Quaternion, TransformStamped, Vector3
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sklearn.cluster import HDBSCAN


class SlalomClusterActionServer(Node):
    """
    Todo hehe
    """

    def __init__(self):
        super().__init__("slalom_cluster_action_server")

        self.declare_parameter(
            name="channel_one_tf",
            value="slalom_layer_0",
        )
        self.channel_one_tf = (
            self.get_parameter("channel_one_tf").get_parameter_value().string_value
        )
        self.channel_one_tf_clustered = self.channel_one_tf + "/clustered"

        self.declare_parameter(
            name="channel_two_tf",
            value="slalom_layer_1",
        )
        self.channel_two_tf = (
            self.get_parameter("channel_two_tf").get_parameter_value().string_value
        )
        self.channel_two_tf_clustered = self.channel_two_tf + "/clustered"

        self.declare_parameter(
            name="channel_three_tf",
            value="slalom_layer_2",
        )
        self.channel_three_tf = (
            self.get_parameter("channel_three_tf").get_parameter_value().string_value
        )
        self.channel_three_tf_clustered = self.channel_three_tf + "/clustered"

        self.declare_parameter(name="output_parent_frame", value="world_ned")
        self.output_parent_frame = (
            self.get_parameter("output_parent_frame").get_parameter_value().string_value
        )

        self.declare_parameter(name="base_link_frame", value="auv4/base_link_ned")
        self.base_link_frame = (
            self.get_parameter("base_link_frame").get_parameter_value().string_value
        )

        self.declare_parameter(name="tf_lookup_interval", value=0.05)
        self.tf_lookup_interval = (
            self.get_parameter("tf_lookup_interval").get_parameter_value().double_value
        )

        self.declare_parameter(name="cache_size", value=5000)
        self.cache_size = (
            self.get_parameter("cache_size").get_parameter_value().integer_value
        )

        # TF components
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=False
        )

        self._action_server = ActionServer(
            self,
            ClusterTf,
            "/auv4/slalom",
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

        self.cache = TfLruCache(size=self.cache_size, logger=self.get_logger())

        self.get_logger().info("Slalom cluster action server initialized")

    def goal_callback(self, goal_request):
        self.get_logger().info("Received goal request, accepting")
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info("Received cancel request, accepting")
        return CancelResponse.ACCEPT

    def handle_accepted(self, goal_handle):
        self.get_logger().info("Goal accepted, executing callback")
        goal_handle.execute()

    def collect_transforms(self, goal_handle, clustering_duration):
        start_time = self.get_clock().now()
        end_time = start_time + Duration(seconds=clustering_duration)

        self.get_logger().info(f"Collecting TFs for {clustering_duration} seconds")

        rate = self.create_rate(1.0 / self.tf_lookup_interval)

        counts = {
            self.channel_one_tf: 0,
            self.channel_two_tf: 0,
            self.channel_three_tf: 0,
        }

        while self.get_clock().now() < end_time:
            if goal_handle.is_cancel_requested:
                self.get_logger().info("Goal canceled during TF collection.")
                return "CANCEL"

            for input_child in [
                self.channel_one_tf,
                self.channel_two_tf,
                self.channel_three_tf,
            ]:
                try:
                    tf = self.tf_buffer.lookup_transform(
                        target_frame=self.output_parent_frame,
                        source_frame=input_child,
                        time=Time(),  # TODO check if a timeout is needed
                    )
                    self.cache.add(tf)
                    counts[input_child] += 1
                except Exception as e:
                    self.get_logger().warn(f"Failed to lookup transform: {e}")
                    self.get_logger().warn(f"Traceback: {traceback.format_exc()}")

            try:
                rate.sleep()
            except:
                self.get_logger().info("Interrupted during TF collection.")
                return "ABORT"

        for input_child in [
            self.channel_one_tf,
            self.channel_two_tf,
            self.channel_three_tf,
        ]:
            self.get_logger().info(
                f"Collected {counts[input_child]} transforms from {self.output_parent_frame} to {input_child}"
            )

        return "SUCCESS"

    def cluster_transforms(self, tfs, min_cluster_size, min_samples) -> list[list[int]]:
        min_num_poses = max(min_cluster_size, min_samples)

        if self.cache.get_count() < min_num_poses:
            self.get_logger().warn(
                f"Not enough transforms collected to perform clustering. "
                f"Collected: {self.cache.get_count()}, Required: {min_num_poses}."
            )
            return [[], [], []]

        positions = np.array([self._get_position_from_transform(tf) for tf in tfs])

        hdbscan = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=0.05,
            allow_single_cluster=True,
            store_centers="centroid",
        )

        labels = hdbscan.fit_predict(positions)
        valid_mask = labels >= 0

        if not np.any(valid_mask):
            return [[], [], []]

        valid_labels = labels[valid_mask]

        max_label = valid_labels.max()
        counts = np.bincount(valid_labels, minlength=max_label + 1)

        top3_indices = np.argsort(-counts)[:3]

        result = []
        for cluster_id in top3_indices:
            if counts[cluster_id] > 0:
                indices = np.where(labels == cluster_id)[0].tolist()
                result.append(indices)

        while len(result) < 3:
            result.append([])

        return result

    def order_transforms(self, tfs, top_indices, curr_pos):
        average_transforms = []

        for indices in top_indices:
            if len(indices) == 0:
                continue

            filtered_tfs = [tfs[i] for i in indices]

            avg_translation, avg_rotation = self._average_transforms(filtered_tfs)
            average_transforms.append((avg_translation, avg_rotation))

        average_transforms.sort(key=SlalomClusterActionServer._comparator(curr_pos))

        return average_transforms

    def publish_transform(self, translation, rotation, latest_time, output_child):
        clustered_transform = TransformStamped()
        clustered_transform.header.stamp = latest_time.to_msg()
        clustered_transform.header.frame_id = self.output_parent_frame
        clustered_transform.child_frame_id = output_child
        clustered_transform.transform.translation = translation
        clustered_transform.transform.rotation = rotation

        self.get_logger().info(
            f"Clustered transform from {self.output_parent_frame} to {output_child} "
            f"with average position: {translation.x:.3f}, {translation.y:.3f}, {translation.z:.3f}"
        )

        self.static_tf_broadcaster.sendTransform(clustered_transform)

    async def execute_callback(self, goal_handle):
        goal = goal_handle.request
        clustering_duration = goal.clustering_duration  # seconds
        min_cluster_size = goal.min_cluster_size
        min_samples = goal.min_samples

        feedback_msg = ClusterTf.Feedback()
        result = ClusterTf.Result()

        # cache_size = int(clustering_duration / lookup_interval) + 10
        collection_result = self.collect_transforms(
            goal_handle=goal_handle, clustering_duration=clustering_duration
        )

        if collection_result == "ABORT":
            goal_handle.abort()
            return result
        elif collection_result == "CANCEL":
            goal_handle.canceled()
            return result

        tfs, latest_time = self.cache.get_all()

        top_3_indices = self.cluster_transforms(
            tfs=tfs, min_cluster_size=min_cluster_size, min_samples=min_samples
        )

        try:
            curr_pos = self.tf_buffer.lookup_transform(
                target_frame=self.output_parent_frame,
                source_frame=self.base_link_frame,
                time=Time(),
            ).transform.translation
        except Exception as e:
            self.get_logger().error(
                f"Failed to lookup transform from world to base link frame: {e}"
                f"Traceback: {traceback.format_exc()}"
            )
            goal_handle.abort()
            return result

        average_transforms = self.order_transforms(tfs, top_3_indices, curr_pos)

        for i, output_child in enumerate(
            [
                self.channel_one_tf_clustered,
                self.channel_two_tf_clustered,
                self.channel_three_tf_clustered,
            ]
        ):
            if i >= len(average_transforms):
                break

            self.publish_transform(
                translation=average_transforms[i][0],
                rotation=average_transforms[i][1],
                latest_time=latest_time,
                output_child=output_child,
            )

        # feedback_msg.current_status = "Clustering complete, static transform published"
        # goal_handle.publish_feedback(feedback_msg)

        # result.clustered_transform = clustered_transform
        # result.total_transforms_collected = len(tfs)
        # result.transforms_in_cluster = len(filtered_tfs)

        goal_handle.succeed()
        return result

    @staticmethod
    def _comparator(origin: Vector3):
        def d(transform: tuple[Vector3, Quaternion]):
            translation = transform[0]
            return (
                ((translation.x - origin.x) ** 2)
                + ((translation.y - origin.y) ** 2)
                + ((translation.z - origin.z) ** 2)
            )

        return d

    @staticmethod
    def _get_position_from_transform(
        tf: TransformStamped,
    ) -> tuple[float, float, float]:
        """Extract position tuple from TransformStamped."""
        t = tf.transform.translation
        return (t.x, t.y, t.z)

    @staticmethod
    def _get_orientation_from_transform(
        tf: TransformStamped,
    ) -> tuple[float, float, float, float]:
        """Extract orientation tuple from TransformStamped."""
        q = tf.transform.rotation
        return (q.x, q.y, q.z, q.w)

    @staticmethod
    def _average_transforms(tfs: list[TransformStamped]) -> tuple[Vector3, Quaternion]:
        """Calculate average translation and orientation from list of transforms."""
        # Average translations
        translations = np.array(
            [SlalomClusterActionServer._get_position_from_transform(tf) for tf in tfs]
        )
        avg_translation = translations.mean(axis=0)

        # Average quaternions using eigenvector method
        try:
            quats = np.array(
                [
                    SlalomClusterActionServer._get_orientation_from_transform(tf)
                    for tf in tfs
                ]
            )
            quat_matrix = np.dot(quats.T, quats)
            eigvals, eigvecs = np.linalg.eigh(quat_matrix)
            avg_quat = eigvecs[
                :, np.argmax(eigvals)
            ]  # eigenvector with largest eigenvalue
        except np.linalg.LinAlgError:
            # Fallback to last quaternion if averaging fails
            avg_quat = SlalomClusterActionServer._get_orientation_from_transform(
                tfs[-1]
            )

        return (
            Vector3(x=avg_translation[0], y=avg_translation[1], z=avg_translation[2]),
            Quaternion(x=avg_quat[0], y=avg_quat[1], z=avg_quat[2], w=avg_quat[3]),
        )

    @staticmethod
    def _tf_to_pose(tf: TransformStamped) -> Pose:
        """Convert TransformStamped to Pose (without covariance or header)."""
        pose = Pose()
        pose.position.x = tf.transform.translation.x
        pose.position.y = tf.transform.translation.y
        pose.position.z = tf.transform.translation.z
        pose.orientation = tf.transform.rotation
        return pose


class TfLruCache:
    def __init__(self, size: int, logger):
        self.size = size

        # idx is the current insertion index (the open spot in the circular buffer)
        self.idx = 0

        self.cache = [None] * self.size
        self.logger = logger

        self.oldest_time = Time()
        self.latest_time = Time()

        self.is_empty_flag = True
        self.count = 0  # number of elements in cache

    @property
    def is_full(self) -> bool:
        return self.count >= self.size

    def _get(self, idx: int) -> TransformStamped:
        return self.cache[idx % self.size]

    def _set(self, tf: TransformStamped):
        self.cache[self.idx] = tf
        self.latest_time = Time.from_msg(tf.header.stamp)
        self.idx = (self.idx + 1) % self.size
        self.count += 1

    def add(self, tf: TransformStamped):
        if self.is_empty_flag:
            self.oldest_time = Time.from_msg(tf.header.stamp)
            self.is_empty_flag = False
            self._set(tf)
            return True

        prev_tf = self._get(self.idx - 1)

        if Time.from_msg(tf.header.stamp) == Time.from_msg(prev_tf.header.stamp):
            self.logger.warn(
                f"Skipping TF with timestamp {tf.header.stamp} as it is the same as the previous one."
            )
            return False

        if self.is_full:
            self.oldest_time = Time.from_msg(self._get(self.idx + 1).header.stamp)

        self._set(tf)
        return True

    def get_oldest_time(self) -> Time:
        return self.oldest_time

    def get_latest_time(self) -> Time:
        return self.latest_time

    def is_empty(self) -> bool:
        return self.is_empty_flag

    def get_all(self) -> tuple[list[TransformStamped], Time]:
        return (
            [self.cache[i] for i in range(self.size) if self.cache[i] is not None],
            self.get_latest_time(),
        )

    def get_count(self) -> int:
        return self.count


def main(args=None):
    rclpy.init(args=args)
    node = SlalomClusterActionServer()
    try:
        rclpy.spin(node, executor=MultiThreadedExecutor())
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
