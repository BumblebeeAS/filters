#!/usr/bin/env python3
import traceback

import tf2_ros
from bb_filters.tf_lru_cache import TfLruCache
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from rclpy.time import Time


class TFLookUpSrvNode(Node):
    def __init__(self, name):
        super().__init__(name)

        self.output_parent_frame = (
            self.declare_parameter(name="output_parent_frame", value="world_ned")
            .get_parameter_value()
            .string_value
        )
        self.base_link_frame = (
            self.declare_parameter(name="base_link_frame", value="auv4/base_link_ned")
            .get_parameter_value()
            .string_value
        )
        cache_size = (
            self.declare_parameter(name="cache_size", value=10000)
            .get_parameter_value()
            .integer_value
        )

        # TF components
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=False
        )

        self.timer = self.create_timer(0.01, self.collect_tfs)
        self.pose_array_publisher_all = self.create_publisher(
            PoseArray, "/auv4/cluster_multi_srv/poses", 10
        )

        self.cache = TfLruCache(size=cache_size, logger=self.get_logger())
        self.enabled = False
        self.start_time = None
        self.num_duplicated_tfs = 0
        self.num_old_tfs = 0
        self.tf_list_in = []

    def collect_tfs(self):
        if not self.enabled or self.start_time is None:
            return

        for input_child in self.tf_list_in:
            try:
                tf = self.tf_buffer.lookup_transform(
                    target_frame=self.output_parent_frame,
                    source_frame=input_child,
                    time=Time(),
                )
                success, is_duplicated, is_old = self.cache.add(tf, self.start_time)

                self.num_old_tfs += int(is_old)
                self.num_duplicated_tfs += int(is_duplicated)

            except Exception as e:
                pass
                # self.get_logger().warn(
                #     f"Failed to lookup transform for {input_child}: {e}"
                # )
                # self.get_logger().warn(f"Traceback: {traceback.format_exc()}")
