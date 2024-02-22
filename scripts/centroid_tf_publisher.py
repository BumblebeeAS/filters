#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Quaternion, TransformStamped
import tf2_ros
from bb_msgs.msg import DetectedObjects
from transforms3d.euler import euler2quat
from collections import defaultdict


class CentroidTFPublisher:
    def __init__(self):
        self.node_name = "centroid_tf_publisher"
        self.rate = 10.0  # Hz
        self.queue_size = 10
        self.object_pose_topic = (
            "/auv4/vision/external/detected_filtered"  # Replace with actual topic name
        )
        self.tf_topic = "/centroid_tf"  # Replace with desired TF topic
        self.accumulation_window = 30
        self.positions = defaultdict(list)
        self.object_yaws = defaultdict(float)
        self.br = tf2_ros.TransformBroadcaster()

        rospy.init_node(self.node_name)

        self.dets_sub = rospy.Subscriber(
            self.object_pose_topic,
            DetectedObjects,
            self.dets_callback,
            queue_size=self.queue_size,
        )
        self.tf_pub = rospy.Publisher(
            self.tf_topic, TransformStamped, queue_size=self.queue_size
        )

        rospy.loginfo(
            f"Node '{self.node_name}' started. Subscribing to '{self.object_pose_topic}' and publishing TF on '{self.tf_topic}'."
        )

        self.spin()

    def dets_callback(self, dets):
        for det in dets.detected:
            self.positions[det.name].append(det.world_coords)
            self.object_yaws[det.name] = det.world_yaw
            if len(self.positions[det.name]) > self.accumulation_window:
                self.positions[det.name].pop(0)  # Maintain a fixed-size window
        self.publish_centroid_tf()

    def publish_centroid_tf(self):
        for name, positions in self.positions.items():
            total_x, total_y, total_z = 0.0, 0.0, 0.0
            for pose in positions:
                total_x += pose[0]
                total_y += pose[1]
                total_z += pose[2]

            centroid_x = total_x / len(positions)
            centroid_y = total_y / len(positions)
            centroid_z = total_z / len(positions)

            # Create TransformStamped message
            tf_msg = TransformStamped()
            tf_msg.header.stamp = rospy.Time.now()
            tf_msg.header.frame_id = "map_ned"  # Assuming the map frame as reference
            tf_msg.child_frame_id = (
                f"{name}/centroid_ned"  # Replace with desired TF frame ID
            )
            tf_msg.transform.translation.x = centroid_x
            tf_msg.transform.translation.y = centroid_y
            tf_msg.transform.translation.z = centroid_z

            w, x, y, z = euler2quat(0, 0, np.deg2rad(self.object_yaws[name]))
            tf_msg.transform.rotation = Quaternion(x, y, z, w)

            self.tf_pub.publish(tf_msg)
            self.br.sendTransform(tf_msg)

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        CentroidTFPublisher()
    except rospy.ROSInterruptException:
        pass
