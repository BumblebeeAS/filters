#!/usr/bin/env python3
import sys, traceback
import rospy
import numpy as np
from geometry_msgs.msg import Quaternion, TransformStamped
import tf2_ros
from bb_msgs.msg import DetectedObjects
from transforms3d.euler import euler2quat
from collections import defaultdict
from bb_msgs.srv import EstimatorToggle, EstimatorToggleRequest, EstimatorToggleResponse

class CentroidTFPublisher:
    def __init__(self):
        self.node_name = "centroid_tf_publisher"
        self.rate = 10.0  # Hz
        self.queue_size = 10
        self.object_pose_topic = (
            "vision/external/detected_filtered"  # Replace with actual topic name
        )
        self.centroid_output_topic = (
            "vision/detected_centroid"
        )
        self.tf_topic = "/centroid_tf"  # Replace with desired TF topic
        self.window_size = defaultdict(lambda: 30)
        self.positions = defaultdict(list)
        self.object_yaws = defaultdict(float)
        self.br = tf2_ros.TransformBroadcaster()
        self.disabled = set()
        self.debug_publishers = {}
        self.detections = {}
        self.centroid_det_pub = rospy.Publisher(
            self.centroid_output_topic, DetectedObjects)
        self.latest = {}

        rospy.init_node(self.node_name)

        self.toggle_srv = rospy.Service("toggle_centroid_filter", EstimatorToggle, self.toggle_cb)

        self.dets_sub = rospy.Subscriber(
            self.object_pose_topic,
            DetectedObjects,
            self.dets_callback,
            queue_size=self.queue_size,
        )
        self.tf_pub = rospy.Publisher(
            self.tf_topic, TransformStamped, queue_size=self.queue_size
        )
        self.timer = rospy.Timer(rospy.Duration.from_sec(1/self.rate), self.publish_centroid_tf)

        self.fallback_tfs = {
            ""
        }

        rospy.loginfo(
            f"Node '{self.node_name}' started. Subscribing to '{self.object_pose_topic}' and publishing TF on '{self.tf_topic}'."
        )

        self.spin()

    def toggle_cb(self, srv: EstimatorToggleRequest):
        rospy.logwarn(f"Toggling object {srv.object_name} {srv.enabled} {srv.reset}")
        if srv.object_name in self.disabled and srv.enabled:
            self.disabled.remove(srv.object_name)
        if not srv.object_name in self.disabled and not srv.enabled:
            self.disabled.add(srv.object_name)
        
        if srv.reset:
            self.positions[srv.object_name] = []
        if srv.window_size > 0:
            self.window_size[srv.object_name] = srv.window_size

        if srv.object_name in self.latest:
            _, stddev, num_estimates = self.latest[srv.object_name]
        else:
            num_estimates = 0
            stddev = 10000

        return EstimatorToggleResponse(
            srv.enabled,
            num_estimates,
            stddev,
            ""
        )

    def dets_callback(self, dets):
        for det in dets.detected:
            if det.name in self.disabled:
                continue
            self.positions[det.name].append(det.world_coords)
            self.object_yaws[det.name] = det.world_yaw
            if len(self.positions[det.name]) > self.window_size[det.name]:
                self.positions[det.name].pop(0)  # Maintain a fixed-size window
            self.detections[det.name] = det

    @staticmethod
    def centroidnp(arr):
        if len(arr) == 0:
            return None
        length, dim = arr.shape
        norm = length * (length + 1)/ 2
        return (np.array([np.average(arr[:, i], weights = [j/norm for j in range(1, length + 1)]) for i in range(dim)]),
                np.linalg.norm(arr.std(axis=0)),
                len(arr))
    
    @staticmethod
    def dbscan_cluster(arr, eps=0.25, min_samples=5):
        """Returns largest cluster"""
        from sklearn.cluster import DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = dbscan.fit_predict(arr)
        #  return max cluster center
        cluster_centers = []
        for label in np.unique(labels):
            if label == -1:
                continue
            cluster_points = arr[labels == label]
            cluster_center = cluster_points.mean(axis=0)
            cluster_centers.append((cluster_center, np.mean(np.var(cluster_points, axis=0)), len(cluster_points)))
        if len(cluster_centers) == 0:
            return None
        return max(cluster_centers, key=lambda x: x[1])

    def publish_centroid_tf(self, event):
        output = DetectedObjects()
        for name, positions in self.positions.items():
            if len(positions) == 0:
                continue
            try:
                p = np.array(positions)
                centroid = CentroidTFPublisher.dbscan_cluster(p)
                if centroid is None:
                    continue
                self.latest[name] = centroid

                # Create TransformStamped message
                tf_msg = TransformStamped()
                tf_msg.header.stamp = rospy.Time.now()
                tf_msg.header.frame_id = "map_ned"  # Assuming the map frame as reference
                tf_msg.child_frame_id = (
                    f"{name}/centroid_ned"  # Replace with desired TF frame ID
                )
                tf_msg.transform.translation.x = centroid[0][0]
                tf_msg.transform.translation.y = centroid[0][1]
                tf_msg.transform.translation.z = centroid[0][2]

                w, x, y, z = euler2quat(0, 0, np.deg2rad(self.object_yaws[name]))
                tf_msg.transform.rotation = Quaternion(x, y, z, w)

                self.tf_pub.publish(tf_msg)
                self.br.sendTransform(tf_msg)
                det = self.detections[name]
                det.world_coords = [*centroid[0]]
                det.extra = (*det.extra, centroid[1], centroid[2]) #err, cluster size
                output.detected.append(det)
            except Exception as e:
                rospy.logerr(f"Error publishing centroid for {name}: {e}")
                traceback.print_exc(file=sys.stdout)
        self.centroid_det_pub.publish(output)
    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        CentroidTFPublisher()
    except rospy.ROSInterruptException:
        pass
