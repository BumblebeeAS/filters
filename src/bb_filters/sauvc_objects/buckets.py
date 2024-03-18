from bb_msgs.msg import DetectedObject, DetectedObjects
from bb_filters import filter
import numpy as np
import rospy
import copy
import tf2_ros
from nav_msgs.msg import Odometry
from circle_fit import taubinSVD
from sklearn.cluster import KMeans, DBSCAN
from sensor_msgs.msg import Image
from geometry_msgs.msg import Quaternion, TransformStamped
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from collections import deque
from bb_msgs.msg import Ping
from std_msgs.msg import Int16
import math

from cv_bridge import CvBridge
import matplotlib as mpl

mpl.use("Agg")

np.set_printoptions(suppress=True)
def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = PILImage.open(buf)
    return img


class Filter(filter.Filter):
    def __init__(self, config, camera_infos: filter.CameraInfos):
        super(Filter, self).__init__(config, camera_infos)
        self.__name__ = "buckets_filter"
        self.blue_bucket_idx = config[
            "blue_bucket_idx"
        ]  # 0 for left most, 3 for right most
        print("blue_bucket: ", self.blue_bucket_idx)

        self.flip_colours = False
        self.bucket_depth = 2.0
        self.bucket_height = 0.3
        self.bucket_diameter = 0.6
        # self.bucket_height = 0.25
        # self.bucket_diameter = 0.4
        self.min_dist_between_buckets = 1.0
        self.num_buckets = config["num_buckets"]
        self.vehicle_depth = 0
        self.acoustics_vecs = deque(maxlen=5)

        self.cv_bridge = CvBridge()
        self.buckets_pub = rospy.Publisher("/buckets_scatter", Image, queue_size=1)
        # self.kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto")
        self.dbscan = DBSCAN(eps=0.5, min_samples=5, metric="euclidean")
        self.points = deque(maxlen=100)
        self.blue_bucket_points = deque(maxlen=100)
        self.sort_by_x = False  # true if buckets have similar y-coords when testing
        self.determine_cluster_by_blue_bucket = True
        self.pinger_idx_pub = rospy.Publisher("/pinger_bucket_idx", Int16, queue_size=1)
        self.depth_sub = rospy.Subscriber(
            "/auv4/nav/map_odom_ned", Odometry, self.update_depth
        )
        self.ping_sub = rospy.Subscriber(
            "/auv4/acoustics/ping", Ping, self.process_ping
        )
        self.br = tf2_ros.TransformBroadcaster()
        self.tf_topic = "baselink_to_pinger_tf"
        self.tf_pub = rospy.Publisher(self.tf_topic, TransformStamped, queue_size=1)
        self.average_pinger_pos = None
        self.best_idx = -1

    def update_depth(self, msg):
        self.vehicle_depth = msg.pose.pose.position.z

    def process_ping(self, msg):
        doa = msg.doa
        elevation = msg.elevation
        depth_from_pinger = self.bucket_depth - self.vehicle_depth
        rospy.loginfo(f"ping mag: {depth_from_pinger / (math.tan(np.deg2rad(elevation)) + 0.1)}")
        magnitude = min(np.abs(depth_from_pinger / (math.tan(np.deg2rad(elevation)) + 0.1)), 5)

        vec = self.camera_infos.compute_object_ray_from_bearing(rospy.Time.now(), doa, elevation)

        vec[3:] += magnitude * vec[:3]
        rospy.loginfo(f"Acoustics vec: {vec}, magnitude: {magnitude}, doa: {doa}")
        self.acoustics_vecs.append(vec)
        A = np.stack(self.acoustics_vecs)
        estimate_pinger_pos = A[:, 3:5]
        if len(estimate_pinger_pos) > 3:
            self.average_pinger_pos = np.mean(estimate_pinger_pos, axis=0)
            tf_msg = TransformStamped()
            tf_msg.header.stamp = rospy.Time.now()
            tf_msg.header.frame_id = "map_ned"
            tf_msg.child_frame_id = "pinger_elevation_estimate_ned"
            tf_msg.transform.translation.x = self.average_pinger_pos[0]
            tf_msg.transform.translation.y = self.average_pinger_pos[1]
            tf_msg.transform.translation.z = self.bucket_depth
            tf_msg.transform.rotation = Quaternion(0, 0, 0, 1)
            self.tf_pub.publish(tf_msg)
            self.br.sendTransform(tf_msg)

    def process(self, bboxes: DetectedObjects) -> DetectedObjects:
        detections = DetectedObjects()
        front_buckets = [
            x
            for x in bboxes.detected
            if x.name in ["red_bucket", "blue_bucket"] and x.source == 288
        ]
        bot_buckets = [
            x
            for x in bboxes.detected
            if x.name in ["red_bucket", "blue_bucket"] and x.source == 289
        ]
        front_img_height = self.camera_infos.get_info(288).height
        for det in front_buckets:
            is_top_visible = det.centre_y - det.bbox_height / 2 > 30
            is_bottom_visible = det.centre_y + det.bbox_height / 2 < front_img_height
            if is_top_visible and is_bottom_visible:
                bucket = self.camera_infos.compute_3d_coords_from_depth(
                    det, self.bucket_depth - self.bucket_height / 2
                )
                if bucket is None:
                    continue
                bucket.real_dims = (
                    self.bucket_diameter,
                    self.bucket_diameter,
                    self.bucket_height,
                )
                bucket.world_coords[2] = self.bucket_depth
                detections.detected.append(copy.deepcopy(bucket))
                if self.flip_colours:
                    detections.detected[-1].name = (
                        detections.detected[-1]
                        .name.replace("red", "green")
                        .replace("blue", "red")
                        .replace("green", "blue")
                    )
                    rospy.loginfo(detections.detected[-1].name)
                bucket.name = "bucket"
                detections.detected.append(copy.deepcopy(bucket))

        if len(bot_buckets) > 0:
            camera_depth = self.camera_infos.get_camera_z(
                289, bot_buckets[0].header.stamp
            )
            est_circle_radius = np.abs(
                (self.bucket_diameter / 2)
                / (camera_depth - (self.bucket_depth - self.bucket_height))
                * self.camera_infos.get_info(289).P[0]
            )
            for det in bot_buckets:
                try:
                    xc, yc, r, sigma = taubinSVD(np.array(det.contour).reshape(-1, 2))
                except:
                    continue
                if np.abs(r - est_circle_radius) < 100:
                    det.centre_x, det.centre_y = max(0, int(xc)), max(0, int(yc))

                    bucket = self.camera_infos.compute_3d_coords_from_depth(
                        det, self.bucket_depth - self.bucket_height
                    )
                    if bucket is None:
                        rospy.logwarn("Failed to compute bucket coord")
                        continue
                    bucket.world_coords[2] += self.bucket_height
                    bucket.real_dims = (
                        self.bucket_diameter,
                        self.bucket_diameter,
                        self.bucket_height,
                    )
                    detections.detected.append(copy.deepcopy(bucket))
                    if self.flip_colours:
                        detections.detected[-1].name = (
                            detections.detected[-1]
                            .name.replace("red", "green")
                            .replace("blue", "red")
                            .replace("green", "blue")
                        )
                        rospy.loginfo(detections.detected[-1].name)
                    bucket.name = "bucket"
                    detections.detected.append(copy.deepcopy(bucket))
                else:
                    rospy.loginfo_throttle(
                        1.0,
                        f"Bucket radius rejected: ({xc}, {yc}), {r} expect {est_circle_radius}",
                    )

        new_points = []
        for det in detections.detected:
            new_points.append((det.world_coords[0], det.world_coords[1]))
            n = "red_bucket" if self.flip_colours else "blue_bucket"
            if det.name == n:
                self.blue_bucket_points.append(
                    (det.world_coords[0], det.world_coords[1])
                )

        if len(new_points) == 0:
            return detections
        self.points.extend(new_points)

        points = np.array(self.points)
        plt.scatter(points[:, 0], points[:, 1])
        fig = plt.figure()
        ax = fig.add_subplot(111)

        if len(self.points) > 10:
            labels = self.dbscan.fit_predict(points)
            cluster_centers = []
            for label in np.unique(labels):
                if label == -1:
                    continue
                cluster_points = points[labels == label]
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1])

                cluster_center = cluster_points.mean(axis=0)
                cluster_centers.append(
                    (
                        cluster_center,
                        len(cluster_points),
                        np.mean(np.var(cluster_points, axis=0)),
                    )
                )
            if self.sort_by_x:
                id_centers = [
                    x[0]
                    for x in sorted(
                        enumerate(cluster_centers), key=lambda x: x[1][0][0]
                    )
                ]
            else:
                id_centers = [
                    x[0]
                    for x in sorted(
                        enumerate(cluster_centers), key=lambda x: x[1][0][1]
                    )
                ]

            ids = {label: k for k, label in enumerate(id_centers)}

            offset = 0
            if (
                self.determine_cluster_by_blue_bucket
                and len(self.blue_bucket_points) > 10
            ):
                blue_bucket_points = np.array(self.blue_bucket_points)
                blue_bucket_labels = self.dbscan.fit_predict(blue_bucket_points)
                blue_bucket_cluster_centers = []
                for label in np.unique(blue_bucket_labels):
                    if label == -1:
                        continue
                    cluster_points = blue_bucket_points[blue_bucket_labels == label]
                    cluster_center = cluster_points.mean(axis=0)
                    blue_bucket_cluster_centers.append(
                        (
                            cluster_center,
                            len(cluster_points),
                            np.mean(np.var(cluster_points, axis=0)),
                        )
                    )
                if len(blue_bucket_cluster_centers) > 0:
                    blue_bucket_centroid = np.array(
                        max(blue_bucket_cluster_centers, key=lambda x: x[1])[0]
                    )

                    dists = [
                        np.linalg.norm(
                            np.array(
                                [cluster_centers[idx][0][0], cluster_centers[idx][0][1]]
                            )
                            - blue_bucket_centroid
                        )
                        for idx in ids.keys()
                    ]
                    # rospy.loginfo(f"Dists: {dists}, blue centroid: {blue_bucket_centroid}, cluster_centers: {[np.array([cluster_centers[idx][0][0], cluster_centers[idx][0][1]]) for idx in ids.values()]}")
                    offset = self.blue_bucket_idx - np.argmin(dists)

                    rospy.loginfo_throttle(
                        1, f"Offset from blue bucket: {offset}, {np.argmin(dists)}"
                    )
            # rospy.loginfo(f"num valid clusters: {len(id_centers) - offset}, vecs: {self.acoustics_vecs}")
            # if len(id_centers) - offset >= self.num_buckets:
            best_cluster, best_distance = None, 10000
            if len(self.acoustics_vecs) > 1:
                A = np.stack(self.acoustics_vecs)
                v = A[:, :2]
                x = A[:, 3:5]
                for i, center in enumerate(cluster_centers):
                    v1 = x - np.array(center[0])[:2]

                    dist = np.mean(
                        np.linalg.norm(
                            v1 - np.diag(v1 @ v.T).reshape(-1, 1) * v, axis=1
                        )
                    )
                    if dist < best_distance:
                        best_cluster = center[0]
                        best_distance = dist
            if best_cluster is not None:
                dists = [
                    np.linalg.norm(
                        np.array(
                            [cluster_centers[idx][0][0], cluster_centers[idx][0][1]]
                        )
                        - best_cluster[:2]
                    )
                    for idx in ids.keys()
                ]
                rospy.loginfo(f"Dists: {dists}, cluster_centers: {[np.array([cluster_centers[idx][0][0], cluster_centers[idx][0][1]]) for idx in ids.values()]}")

                self.best_idx = np.argmin(dists) + offset
                if self.best_idx <= 3 and self.best_idx >= 0:
                    rospy.loginfo_throttle(
                        1, f"pinger est: {self.best_idx}, dists: {dists}"
                    )
                    pinger_idx_message = Int16(self.best_idx)
                    self.pinger_idx_pub.publish(pinger_idx_message)

                else:
                    rospy.logwarn(f"pinger best_idx invalid: {self.best_idx}")
            # else:
            #     rospy.loginfo("insufficient buckets")
            for i, idx in enumerate(
                [ids[label] for label in labels[-len(new_points) :] if label >= 0]
            ):
                if idx + offset > self.num_buckets - 1 or idx + offset < 0:
                    continue
                new_det = copy.deepcopy(detections.detected[i])
                new_det.name = f"bucket_{idx + offset}"
                detections.detected.append(new_det)

                if self.best_idx != -1 and idx + offset == self.best_idx:
                    new_det = copy.deepcopy(new_det)
                    new_det.name = f"bucket_pinger"
                    detections.detected.append(new_det)
        # with open("buckets.txt", "w") as f:
        #     for det in detections.detected:
        #         f.write(f"{det.name} {det.world_coords[0]} {det.world_coords[1]}\n")

        img = fig2img(fig)
        self.buckets_pub.publish(self.cv_bridge.cv2_to_imgmsg(np.array(img)))
        return detections

