from bb_msgs.msg import DetectedObject, DetectedObjects
from bb_filters import filter
import numpy as np
import rospy
import copy
from circle_fit import taubinSVD
from sklearn.cluster import KMeans, DBSCAN
from sensor_msgs.msg import Image
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from collections import deque

from cv_bridge import CvBridge
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
        self.blue_bucket_idx = config["blue_bucket_idx"]  # 0 for left most, 3 for right most
        print("blue_bucket: ", self.blue_bucket_idx)
        self.bucket_depth = 2.0
        # self.bucket_height = 0.3
        # self.bucket_diameter = 0.6
        self.bucket_height = 0.25
        self.bucket_diameter = 0.4
        self.min_dist_between_buckets = 0.5

        self.cv_bridge = CvBridge()
        self.buckets_pub = rospy.Publisher("/buckets_scatter", Image, queue_size=1)
        # self.kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto")
        self.kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
        self.dbscan = DBSCAN(eps=0.5, min_samples = 5, metric='euclidean')
        self.points = deque(maxlen=100)
        self.blue_bucket_points = deque(maxlen=100)
        self.sort_by_x = False # true if buckets have similar y-coords when testing
        self.determine_cluster_by_blue_bucket = True

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
                bucket.real_dims = self.bucket_diameter, self.bucket_diameter, self.bucket_height
                bucket.world_coords[2] = self.bucket_depth
                detections.detected.append(copy.deepcopy(bucket))
                bucket.name = "bucket"
                detections.detected.append(copy.deepcopy(bucket))

        if len(bot_buckets) > 0:
            camera_depth = self.camera_infos.get_camera_z(289, bot_buckets[0].header.stamp)
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

                    bucket = self.camera_infos.compute_3d_coords_from_depth(det, self.bucket_depth - self.bucket_height)
                    if bucket is None:
                        rospy.logwarn("Failed to compute bucket coord")
                        continue
                    bucket.world_coords[2] += self.bucket_height
                    bucket.real_dims = self.bucket_diameter, self.bucket_diameter, self.bucket_height
                    detections.detected.append(copy.deepcopy(bucket))
                    bucket.name = "bucket"
                    detections.detected.append(copy.deepcopy(bucket))
                else:
                    rospy.loginfo_throttle(1.0, f"Bucket radius rejected: ({xc}, {yc}), {r} expect {est_circle_radius}")

        new_points = []
        for det in detections.detected:
            new_points.append((det.world_coords[0], det.world_coords[1]))
            if det.name == "blue_bucket":
                self.blue_bucket_points.append((det.world_coords[0], det.world_coords[1]))

        if len(new_points) == 0:
            return detections
        self.points.extend(new_points)

        points = np.array(self.points)
        plt.scatter(points[:, 0], points[:, 1])
        fig = plt.figure()
        ax = fig.add_subplot(111)

        if len(self.points) > 10:
            labels=self.dbscan.fit_predict(points)
            cluster_centers = []
            for label in np.unique(labels):
                if label == -1:
                    continue
                cluster_points = points[labels == label]
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1])

                cluster_center = cluster_points.mean(axis=0)
                cluster_centers.append((cluster_center, len(cluster_points), np.mean(np.var(cluster_points, axis=0))))
            if self.sort_by_x:
                id_centers = [x[0] for x in sorted(
                    enumerate(cluster_centers), key=lambda x: x[1][0][0])]
            else:
                id_centers = [x[0] for x in sorted(
                    enumerate(cluster_centers), key=lambda x: x[1][0][1])]

            ids = {label: k for k, label in enumerate(id_centers)}


            offset = 0
            if self.determine_cluster_by_blue_bucket:
                blue_bucket_points = np.array(self.blue_bucket_points)
                blue_bucket_labels = self.dbscan.fit_predict(blue_bucket_points)
                blue_bucket_cluster_centers = []
                rospy.loginfo(f"Blue bucket labels: {np.unique(blue_bucket_labels)}")
                for label in np.unique(blue_bucket_labels):
                    if label == -1:
                        continue
                    cluster_points = blue_bucket_points[blue_bucket_labels == label]
                    cluster_center = cluster_points.mean(axis=0)
                    blue_bucket_cluster_centers.append((cluster_center, len(cluster_points), np.mean(np.var(cluster_points, axis=0)))
                    )
                if len(blue_bucket_cluster_centers)>0:
                    blue_bucket_centroid = np.array(max(blue_bucket_cluster_centers, key=lambda x: x[1])[0])


                    dists = [
                        np.linalg.norm(
                            np.array([cluster_centers[idx][0][0], cluster_centers[idx][0][1]]) - blue_bucket_centroid
                        ) for idx in ids.values()]
                    # rospy.loginfo(f"Dists: {dists}, blue centroid: {blue_bucket_centroid}, cluster_centers: {                        [np.array([cluster_centers[idx][0][0], cluster_centers[idx][0][1]]) for idx in ids.values()]}")
                    offset = np.argmin(dists) - self.blue_bucket_idx
        
            if offset!=0:
                rospy.loginfo_throttle(1, f"Offset from blue bucket: {offset}")
            
            for i, idx in enumerate([ids[label] for label in labels[-len(new_points):]]):
                new_det = detections.detected[i]
                new_det.name += f"_{idx + offset}"
                detections.detected.append(new_det)
        # with open("buckets.txt", "w") as f:
        #     for det in detections.detected:
        #         f.write(f"{det.name} {det.world_coords[0]} {det.world_coords[1]}\n")

        img = fig2img(fig)
        self.buckets_pub.publish(self.cv_bridge.cv2_to_imgmsg(np.array(img)))
        return detections
