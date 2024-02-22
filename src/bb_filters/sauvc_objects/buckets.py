from bb_msgs.msg import DetectedObject, DetectedObjects
from bb_filters import filter
import numpy as np
import rospy
from circle_fit import taubinSVD


class Filter(filter.Filter):
    def __init__(self, config, camera_infos: filter.CameraInfos):
        super(Filter, self).__init__(config, camera_infos)
        self.__name__ = "buckets_filter"
        self.blue_bucket_idx = 1  # 0 for left most, 3 for right most
        self.bucket_depth = 2.0
        self.bucket_height = 0.3
        self.bucket_diameter = 0.6

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
                det.real_dims = self.bucket_diameter, self.bucket_diameter, self.bucket_height
                bucket.world_coords[2] = self.bucket_depth
                bucket.name = "bucket"
                detections.detected.append(bucket)

        if len(bot_buckets) == 0:
            return detections
        camera_depth = self.camera_infos.get_camera_z(289, bot_buckets[0].header.stamp)
        est_circle_radius = np.abs(
            (self.bucket_diameter / 2)
            / (camera_depth - (self.bucket_depth - self.bucket_height))
            * self.camera_infos.get_info(289).P[0]
        )
        for det in bot_buckets:
            xc, yc, r, sigma = taubinSVD(np.array(det.contour).reshape(-1, 2))
            if np.abs(r - est_circle_radius) < 50:
                det.centre_x, det.centre_y = max(0, int(xc)), max(0, int(yc))

                det = self.camera_infos.compute_3d_coords_from_depth(det, self.bucket_depth - self.bucket_height)
                det.world_coords[2] += self.bucket_height
                det.real_dims = self.bucket_diameter, self.bucket_diameter, self.bucket_height
                det.name = "bucket"
                detections.detected.append(det)
            else:
                rospy.loginfo(f"Bucket not round enough: {xc}, {yc}, {r}, {sigma}")
        return detections
