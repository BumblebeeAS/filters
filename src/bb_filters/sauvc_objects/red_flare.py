from bb_msgs.msg import DetectedObject, DetectedObjects
from bb_filters import filter
import numpy as np
import rospy


class Filter(filter.Filter):
    def __init__(self, config, camera_infos: filter.CameraInfos):
        super(Filter, self).__init__(config, camera_infos)
        self.__name__ = "red_flare_filter"
        self.estimate_x, self.estimate_y, self.estimate_z, self.estimate_yaw = self.camera_infos.get_object_pos("red_flare/estimate_base_link")
        self.flare_height = config["flare_height"]
        self.flare_width = config["flare_width"]
        self.flare_yaw = self.estimate_yaw
        self.estimate_pos = self.estimate_x, self.estimate_y


    def process(self, bboxes: DetectedObjects) -> DetectedObjects:
        detections = DetectedObjects()
        red_flares = [
            x for x in bboxes.detected if x.name == "red_flare" and x.source == 288
        ]
        if len(red_flares) == 0:
            return detections

        # filter by rectangularity?
        det = max(
            red_flares, key=lambda x: x.extra[0]
        )  # get flare with highest confidence or height?
        if det.bbox_width > det.bbox_height:  # filter case where flare toppled
            # rospy.logwarn_throttle(1, "red flare not upright")
            return detections
        img_height = self.camera_infos.get_info(det.source).height

        is_top_visible = det.centre_y - det.bbox_height / 2 > 30
        is_bottom_visible = det.centre_y + det.bbox_height / 2 < img_height - 30
        if is_top_visible and is_bottom_visible:
            distance = (
                self.flare_height
                * self.camera_infos.get_info(det.source).P[5]
                / (det.bbox_height)
            )
            det = self.camera_infos.compute_3d_coords_from_distance(det, distance)
        else:
            distance = (
                self.flare_width
                * self.camera_infos.get_info(det.source).P[0]
                / (det.bbox_width)
            )
            det = self.camera_infos.compute_3d_coords_from_distance(det, distance)
        det.real_dims = [self.flare_width, self.flare_width, self.flare_height]
        det.world_yaw = self.flare_yaw * 180 / np.pi
        det.name = "red_flare"
        # if np.abs(det.world_coords[0] - self.estimate_pos[0]) > 1.5:
        #     rospy.logwarn_throttle(1, "red flare det far from estimate")
        #     return detections
        detections.detected.append(det)
        return detections
