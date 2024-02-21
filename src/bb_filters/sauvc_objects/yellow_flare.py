from bb_msgs.msg import DetectedObject, DetectedObjects
from bb_filters import filter
import numpy as np
import rospy
class Filter(filter.Filter):
    def __init__(self, config, camera_infos: filter.CameraInfos):
        super(Filter, self).__init__(config, camera_infos)
        self.__name__ = "yellow_flare_filter"
        self.flare_height = 0.8
        self.flare_width = 0.02

    def process(self, bboxes: DetectedObjects) -> DetectedObjects:
        detections = DetectedObjects()
        yellow_flares = [x for x in bboxes.detected if x.name=="yellow_flare"]
        if len(yellow_flares) == 0:
            return detections

        # filter by rectangularity?
        det = max(yellow_flares, key=lambda x: x.extra[0]) # get flare with highest confidence or height?
        if det.bbox_width > det.bbox_height: # filter case where flare toppled
            return detections
        img_height = self.camera_infos.get_info(det.source).height

        is_top_visible = det.centre_y - det.bbox_height / 2 > 30
        is_bottom_visible = (
            det.centre_y + det.bbox_height / 2 < img_height - 30
        )
        if is_top_visible and is_bottom_visible:
            distance = (
                self.flare_height
                * self.camera_infos.get_info(det.source).P[5]
                / (det.bbox_height)
            )
            det = self.camera_infos.compute_3d_coords_from_distance(
                det, distance
            )
        else:
            distance = (
                self.flare_width
                * self.camera_infos.get_info(det.source).P[0]
                / (det.bbox_width)
            )
            det = self.camera_infos.compute_3d_coords_from_distance(
                det, distance
            )
        det.real_dims = [self.flare_width, self.flare_width, self.flare_height]
        det.name = "yellow_flare"
        detections.detected.append(det)
        return detections
