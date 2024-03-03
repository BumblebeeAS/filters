from bb_msgs.msg import DetectedObject, DetectedObjects
from bb_filters import filter
import numpy as npfilter
import rospy
class Filter(filter.Filter):
    def __init__(self, config, camera_infos: filter.CameraInfos):
        super(Filter, self).__init__(config, camera_infos)
        self.__name__ = "orange_flare_filter"
        self.estimate_x, self.estimate_y, self.estimate_z, self.estimate_yaw = self.camera_infos.get_object_pos("gate/estimate_base_link")

        self.flare_height = 1.45
        self.flare_width = 0.12
        self.latest_pos = None

    def process(self, bboxes: DetectedObjects) -> DetectedObjects:
        detections = DetectedObjects()
        orange_flares = [x for x in bboxes.detected if (x.name=="orange_flare" or x.name=="qualification_gate_side") and x.source == 288 and\
                        #  (filter.get_aspect_ratio(x) < 0.2) and\
                        #  x.bbox_width < x.image_height * 0.3 and\
                         x.color < 40]
        if len(orange_flares) == 0:
            if self.latest_pos is None:
                det = DetectedObject()
                det.world_coords = [self.estimate_x, self.estimate_y, self.estimate_z]
                det.real_dims = [0.2, 0.2, 1.45]
                det.name = "orange_flare"
                detections.detected.append(det)
            return detections

        # filter by rectangularity?
        det = max(orange_flares, key=lambda x: x.extra[0]) # get flare with highest confidence or height?
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
        det.real_dims = [0.2, 0.2, 1.45]
        det.name = "orange_flare"
        detections.detected.append(det)
        self.latest_pos = det.world_coords
        return detections
