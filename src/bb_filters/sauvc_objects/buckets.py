from bb_msgs.msg import DetectedObject, DetectedObjects
from bb_filters import filter
import numpy as np
import rospy
class Filter(filter.Filter):
    def __init__(self, config, camera_infos: filter.CameraInfos):
        super(Filter, self).__init__(config, camera_infos)
        self.__name__ = "buckets_filter"
        self.blue_bucket_idx = 1 # 0 for left most, 3 for right most

    def process(self, bboxes: DetectedObjects) -> DetectedObjects:
        detections = DetectedObjects()
        red_bucket = [x for x in bboxes.detected if x.name=="red_bucket"]
        blue_bucket = [x for x in bboxes.detected if x.name=="blue_bucket"]

        return detections
