from bb_msgs.msg import DetectedObject, DetectedObjects
from bb_filters import filter
import numpy as np
import rospy
class Filter(filter.Filter):
    def __init__(self, config, camera_infos: filter.CameraInfos):
        super(Filter, self).__init__(config, camera_infos)
        self.__name__ = "quallification_gate_filter"
        self.gate_orientation = np.pi/2
        self.gate_width = 1.5
        self.gate_height = 1.0
        self.R = np.array([
            [np.cos(self.gate_orientation), -np.sin(self.gate_orientation)],
            [np.sin(self.gate_orientation), np.cos(self.gate_orientation)]
        ])

    def process(self, bboxes: DetectedObjects) -> DetectedObjects:
        detections = DetectedObjects()
        gate_sides = [x for x in bboxes.detected if x.name=="qualification_gate_side"]
        gate = [x for x in bboxes.detected if x.name=="qualification_gate"]
        if len(gate_sides) != 2 and len(gate) != 1:
            return detections
        det = gate_sides[0] if len(gate_sides) > 0 else gate[0]

        camera_yaw = self.camera_infos.get_camera_yaw(det.source, det.header.stamp)
        if camera_yaw is None:
            rospy.logerr("get_camera_yaw failed, possibly due to vehicle tilt")
            return detections
        gate_sides = sorted(gate_sides, key=lambda x: x.centre_x)
        if len(gate_sides) != 2:
            x1, x2 = gate[0].centre_x - gate[0].bbox_width/2, gate[0].centre_x + gate[0].bbox_width/2
            y1, y2 = gate[0].centre_y - gate[0].bbox_height/2, gate[0].centre_y + gate[0].bbox_height/2
        else:
            x1, x2 = gate_sides[0].centre_x, gate_sides[1].centre_x
            y1 = min(gate_sides[0].centre_y - gate_sides[0].bbox_height, gate_sides[1].centre_y - gate_sides[1].bbox_height)
            y2 = max(gate_sides[0].centre_y + gate_sides[0].bbox_height, gate_sides[1].centre_y + gate_sides[1].bbox_height)

        # approach 1: distance based
        est_distance_from_width = self.gate_width * self.camera_infos.get_info(det.source).P[0]/(x2 - x1)
        est_distance_from_height = self.gate_height * self.camera_infos.get_info(det.source).P[5]/(y2 - y1)

        distance = (est_distance_from_height + est_distance_from_width) / 2
        # if np.abs(est_distance_from_height - est_distance_from_width) < 2.0:
        # else:
        #     distance = min(est_distance_from_height + est_distance_from_width)

        # TODO: compute 3d world coord from distance and camera intrinsics / extrinsics

        return detections
