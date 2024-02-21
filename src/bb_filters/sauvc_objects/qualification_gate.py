from bb_msgs.msg import DetectedObject, DetectedObjects
from bb_filters import filter
import numpy as np
import rospy


class Filter(filter.Filter):
    def __init__(self, config, camera_infos: filter.CameraInfos):
        super(Filter, self).__init__(config, camera_infos)
        self.__name__ = "quallification_gate_filter"
        self.gate_orientation = np.pi / 2
        self.gate_width = 1.5
        self.gate_height = 1.0
        self.gate_depth = 0.6
        self.R = np.array(
            [
                [np.cos(-self.gate_orientation), -np.sin(-self.gate_orientation)],
                [np.sin(-self.gate_orientation), np.cos(-self.gate_orientation)],
            ]
        )

    def process(self, bboxes: DetectedObjects) -> DetectedObjects:
        detections = DetectedObjects()
        gate_sides = [x for x in bboxes.detected if x.name == "qualification_gate_side"]
        gate = [x for x in bboxes.detected if x.name == "qualification_gate"]
        if len(gate_sides) < 2 and len(gate) != 1:
            return detections
        gate_sides = sorted(gate_sides, key=lambda x: x.centre_x)
        if len(gate_sides) > 2:
            gate_sides = gate_sides[0], gate_sides[-1]
        det = gate_sides[0] if len(gate_sides) > 0 else gate[0]

        camera_yaw = self.camera_infos.get_camera_yaw(det.source, det.header.stamp)
        if camera_yaw is None:
            rospy.logerr("get_camera_yaw failed, possibly due to vehicle tilt")
            return detections
        if len(gate_sides) != 2:
            x1, x2 = (
                gate[0].centre_x - gate[0].bbox_width / 2,
                gate[0].centre_x + gate[0].bbox_width / 2,
            )
            y1, y2 = (
                gate[0].centre_y - gate[0].bbox_height / 2,
                gate[0].centre_y + gate[0].bbox_height / 2,
            )
        else:
            x1, x2 = gate_sides[0].centre_x, gate_sides[1].centre_x
            y1 = min(
                gate_sides[0].centre_y - gate_sides[0].bbox_height / 2,
                gate_sides[1].centre_y - gate_sides[1].bbox_height / 2,
            )
            y2 = max(
                gate_sides[0].centre_y + gate_sides[0].bbox_height / 2,
                gate_sides[1].centre_y + gate_sides[1].bbox_height / 2,
            )



        # # approach 1: distance based on width / height
        # dist_approaches = 0
        # distances = 0
        # if x2 - x1 > 20:
        #     dist_approaches += 1
        #     distances += (
        #         self.gate_width
        #         * self.camera_infos.get_info(det.source).P[0]
        #         / (x2 - x1)
        #     )
        # if y2 - y1 > 20:
        #     dist_approaches += 1
        #     distances += (
        #         self.gate_height
        #         * self.camera_infos.get_info(det.source).P[5]
        #         / (y2 - y1)
        #     )

        # gate_detection = det
        # gate_detection.centre_x = int((x1 + x2) / 2)
        # gate_detection.centre_y = int((y1 + y2) / 2)
        # gate_detection.bbox_width = int(x2 - x1)
        # gate_detection.bbox_height = int(y2 - y1)
        # gate_detection.bbox_area = int(
        #     gate_detection.bbox_width * gate_detection.bbox_height
        # )
        # gate_detection.move_coords = 1
        # gate_detection = self.camera_infos.compute_3d_coords_from_distance(
        #     gate_detection, distances / dist_approaches
        # )
        # gate_detection.real_dims = [0.2, 1.5, 1.2]
        # gate_detection.world_yaw = self.gate_orientation * 180 / np.pi
        # detections.detected.append(gate_detection)



        ## approach 2 using geometry

        left_ray = self.camera_infos.compute_object_ray_from_camera_coord(
            det.source, det.header.stamp, x1, y1
        )
        right_ray = self.camera_infos.compute_object_ray_from_camera_coord(
            det.source, det.header.stamp, x2, y2
        )
        centre_ray = self.camera_infos.compute_object_ray_from_camera_coord(
            det.source, det.header.stamp, (x1 + x2) / 2, (y1 + y2) / 2
        )
        cam_pos = left_ray[3:5]
        r1, r2 = np.linalg.inv(self.R)@left_ray[:2], np.linalg.inv(self.R)@right_ray[:2]
        l = self.gate_width / np.abs(r1[1]/r1[0]-r2[1]/r2[0])
        centroid = cam_pos + self.R @ np.array([l, r1[1] / r1[0] + self.gate_width / 2])
        gate_detection = det
        gate_detection.centre_x = int((x1 + x2) / 2)
        gate_detection.centre_y = int((y1 + y2) / 2)
        gate_detection.move_coords = 2
        gate_detection.world_coords = [centroid[0], centroid[1], self.gate_depth]
        gate_detection.real_dims = [0.2, 1.5, 1.2]
        gate_detection.world_yaw = self.gate_orientation * 180 / np.pi
        gate_detection.name = "qualification_gate"
        gate_detection.header.frame_id = self.camera_infos.map_frame
        gate_detection.object_ray = centre_ray
        detections.detected.append(gate_detection)

        return detections
