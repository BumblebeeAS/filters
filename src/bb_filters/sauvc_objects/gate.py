from bb_msgs.msg import DetectedObject, DetectedObjects
from bb_filters import filter
import numpy as np
import rospy
from math import sin, cos


class Filter(filter.Filter):
    def __init__(self, config, camera_infos: filter.CameraInfos):
        super(Filter, self).__init__(config, camera_infos)
        self.__name__ = "gate_filter"
        self.gate_orientation = 0.0
        self.gate_width = 1.5
        self.gate_side_width = 0.04
        self.gate_height = 1.5
        self.gate_depth = 1.25
        self.R = np.array(
            [
                [np.cos(-self.gate_orientation), -np.sin(-self.gate_orientation)],
                [np.sin(-self.gate_orientation), np.cos(-self.gate_orientation)],
            ]
        )

    def process(self, bboxes: DetectedObjects) -> DetectedObjects:
        detections = DetectedObjects()
        gate_left = [x for x in bboxes.detected if x.name == "gate_left"]
        gate_right = [x for x in bboxes.detected if x.name == "gate_right"]
        gate_sides = []
        if len(gate_left) == 0:
            gate_left = None
        else:
            gate_left = max(gate_left, key=lambda x: x.extra[0])
            gate_sides.append(gate_left)
        if len(gate_right) == 0:
            gate_right = None
        else:
            gate_right = max(gate_right, key=lambda x: x.extra[0])
            gate_sides.append(gate_right)

        if len(gate_sides) == 0:
            return detections

        img_height = self.camera_infos.get_info(gate_sides[0].source).height
        if len(gate_sides) == 1:
            print("1side")
            gate_side = gate_sides[0]
            # amount to transform from point of consideration
            if gate_side.name == "gate_left":
                dx, dy = self.gate_width / 2 * cos(
                    self.gate_orientation + np.pi/2
                ), self.gate_width / 2 * sin(self.gate_orientation + np.pi/2)
            else:
                dx, dy =  self.gate_width / 2 * cos(
                    self.gate_orientation + 3*np.pi/2
                ), self.gate_width / 2 * sin(self.gate_orientation + 3*np.pi/2)
            is_top_visible = gate_side.centre_y - gate_side.bbox_height / 2 > 30
            is_bottom_visible = (
                gate_side.centre_y + gate_side.bbox_height / 2 < img_height - 30
            )
            det = None
            if is_top_visible and is_bottom_visible:
                distance = (
                    self.gate_height
                    * self.camera_infos.get_info(gate_side.source).P[5]
                    / (gate_side.bbox_height)
                )
                det = self.camera_infos.compute_3d_coords_from_distance(
                    gate_side, distance
                )
            elif not is_bottom_visible and not is_top_visible:
                distance = (
                    self.gate_side_width
                    * self.camera_infos.get_info(gate_side.source).P[0]
                    / (gate_side.bbox_width)
                )
                det = self.camera_infos.compute_3d_coords_from_distance(
                    gate_side, distance
                )
            elif is_bottom_visible:
                gate_side.centre_y += int(gate_side.bbox_height / 2)
                det = self.camera_infos.compute_3d_coords_from_depth(
                    gate_side, self.gate_depth + self.gate_height / 2
                )
                det.centre_y -= int(gate_side.bbox_height / 2)
                det.world_coords[2] -= self.gate_height / 2
            else:
                gate_side.centre_y -= int(gate_side.bbox_height / 2)
                det = self.camera_infos.compute_3d_coords_from_depth(
                    gate_side, self.gate_depth - self.gate_height / 2
                )
                det.centre_y += int(gate_side.bbox_height / 2)
                det.world_coords[2] += self.gate_height / 2
            print(dx, dy, gate_side.name)
            det.world_coords[0] += dx
            det.world_coords[1] += dy
            det.real_dims = [0.2, 1.5, 1.5]
            det.world_yaw = self.gate_orientation * 180 / np.pi
            det.name = "gate"
            detections.detected.append(det)
            return detections

        # gate = [x for x in bboxes.detected if x.name == "gate"]
        det = gate_sides[0]

        camera_yaw = self.camera_infos.get_camera_yaw(det.source, det.header.stamp)
        if camera_yaw is None:
            rospy.logerr("get_camera_yaw failed, possibly due to vehicle tilt")
            return detections

        # only if considering height of sides
        is_top_visible = gate_sides[0].centre_y - gate_sides[0].bbox_height / 2 > 30
        is_bottom_visible = (
            gate_sides[0].centre_y + gate_sides[0].bbox_height / 2 < img_height - 30
        )

        left_ray = self.camera_infos.compute_object_ray_from_camera_coord(
            det.source, det.header.stamp, gate_left.centre_x, gate_left.centre_y
        )
        right_ray = self.camera_infos.compute_object_ray_from_camera_coord(
            det.source, det.header.stamp, gate_right.centre_x, gate_right.centre_y
        )
        centre_ray = self.camera_infos.compute_object_ray_from_camera_coord(
            det.source, det.header.stamp, (gate_left.centre_x + gate_right.centre_x) / 2, gate_right.centre_y
        )
        cam_pos = left_ray[3:5]
        r1, r2 = np.linalg.inv(self.R)@left_ray[:2], np.linalg.inv(self.R)@right_ray[:2]
        l = self.gate_width / np.abs(r1[1]/r1[0]-r2[1]/r2[0])
        centroid = cam_pos + self.R @ np.array([l, r1[1] / r1[0] + self.gate_width / 2])

        gate_detection = det
        gate_detection.centre_x = int((gate_left.centre_x + gate_right.centre_x) / 2)
        gate_detection.move_coords = 2
        gate_detection.world_coords = [centroid[0], centroid[1], self.gate_depth]
        gate_detection.real_dims = [0.2, 1.5, 1.5]
        gate_detection.world_yaw = self.gate_orientation * 180 / np.pi
        gate_detection.name = "gate"
        gate_detection.header.frame_id = self.camera_infos.map_frame
        gate_detection.object_ray = centre_ray
        detections.detected.append(gate_detection)

        return detections
