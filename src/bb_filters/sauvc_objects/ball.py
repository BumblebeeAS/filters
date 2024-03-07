from bb_msgs.msg import DetectedObject, DetectedObjects
from bb_filters import filter
import numpy as np
import rospy
from circle_fit import taubinSVD

class Filter(filter.Filter):
    def __init__(self, config, camera_infos: filter.CameraInfos):
        super(Filter, self).__init__(config, camera_infos)
        self.__name__ = "ball_filter"
        self.ball_diameter = 0.36
        self.ball_depth = 1.96

    def process(self, bboxes: DetectedObjects) -> DetectedObjects:
        detections = DetectedObjects()
        balls = [
            x for x in bboxes.detected if x.name == "ball" and x.source == 289
        ]
        if len(balls) == 0:
            return detections

        # filter by rectangularity?
        ball = max(
            balls, key=lambda x: x.extra[0]
        )  # get flare with highest confidence or height?

        if ball is not None:
            camera_depth = self.camera_infos.get_camera_z(289, balls[0].header.stamp)
            est_circle_radius = np.abs(
                (self.ball_diameter / 2)
                / (camera_depth - (self.ball_depth - self.ball_diameter / 2))
                * self.camera_infos.get_info(289).P[0]
            )
            try:
                xc, yc, r, sigma = taubinSVD(np.array(ball.contour).reshape(-1, 2))
            except:
                return detections
            if np.abs(r - est_circle_radius) < 100:
                ball.centre_x, ball.centre_y = max(0, int(xc)), max(0, int(yc))

                ball = self.camera_infos.compute_3d_coords_from_depth(ball, self.ball_depth - self.ball_diameter / 2)
                if ball is None:
                    rospy.logwarn("Failed to compute ball coord")
                    return detections
                ball.world_coords[2] -= self.ball_diameter / 2
                ball.real_dims = self.ball_diameter, self.ball_diameter, self.ball_diameter
                ball.name = "ball"
            else:
                rospy.loginfo_throttle(1.0, f"Bucket radius rejected: ({xc}, {yc}), {r} expect {est_circle_radius}")

        detections.detected.append(ball)
        return detections
