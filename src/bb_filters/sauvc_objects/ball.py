from bb_msgs.msg import DetectedObject, DetectedObjects
from bb_filters import filter
import numpy as np
import rospy
from circle_fit import taubinSVD
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class Filter(filter.Filter):
    def __init__(self, config, camera_infos: filter.CameraInfos):
        super(Filter, self).__init__(config, camera_infos)
        self.__name__ = "ball_filter"
        self.ball_diameter = config["ball_diameter"]
        self.ball_depth = config["ball_depth"]
        self.bridge = CvBridge()

        self.ball_mask_plt_pub = rospy.Publisher(
            "/vision/ball_mask_plt", Image, queue_size=1
        )

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
            if np.abs(r - est_circle_radius) < 200:
                ball.centre_x, ball.centre_y = max(0, int(xc)), max(0, int(yc))

                ball = self.camera_infos.compute_3d_coords_from_depth(ball, self.ball_depth - self.ball_diameter / 2)
                if ball is None:
                    rospy.logwarn("Failed to compute ball coord")
                    return detections
                ball.world_coords[2] -= self.ball_diameter / 2
                ball.real_dims = self.ball_diameter, self.ball_diameter, self.ball_diameter
                ball.name = "ball"
            else:
                rospy.loginfo_throttle(1.0, f"Ball radius rejected: ({xc}, {yc}), {r} expect {est_circle_radius}")
                return detections

        detections.detected.append(ball)
        return detections
    def get_circle_area(cnt):
        rad = cv2.minEnclosingCircle(cnt)[1]
        return np.pi * rad * rad
    def get_circularity(cnt):
        return cv2.contourArea(cnt) / (Filter.get_circle_area(cnt) + 0.001)
    def get_rectangularity(cnt):
        return cv2.contourArea(cnt) / (cv2.minAreaRect(cnt)[1][0] * cv2.minAreaRect(cnt)[1][1] + 0.001)

    # def process_bot_cam(self, img: np.ndarray) -> DetectedObjects:
    #     dets = DetectedObjects()

    #     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #     # Apply the configuration parameters to the image processing
    #     mask = cv2.inRange(hsv, (10, 0, 0), (173, 255, 255))
    #     rgb_mask = cv2.inRange(img, (0, 68, 71), (74, 255, 224))
    #     res = cv2.bitwise_and(img, img, mask=np.bitwise_and(mask, rgb_mask))
    #     kernel = np.ones((3, 3), np.uint8)
    #     mask = cv2.dilate(mask, kernel, iterations=1)
    #     contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     debug_mask = np.zeros_like(mask)

        
    #     for contour in contours:
    #         print("found yellow ball")
    #         area = cv2.contourArea(contour)
    #         if area <200:
    #             continue
    #         # filter by rectangularity?
    #         # if len(contour) < 3 or len(contour) > 10:
    #         #     continue

    #         if Filter.get_rectangularity(contour) < 0.5:
    #             continue
    #         det = DetectedObject()
    #         det.bbox_width = int(contour[:, 0, 0].max() - contour[:, 0, 0].min())
    #         det.bbox_height = int(contour[:, 0, 1].max() - contour[:, 0, 1].min())

    #         aspect_ratio = det.bbox_width / (det.bbox_height + 1)
    #         if aspect_ratio < 0.7 or aspect_ratio > 1.2:
    #             continue

    #         det.contour = contour
    #         det.centre_x = int(contour[:, 0, 0].mean())
    #         det.centre_y = int(contour[:, 0, 1].mean())
    #         det.name = "ball"
    #         det.color = hsv[det.centre_y, det.centre_x, 0]
    #         det.source = 289
    #         det.header.stamp = rospy.Time.now()
    #         det.extra = [0.5]
    #         dets.detected.append(det)
    #         cv2.drawContours(debug_mask, [contour], 0, (255, 255, 255), -1)
    #     self.ball_mask_plt_pub.publish(
    #         self.bridge.cv2_to_imgmsg(debug_mask, encoding="mono8")
    #     )
    #     return dets