#!/usr/bin/env python
import cv2
import rospy
import numpy as np
from tf2_ros import Buffer
from transforms3d.quaternions import quat2axangle, quat2mat
from transforms3d.axangles import mat2axangle
from bb_msgs.msg import DetectedObjects
from sensor_msgs.msg import CameraInfo
from operator import attrgetter

class CameraInfos:
    def __init__(self, buffer: Buffer, map_frame: str = "map_ned"):
        self.infos = {}
        self.map_frame = map_frame
        self.buffer = buffer
    
    def set_info(self, frame_id, info: CameraInfo):
        self.infos[frame_id] = info
    
    def get_info(self, frame_id):
        return self.infos.get(frame_id, None)

    def get_camera_yaw(self, frame_id, stamp: rospy.Time):
        tf = self.buffer.lookup_transform(
            self.map_frame, self.infos[frame_id].header.frame_id, stamp, timeout = rospy.Duration(0.05))
        cam_mat = quat2mat(attrgetter("w","x","y","z")(tf.transform.rotation))
        if np.abs(np.dot(np.array([0,0,1]),cam_mat@np.array([0,0,1]))) < 0.2: # front cam
            cam_z = cam_mat @ np.array([0,0,1])
            yaw = np.arctan2(cam_z[1], cam_z[0])
            return yaw
        elif np.dot(np.array([0,0,1]),cam_mat@np.array([0,0,1])) > 0.8: # bottom cam:
            cam_z = cam_mat @ np.array([0,-1,0])
            yaw = np.arctan2(cam_z[1], cam_z[0])
            return yaw
        else:
            return None

class Filter(object):
    def __init__(self, config, camera_infos: CameraInfos):
        self.config = config
        self.camera_infos = camera_infos

    # returns processed image, list of DetectedObject msg
    def process(self, bboxes: DetectedObjects) -> DetectedObjects:
        pass


def draw_detected_object(out_img, info, cnt, shape='circle'):

    print(type(info))
    draw_centroid(out_img, info.bbox[:2], 2, Color.orange)
    cv2.drawContours(out_img, [cnt], -1, Color.purple.bgr(), 1)

    '''
    draw_text(out_img, "H:{:.1f}".format(info.color[0]), (info.centroid[0] + 20, info.centroid[1] + 10))
    draw_text(out_img, "L:{:.1f}".format(info.color[1]), (info.centroid[0] + 75, info.centroid[1] + 10))
    draw_text(out_img, "A:{:.1f}".format(info.color[2]), (info.centroid[0] + 20, info.centroid[1] + 25))
    draw_text(out_img, "B:{:.1f}".format(info.color[3]), (info.centroid[0] + 75, info.centroid[1] + 25))
    draw_text(out_img, "Area:{:.1f}".format(info.bbox_area, (info.centroid[0] + 20, info.centroid[1] + 40))
    draw_text(out_img, "Color:{}".format(info.predicted_color), (info.centroid[0] + 20, info.centroid[1] + 55))
    draw_text(out_img, "Dist:{:.1f}".format(info.distance), (info.centroid[0] + 20, info.centroid[1] + 70))
    '''
    # if info.name == 'oval' or info.name[-4:] == 'oval':
    #     ellipse = cv2.fitEllipse(cnt)
    #     cv2.ellipse(out_img,ellipse,cnt_color,2)
    # if info.name not in ['vertical_coin']:
    #     cv2.drawContours(out_img, [np.int0(cv2.boxPoints(cv2.minAreaRect(cnt)))], -1, cnt_color, 2)
    # else:
    if shape == 'rect':
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(out_img, [box], 0, Color.orange.bgr(), 2)


        #x1, y1, w, h = cv2.boundingRect(cnt)
        #cv2.rectangle(out_img, (x1,y1), (x1+w, y1+h), Color.orange.bgr(), 2)

    else: #if shape == 'circle':
        centroid, radius = cv2.minEnclosingCircle(cnt)
        cv2.circle(out_img, (int(centroid[0]), int(centroid[1])), int(radius), Color.orange.bgr(), 2)



''' Contour information '''

def get_centroid(cnt):
    mom = cv2.moments(cnt)
    centroid_x = int((mom['m10'] + 0.0001) / (mom['m00'] + 0.0001))
    centroid_y = int((mom['m01'] + 0.0001) / (mom['m00'] + 0.0001))
    return (centroid_x, centroid_y)

def get_ratio(cnt):
    rect = cv2.minAreaRect(cnt)
    if rect[1][0] <= 0 or rect[1][1] <= 0:
        return 0

    if rect[1][0] > rect[1][1]:
        return rect[1][0] / float(rect[1][1])
    else:
        return rect[1][1] / float(rect[1][0])

# TODO take into account the front camera
def get_offset(centroid, proc_size):
    x = proc_size[0]
    y = proc_size[1]
    dx = (centroid[0] - (x / 2)) / float(x)
    dy = ((y / 2) - centroid[1]) / float(y)
    return dx, dy

def get_rect_area(cnt):
    rect = cv2.minAreaRect(cnt)
    return int(rect[1][0] * rect[1][1])

def get_rect_angles(cnt):
    (_x,_y), (rect_width,rect_height), long_edge_angle = cv2.minAreaRect(cnt)

    # get the smallest angle to turn (less than 90 degrees) parallel to LONG EDGE
    if rect_width < rect_height:
        long_edge_angle += 180
    else:
        long_edge_angle += 90

    if long_edge_angle > 90:
        long_edge_angle = -1 * abs(180 - long_edge_angle)
    """
    if (rect_width < rect_height):
        if (long_edge_angle == -90):
            long_edge_angle = 0
        else:
            long_edge_angle = -1 * abs(long_edge_angle)
    else:
        if (long_edge_angle == -90):
            long_edge_angle = 0
        else:
            long_edge_angle = (90 - abs(long_edge_angle))
    """

    # get the smallest angle to turn (less than 90 degrees) parallel to SHORT EDGE
    if long_edge_angle > 0:
        short_edge_angle = long_edge_angle - 90
    else:
        short_edge_angle = 90 + long_edge_angle

    return long_edge_angle, short_edge_angle
    

def get_rect_angle_long(cnt):
    """ Returns angle perpendicular to long side of a rectangle """
    return (get_rect_angles(cnt))[0]


def get_rect_angle_short(cnt):
    """ Returns angle perpendicular to short side of a rectangle """
    return (get_rect_angles(cnt))[1]

def get_circle_area(cnt):
    rad = cv2.minEnclosingCircle(cnt)[1]
    return np.pi * rad * rad


''' Shape validation '''

def get_rectangularity(cnt):
    return cv2.contourArea(cnt) / (get_rect_area(cnt) + 0.001)

def get_circularity(cnt):
    return cv2.contourArea(cnt) / (get_circle_area(cnt) + 0.001)

def get_pixel_area(cnt, mask):
    minRect = cv2.boundingRect(cnt) #x, y, w, h
    roi = mask[minRect[1]:minRect[1]+minRect[3], minRect[0]:minRect[0]+minRect[2]]
    return cv2.countNonZero(roi)

def is_circle(cnt, ratio_min=0.8, ratio_max=1.2, circle_limit=0.7):
    """ Checks if rectangle with certain ratio """
    asp_rat = get_ratio(cnt)
    return ratio_min <= asp_rat and asp_rat <= ratio_max \
        and get_circularity(cnt) > circle_limit


def is_rect(cnt, ratio_min=0, ratio_max=3, rect_limit=0.6):
    """ Checks if rectangle with certain ratio """
    asp_rat = get_ratio(cnt)
    return ratio_min <= asp_rat and asp_rat <= ratio_max \
        and get_rectangularity(cnt) > rect_limit

def is_within(x, ref):
    return ref[0] <= x <= ref[1]

def is_correct_area(area, ref_area):
    return is_within(area, ref_area)

def is_correct_ratio(ratio, ref_ratio):
    return is_within(ratio, ref_ratio)


def format_info(infos):
    # Assuming that object with highest score is inserted first
    area = infos[0].bbox_area if infos else 0.0
    # predicted_color = get_color(infos[0].color).name if infos else ""
    predicted_color = str(infos[0].color) if infos else ""
    angle = infos[0].angle if infos else 0.0
    area = ("area: ", "{:.4f}".format(area))
    hue = ("color: ", predicted_color)
    angle = ("angle: ", "{:.4f}".format(angle))
    detected = ("detected: ", str(len(infos)))
    return [area, hue, angle, detected]
