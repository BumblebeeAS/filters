# create rclpy node that subscribes to DetectedObject2DArray, finds red_placard, and estimates pose based on camera_info, and known dimensions of placard being 60cm x 60cm

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
import numpy as np
import cv2
from rclpy.time import Time
from rclpy.qos import QoSProfile
import tf2_ros
from geometry_msgs.msg import TransformStamped, PoseStamped
from bb_perception_msgs.msg import DetectedObject2DArray
from ament_index_python.packages import get_package_share_directory
from ml_detector.schema_validator import get_config, load_schema
from pathlib import Path

class PlacardPoseEstimator(Node):
    def __init__(self):
        super().__init__('placard_pose_estimator')
        self.bridge = CvBridge()


        objects_schema_path = (
            Path(get_package_share_directory("ml_detector"))
            / "configs"
            / "objects_schema.json"
        )
        self.objects_schema = load_schema(objects_schema_path)
        self.declare_parameter("objects_config", "robotx.yaml")
        self.objects_config = get_config(
            Path(get_package_share_directory("ml_detector"))
            / "configs"
            / "objects"
            / self.get_parameter("objects_config").get_parameter_value().string_value,
            self.objects_schema,
        )
        self.id_to_name = {
            obj["label"]: obj["name"] for obj in self.objects_config["objects"]
        }
        self.name_to_id = {v: k for k, v in self.id_to_name.items()}
        self.green_placard_id = self.name_to_id["placard_symbol_green"]
        self.red_placard_id = self.name_to_id["placard_symbol_red"]
        self.blue_placard_id = self.name_to_id["placard_symbol_blue"]
        self.cam_info = None
        self.cam_info_sub = self.create_subscription(
            CameraInfo,
            '/asv4/left_cam/camera_info',
            self.camera_info_callback,
            10)
        self.det2d_sub = self.create_subscription(
            DetectedObject2DArray,
            '/asv4/vision/detections_2d',
            self.detected_objects_callback,
            10)
        self.pub = self.create_publisher(PoseStamped, '/asv4/vision/placard_pose', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()

    def camera_info_callback(self, msg):
        self.cam_info = msg

    def detected_objects_callback(self, msg):
        for obj in msg.objects:
            if obj.hypothesis.class_id in [
                self.red_placard_id,
                self.green_placard_id,
                self.blue_placard_id
            ]:
                self.estimate_pose(obj)

    def estimate_pose(self, obj):
        # get camera info
        cam_info = self.cam_info
        # get object centroid
        x = obj.x
        y = obj.y
        # get object pose
        # get camera intrinsic parameters
        fx = cam_info.P[0]
        fy = cam_info.P[5]
        cx = cam_info.P[2]
        cy = cam_info.P[6]
        
        contour = np.asarray(obj.contour, dtype=np.int32)
        # estimate yaw and translation of object from camera from contour

        known_width = 0.6  # 60cm
        known_height = 0.6
        # check if contour has 4 points
        if len(contour) != 4:
            return
        print("Contour has 4 points")


if __name__ == '__main__':
    rclpy.init()
    node = PlacardPoseEstimator()
    rclpy.spin(node)
    rclpy.shutdown()

        # z = fx * obj_height / obj.bbox_width
        # # estimate object position in camera frame
        # x = (x - cx) * z / fx
        # y = (y - cy) * z / fy
        # # create PoseStamped message
        # pose = PoseStamped()
        # pose.header.stamp = Time.now()
        # pose.header.frame_id = cam_info.header.frame_id

        # self.pub.publish(pose)
        # # broadcast transform
        # t = TransformStamped()
        # t.header.stamp = Time.now()
        # t.header.frame_id = 'map_ned'
        # t.child_frame_id = obj.label
        # t.transform.translation.x = x
        # t.transform.translation.y = y
        # t.transform.translation.z = z
        # t.transform.rotation.w = 1
        # self.tf_broadcaster.sendTransform(t)