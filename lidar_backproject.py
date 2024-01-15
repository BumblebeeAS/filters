import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py.point_cloud2 import read_points, read_points_numpy, create_cloud
from sensor_msgs.msg import CameraInfo

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs.tf2_geometry_msgs import _transform_to_affine
import numpy as np
import cv2
from std_msgs.msg import Header

from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Transform
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Vector3

from geometry_msgs.msg import Point
import tf_transformations

import message_filters
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage, NavSatFix
from bb_msgs.msg import DetectedObjectsStamped
from visualization_msgs.msg import MarkerArray, Marker
from copy import deepcopy
from std_msgs.msg import ColorRGBA
from rclpy.duration import Duration
from tf2_geometry_msgs import PoseStamped as TF2PoseStamped
from operator import attrgetter


class LidarBackproject(Node):

    bridge = CvBridge()

    def __init__(self):
        super().__init__('lidar_backproject')

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.tf_buffer = Buffer(Duration(seconds=15), self)
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        self.camera_k = np.array([
            [762.72232056, 0., 640],
            [0., 762.72231102, 360],
            [0.,      0.,       1.]
        ])
        self.camera_k = self.camera_k / self.camera_k[2][2]

        self.detections = []

        self.publisher_ = self.create_publisher(
            MarkerArray, '/stereo_visualization', 10)

        # self.detections_pub_3d = self.create_publisher(
        #     DetectedObjectsStamped, '/wamv/vision/stereo/detected', 10)

        self.detections_pub_3d = self.create_publisher(
            # DetectedObjectsStamped, '/wamv/vision/fused_detections', 10)
            DetectedObjectsStamped, '/wamv/vision/lidar/detected_stamped', 10)

        self.camera_info_sub = self.create_subscription(
            CameraInfo, 
            '/wamv/sensors/cameras/mid_cam_sensor/optical/camera_info',
            self.info_cb,
            1)

        self.leftSub = message_filters.Subscriber(
            self,
            CompressedImage,
            self.resolve_topic_name('/wamv/sensors/cameras/mid_cam_sensor/optical/image_rect_color/compressed')
        )

        self.pointcloud_subscriber = message_filters.Subscriber(
            self,
            PointCloud2,
            "/nonground"
        )

        self.ml_detection_sub = message_filters.Subscriber(
            self,
            DetectedObjectsStamped,
            self.resolve_topic_name('/wamv/vision/external/detected_stamped')
        )

        self.ApproxTime = message_filters.ApproximateTimeSynchronizer(
            [self.leftSub, self.pointcloud_subscriber, self.ml_detection_sub],
            1,
            0.1)
        
        self.ApproxTime.registerCallback(self.sync_callback)

        h = np.expand_dims(np.arange(0,128, 0.5), 1)
        sv = np.ones((256, 2))*255
        hsv = np.hstack((h, sv))
        hsv[0] = [0,0,0]
        self.custom_cmap = cv2.cvtColor(np.expand_dims(hsv, 1).astype(np.uint8), cv2.COLOR_HSV2BGR)

    def sync_callback(self, img: CompressedImage, pc: PointCloud2, detected_objects : DetectedObjectsStamped):

        detected_objects_3d = DetectedObjectsStamped()
        detected_objects_3d.header = detected_objects.header
        detected_objects_3d.header.frame_id = 'wamv/wamv/base_link'

        if len(detected_objects.detected) > 0:
            markers = MarkerArray()
            marker = Marker()
            marker.header = img.header
            marker.scale = Vector3(x=1.0, y=1.0, z=1.0)
            marker.pose.orientation = Quaternion(
                w=0.707, x=0.707, y=0.0, z=0.0)
            marker.action = Marker.DELETEALL
            markers.markers.append(marker)
            marker.action = Marker.ADD

            try:
                t = self._tf_buffer.lookup_transform(
                    'wamv/wamv/base_link/mid_cam_sensor_optical',
                    'wamv/wamv/os1_link',
                    rclpy.time.Time()
                )

                # Get Image
                img = self.bridge.compressed_imgmsg_to_cv2(img)

                # Read Point Cloud
                xyz = read_points_numpy(pc)
                xyz[:, 3] = 1

                # Transform to camera frame
                t_mat = self.msg_to_se3(t)
                camera_xyz = xyz.dot(t_mat.T)

                # Filter Points Behind Camera
                camera_xyz = camera_xyz[camera_xyz[:,2] > 0]

                #Transform to image frame
                i_mat = np.hstack((self.camera_k, np.zeros((3,1))))
                image_pts = camera_xyz.dot(i_mat.T)

                # Take those in camera grame
                image = np.zeros((720, 1280))
                image_pts[:,0] /= image_pts[:,2]
                image_pts[:,1] /= image_pts[:,2]
                image_pts = image_pts[((0 <= image_pts[:,0]) & (image_pts[:,0] < 1280) & (0 <= image_pts[:,1]) & (image_pts[:,1] < 720))]
                for u, v, w  in image_pts:
                    # if 0 <= u < 1280 and 0 <= v < 720:
                    image[int(v), int(u)] = w

                # Overlay Lidar
                depth = image.copy()
                image *= (255.0/image.max())
                mask = cv2.applyColorMap(image.astype(np.uint8), self.custom_cmap)
                out = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 0.5, mask, 0.5, 0.0)

                # Process Detections
                for i, obj in enumerate(detected_objects.detected):
                    marker.id = i
                    cnt = np.array(obj.contour).reshape(-1, 2).astype(np.int32)
                    mask = np.zeros_like(depth, np.uint8)
                    cv2.drawContours(mask, [cnt.astype(np.int32)], 0, (1), -1)
                    cluster = np.multiply(depth, mask)
                    cluster = cluster[cluster!=0]
                    objDepth = np.median(cluster)
                    self.get_logger().info(f"LidarFuse {i} {obj.name} {objDepth}")
                    
                    M = cv2.moments(cnt.astype(np.int32))
                    cX = M["m10"]/max(M["m00"], 1e-8)  # Avoid Division by Zero
                    cY = M["m01"]/max(M["m00"], 1e-8)

                    # Calculate Ray
                    r = np.dot(np.linalg.inv(self.camera_k), np.array([cX, cY, 1.0]).T)

                    ny = np.array([0, 1, 0])
                    nx = np.array([1, 0, 0])

                    color = ColorRGBA()
                    if "round" in obj.name:
                        marker.type = Marker.SPHERE
                    else:
                        marker.type = Marker.CYLINDER

                    if "green" in obj.name:
                        color.r = 0.0
                        color.g = 1.0
                        color.b = 0.0
                    elif "red" in obj.name:
                        color.r = 1.0
                        color.g = 0.0
                        color.b = 0.0
                    elif "black" in obj.name:
                        color.r = 0.0
                        color.g = 0.0
                        color.b = 0.0
                    elif "orange" in obj.name:
                        color.r = 1.0
                        color.g = 0.64
                        color.b = 0.0
                    elif "white" in obj.name:
                        color.r = 1.0
                        color.g = 1.0
                        color.b = 1.0
                    elif "rgb" in obj.name:
                        color.r = 0.0
                        color.g = 0.0
                        color.b = 1.0
                    else:
                        print(obj.name)
                    color.a = 1.0

                    p = Point()
                    p.x = objDepth * np.dot(r, nx)
                    p.y = objDepth * np.dot(r, ny)
                    p.z = 1.0 * objDepth

                    marker.color = color
                    marker.pose.position = p
                    markers.markers.append(deepcopy(marker))

                    try:
                        tf2_pose = TF2PoseStamped()
                        tf2_pose.header = marker.header
                        tf2_pose.pose = marker.pose
                        # tf2_pose.header.stamp.sec = marker.header.stamp.sec - 1
                        world_pose = self.tf_buffer.transform(
                            tf2_pose, "world_ned", Duration(seconds=0.0))
                        obj.world_coords = list(
                            attrgetter("x", "y", "z")(world_pose.pose.position))
                        if np.isnan(obj.world_coords[0]):
                            continue
                    except Exception as e:
                        self.get_logger().warn(
                            f"Failed to convert to world {e}")
                    try:
                        world_pose = self.tf_buffer.transform(
                            tf2_pose, "wamv/wamv/base_link_ned", Duration(seconds=0.0))
                        obj.move_coords = 2
                        obj.rel_coords = list(
                            attrgetter("x", "y", "z")(world_pose.pose.position))
                        obj.tracker_confidence.append(obj.extra[0])
                        obj.real_dims = [0.5, 0.5, 0.5]
                    except Exception as e:
                        self.get_logger().warn(
                            f"Failed to convert to world_ned {e}")
                    detected_objects_3d.detected.append(obj)
                self.detections_pub_3d.publish(detected_objects_3d)
                self.publisher_.publish(markers)

            except TransformException as e:
                print(e)
        else:
            print(detected_objects_3d)
        
        # print(xyz)

    def detection_cb(self, msg):
        self.detections = []

        for detected in msg.detected:
            cnts = np.vstack((detected.contour[0::2], detected.contour[1::2])).T.astype(np.int32)
            self.detections.append([detected.name, cnts])

    def info_cb(self, msg):
        self.camera_k = np.reshape(np.array(msg.k), (3, -1))

    def pose_to_pq(msg):
        """Convert a C{geometry_msgs/Pose} into position/quaternion np arrays

        @param msg: ROS message to be converted
        @return:
        - p: position as a np.array
        - q: quaternion as a numpy array (order = [x,y,z,w])
        """
        p = np.array([msg.position.x, msg.position.y, msg.position.z])
        q = np.array([msg.orientation.x, msg.orientation.y,
                    msg.orientation.z, msg.orientation.w])
        return p, q


    def pose_stamped_to_pq(self, msg):
        """Convert a C{geometry_msgs/PoseStamped} into position/quaternion np arrays

        @param msg: ROS message to be converted
        @return:
        - p: position as a np.array
        - q: quaternion as a numpy array (order = [x,y,z,w])
        """
        return pose_to_pq(msg.pose)


    def transform_to_pq(self, msg):
        """Convert a C{geometry_msgs/Transform} into position/quaternion np arrays

        @param msg: ROS message to be converted
        @return:
        - p: position as a np.array
        - q: quaternion as a numpy array (order = [x,y,z,w])
        """
        p = np.array([msg.translation.x, msg.translation.y, msg.translation.z])
        q = np.array([msg.rotation.x, msg.rotation.y,
                    msg.rotation.z, msg.rotation.w])
        return p, q


    def transform_stamped_to_pq(self, msg):
        """Convert a C{geometry_msgs/TransformStamped} into position/quaternion np arrays

        @param msg: ROS message to be converted
        @return:
        - p: position as a np.array
        - q: quaternion as a numpy array (order = [x,y,z,w])
        """
        return self.transform_to_pq(msg.transform)


    def msg_to_se3(self, msg):
        """Conversion from geometric ROS messages into SE(3)

        @param msg: Message to transform. Acceptable types - C{geometry_msgs/Pose}, C{geometry_msgs/PoseStamped},
        C{geometry_msgs/Transform}, or C{geometry_msgs/TransformStamped}
        @return: a 4x4 SE(3) matrix as a numpy array
        @note: Throws TypeError if we receive an incorrect type.
        """
        if isinstance(msg, Pose):
            p, q = self.pose_to_pq(msg)
        elif isinstance(msg, PoseStamped):
            p, q = self.pose_stamped_to_pq(msg)
        elif isinstance(msg, Transform):
            p, q = self.transform_to_pq(msg)
        elif isinstance(msg, TransformStamped):
            p, q = self.transform_stamped_to_pq(msg)
        else:
            raise TypeError("Invalid type for conversion to SE(3)")
        norm = np.linalg.norm(q)
        if np.abs(norm - 1.0) > 1e-3:
            raise ValueError(
                "Received un-normalized quaternion (q = {0:s} ||q|| = {1:3.6f})".format(
                    str(q), np.linalg.norm(q)))
        elif np.abs(norm - 1.0) > 1e-6:
            q = q / norm
        g = tf_transformations.quaternion_matrix(q)
        g[0:3, -1] = p
        return g

def main(args=None):
    rclpy.init(args=args)

    lidar_backproject = LidarBackproject()

    rclpy.spin(lidar_backproject)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
    