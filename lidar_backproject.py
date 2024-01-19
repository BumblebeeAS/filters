from typing import List
import sys
import argparse
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from typing import Union
from rclpy.clock import Clock
import message_filters
from sensor_msgs.msg import PointCloud2
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from sensor_msgs_py.point_cloud2 import read_points_numpy
import numpy as np
from cv_bridge import CvBridge
import cv2

class LidarBackproject(Node):

    def __init__(self, node_name, topics, cam_info_topics, pointcloud_topic, debug=False, namespace="", **kwargs) -> None:
        super().__init__(node_name, namespace=namespace)

        self.get_logger().info(f"Starting {node_name} node")

        self.debug = debug

        self.namespace = namespace

        self.topics = [
            (self.resolve_topic_name(f"/{self.namespace}/{topic}") if topic[0] != "/"  else topic) for topic in topics
        ]

        self.cam_info_topics = [
            (self.resolve_topic_name(f"/{self.namespace}/{topic}") if topic[0] != "/"  else topic) for topic in cam_info_topics\
        ]

        self.pc_topic = self.resolve_topic_name(f"/{self.namespace}/{pointcloud_topic}") if pointcloud_topic[0] != "/"  else pointcloud_topic

        self.timer = [
            self.create_timer(
                5,
                lambda: self.subscriber_timeout_cb(i),
                clock=Clock()
            )
            for i in range(len(topics))
        ]
        self.timeouts = [True] * len(topics)

        self.pc_timer = self.create_timer(
            5,
            self.pc_timeout_cb,
            clock=Clock()
        )
        self.pc_timeout = True

        self.pc_sub = message_filters.Subscriber(
            self,
            PointCloud2,
            self.pc_topic
        )

        self.image_subs = [
            message_filters.Subscriber(
                self,
                CompressedImage if topic.endswith("compressed") else Image,
                topic,
            )
            for topic in self.topics
        ]

        self.subs = [
            message_filters.ApproximateTimeSynchronizer(
            [self.pc_sub, image_sub],
            1,
            0.1)
            for image_sub in self.image_subs
        ]

        for i, image_sub in enumerate(self.subs):
            image_sub.registerCallback(lambda msg: self.img_cb(msg, i))

        self.cam_info_subs = [
            self.create_subscription(
                CameraInfo,
                topic,
                self.cam_info_cb,
                1,
            )
            for i, topic in enumerate(self.cam_info_topics)
        ]

        
        self.tf_buffers = [Buffer() for i in self.topics]
        self.tf_listeners = [TransformListener(buffer, self) for buffer in self.tf_buffers]

        self.pub_topics = [
            f"{topic.split('image_')[0]}backprojection/compressed" for topic in self.topics
        ]
        self.proj_pubs = [
            self.create_publisher(
                CompressedImage,
                pub_topic,
                1
            ) for pub_topic in self.pub_topics
        ]

        if self.debug:
            self.debug_pubs = [
                self.create_publisher(
                    CompressedImage,
                    f"{topic.split('image_')[0]}backprojection/debug/compressed",
                    1
                ) for topic in self.topics
            ]

        self.bridge = CvBridge()

        self.get_logger().info(f"Subscribing to PointCloud [{self.pc_topic}]")
        for topic in self.topics:
            self.get_logger().info(f"Subscribing to Camera [{topic}]")

        for topic in self.pub_topics:
            self.get_logger().info(f"Publishing to [{topic}]")

    def img_cb(self, pc: PointCloud2, img: Union[CompressedImage, Image], i, compressed=False):
        if not img.header.frame_id in self.cam_info:
            self.get_logger().warn(f"Waiting for {img.header.frame_id} camera info")
            return
        camera_info: CameraInfo = self.cam_info[img.header.frame_id]
        self.timeouts[i] = False
        self.pc_timeout = False

        try:
            t = self.tf_buffers[i].lookup_transform(
                img.header.frame_id,
                pc.header.frame_id,
                rclpy.time.Time()
            )

            # Read PC
            xyz = read_points_numpy(pc)
            xyz[:, 3] = 1

            # Transform to camera frame
            t_mat = self.msg_to_se3(t)
            camera_xyz = xyz.dot(t_mat.T)

            # Filter Points Behind Camera
            camera_xyz = camera_xyz[camera_xyz[:,2] > 0]

            #Transform to image frame
            camera_k = np.reshape(np.array(camera_info.k), (3, -1))
            i_mat = np.hstack((camera_k, np.zeros((3,1))))
            image_pts = camera_xyz.dot(i_mat.T)

            # Take those in camera frame
            image = np.zeros((camera_info.height, camera_info.width))
            image_pts[:,0] /= image_pts[:,2]
            image_pts[:,1] /= image_pts[:,2]
            image_pts = image_pts[((0 <= image_pts[:,0]) & (image_pts[:,0] < 1280) & (0 <= image_pts[:,1]) & (image_pts[:,1] < 720))]
            for u, v, w  in image_pts:
                image[int(v), int(u)] = w

            pub_msg = self.bridge.cv2_to_compressed_imgmsg(image)
            self.proj_pubs[i].publish(pub_msg)

            if self.debug:
                # Get Image
                img = self.bridge.compressed_imgmsg_to_cv2(img)
                image *= (255.0/image.max())
                mask = cv2.applyColorMap(image.astype(np.uint8), self.custom_cmap)
                out = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 0.5, mask, 0.5, 0.0)
                debug_msg = self.bridge.cv2_to_compressed_imgmsg(out)
                self.debug_pubs[i].publish(debug_msg)

        except TransformException as e:
            self.get_logger().warn(f"Failed lookup of {img.header.frame_id} to {pc.header.frame_id}")



    def cam_info_cb(self, camera_info: CameraInfo):
        self.cam_info[camera_info.header.frame_id] = camera_info

    def subscriber_timeout_cb(self, i):
        if self.timeouts[i]:
            self.get_logger().warn(f"Topic [{self.topics[i]}] does not appear to be publishing")
        else:
            self.timeouts[i] = True

    def pc_timeout_cb(self):
        if self.pc_timeout:
            self.get_logger().warn(f"Topic [{self.pc_topic}] does not appear to be publishing")
        else:
            self.pc_timeout = True

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

def main(argv=sys.argv[1:]):
    try:
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "-n", "--namespace", choices=["auv3", "auv4", "asv4", "wamv"], required=True
        )
        parser.add_argument(
            "-t", "--topics", nargs="+", help="input topic name(s)", metavar="TOPIC"
        )
        parser.add_argument(
            "--cam_info_topics", nargs="+", help="camera info topic name(s)", metavar="CAM_INFO_TOPIC"
        )
        parser.add_argument(
            "-pc", "--pointcloud", help="point cloud topic name(s)", metavar="POINTCLOUD_TOPIC"
        )
        parser.add_argument(
            "--debug", help="enable debug mode", metavar="DEBUG"
        )

        opt, rest = parser.parse_known_args(argv[argv.index("--")+1 if "--" in argv else 0:])

        opt.topics = opt.topics or ["front_cam/image_rect_color/compressed"]
        opt.cam_info_topics = opt.cam_info_topics or [f"{topic.split('image_')[0]}camera_info" for topic in opt.topics]
        opt.pointcloud = opt.pointcloud or "merged_cloud"

        rclpy.init()

        lidar_backproject = LidarBackproject(
            "lidar_backproject", 
            topics=opt.topics, 
            cam_info_topics=opt.cam_info_topics,
            pointcloud_topic=opt.pointcloud,
            debug=opt.debug,
            namespace=opt.namespace)

        rclpy.spin(lidar_backproject)

        rclpy.shutdown()

    except rclpy.exceptions.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()