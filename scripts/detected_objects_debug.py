import rospy
from copy import copy
from visualization_msgs.msg import Marker, MarkerArray
from bb_msgs.msg import DetectedObject, DetectedObjects
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
import tf2_ros
from math import pi
from transforms3d.euler import euler2quat
import matplotlib

rgb_colors = [
    matplotlib.colors.to_rgb(hex) for hex in matplotlib.colors.cnames.values()
]

def callback(msg):
    global object_markers, marker_pub

    # Clear existing markers
    marker_msg = MarkerArray()
    for marker_id in object_markers.keys():
        object_markers[marker_id].action = Marker.DELETE
        marker_msg.markers.append(object_markers[marker_id])

    marker_pub.publish(marker_msg)
    object_markers.clear()
    marker_msg = MarkerArray()

    # Create markers for each detected object
    idx = 0
    for obj in msg.detected:
        color = rgb_colors[idx % len(rgb_colors)]
        # Set pose based on move_coords
        if obj.move_coords == 2:
            # Use world_coords for position and yaw angle
            position = [obj.world_coords[0], obj.world_coords[1], obj.world_coords[2]]
            yaw = obj.world_yaw * pi / 180

            # Transform pose to fixed frame if needed
            if rospy.has_param("~fixed_frame"):
                fixed_frame = rospy.get_param("~fixed_frame")
                try:
                    trans = tf_buffer.transform(
                        PoseStamped(
                            header=msg.header,
                            pose=Pose(
                                position=position, orientation=Quaternion()
                            ),
                        ),
                        msg.header.frame_id,
                        fixed_frame,
                        rospy.Duration(1.0),
                    )
                    position = trans.pose.position
                    yaw = tf2_ros.transform_quaternion(
                        trans.pose.orientation, fixed_frame
                    )[2]
                except (
                    tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException,
                ):
                    rospy.logwarn("Error transforming pose to fixed frame")
            marker = Marker()
            marker.header = obj.header
            marker.ns = "bounding_box"
            marker.type = Marker.CUBE
            marker.color.a = 1.0
            marker.id = idx
            marker.color.r, marker.color.g, marker.color.b = color
            marker.pose.position.x = position[0]
            marker.pose.position.y = position[1]
            marker.pose.position.z = position[2]
            w, x, y, z = euler2quat(0, 0, yaw)
            quat = Quaternion(x, y, z, w)
            marker.pose.orientation = quat

            # Set scale based on real_dims
            marker.scale.x = obj.real_dims[0]
            marker.scale.y = obj.real_dims[1]
            marker.scale.z = obj.real_dims[2]
            marker.id = idx
            object_markers[idx] = marker
            marker_msg.markers.append(marker)
            idx += 1
        if len(obj.object_ray) == 6:
            # Use object_ray for line strip
            marker = Marker()
            marker.header = obj.header
            marker.ns = "bounding_box"
            marker.type = Marker.LINE_STRIP
            marker.id = idx
            marker.color.r, marker.color.g, marker.color.b = color
            marker.color.a = 1.0
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            p1 = Point()
            p1.x = obj.object_ray[3]
            p1.y = obj.object_ray[4]
            p1.z = obj.object_ray[5]
            p2 = Point()
            p2.x = obj.object_ray[3] + obj.object_ray[0] * 50.0
            p2.y = obj.object_ray[4] + obj.object_ray[1] * 50.0
            p2.z = obj.object_ray[5] + obj.object_ray[2] * 50.0
            marker.points.append(p1)
            marker.points.append(p2)
            marker.id = idx
            object_markers[idx] = copy(marker)
            marker_msg.markers.append(object_markers[idx])
            idx += 1

    marker_pub.publish(marker_msg)


# ROS node initialization
rospy.init_node("detected_object_visualizer")

# Publisher for MarkerArray
marker_pub = rospy.Publisher("/visualization_marker_array", MarkerArray, queue_size=10)

# Subscriber for DetectedObjectsStamped
detected_objects_sub = rospy.Subscriber("/auv4/vision/external/detected_filtered", DetectedObjects, callback)

# Buffer for storing object markers
object_markers = {}

# TF buffer and listener
tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)

rospy.spin()