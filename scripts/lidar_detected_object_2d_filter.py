import rclpy
from rclpy.node import Node
import time
from collections import defaultdict
import numpy as np
from bb_msgs.msg import DetectedObject, DetectedObjectsStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from bb_filters.sort_3d import SORT3D

class LidarTrackerFilter(Node):
    def __init__(self):
        super().__init__('motracker_lidar_iou_tracker_node')
        self.declare_parameter("fused_dets_pub_topic", "/wamv/vision/fused_detections")
        self.declare_parameter("detections_2d_topic", "/wamv/vision/external/detected_stamped")
        self.declare_parameter("detections_3d_topic", "/wamv/vision/lidar/detected_stamped")
        self.fused_dets_pub_topic = self.get_parameter("fused_dets_pub_topic")\
                                        .get_parameter_value().string_value
        self.detections_2d_topic = self.get_parameter("detections_2d_topic")\
                                    .get_parameter_value().string_value
        self.detections_3d_topic = self.get_parameter("detections_3d_topic")\
                                    .get_parameter_value().string_value
        self.tracked_objects_pub = self.create_publisher(
            DetectedObjectsStamped, self.fused_dets_pub_topic, 2)
        self.tracks_vis_pub = self.create_publisher(
            MarkerArray, "/lidar_tracks_vis", 2
        )
        self.detections_3d_sub = self.create_subscription(
            DetectedObjectsStamped, self.detections_3d_topic,
            self.object_list_callback, 1)
        self.detections_2d_sub = self.create_subscription(
            DetectedObjectsStamped, self.detections_2d_topic,
            self.object_list_callback, 1)
        self.tracker = SORT3D(
            max_lost=4, # constant acc cannot model changing directions well
            # min_detection_confidence=0.2,
            # max_detection_confidence=1.0,
            process_noise_scale=0.1,
            measurement_noise_scale=0.05,
            time_step=0.2,
            dist_threshold=1.6)
        self.latest_header = None
        self.min_age = 1
        self.track_counts = defaultdict(lambda: defaultdict(float))
        self.entities = {}
        self.reverse_entities = {}

    

    def name_to_id(self, name):
        if name in self.entities.keys():
            return self.entities[name]
        id = len(self.entities)

        self.entities[name] = id
        self.reverse_entities[id] = name
        return id
        # return self.entities

    def id_to_name(self, id):        
        if id >= len(self.entities) or id < 0:
            return ""
        return self.reverse_entities[id]


    def object_list_callback(self, msg: DetectedObjectsStamped):
        self.latest_header = msg.header
        bboxes, confidences, ids = [], [], []

        rays_2d, confidences_2d, ids_2d = [], [], []

        for det in msg.detected:
            if det.move_coords == 2 and len(det.world_coords) != 0:
                id = self.name_to_id(det.name)
                if det.world_coords[2] < -3 or det.world_coords[2] > 3:
                    continue
                if det.real_dims[0] > 4 or det.real_dims[1] > 4 or det.real_dims[2] > 4:
                    # self.get_logger().info("object dims out of bounds")
                    continue
                if det.real_dims[0] < 0.01 and det.real_dims[1] < 0.01 and det.real_dims[0] < 0.01:
                    continue
                confidence = det.extra[0] if len(det.extra)>0 else 0.8
                if confidence < 0.5:
                    continue
                bboxes.append([det.world_coords[0],
                            det.world_coords[1],
                            det.world_coords[2],
                            max(0.5, det.real_dims[0]*1.2),
                            max(0.5, det.real_dims[1]*1.2),
                            max(0.5, det.real_dims[2]*1.2),
                            det.world_yaw])
                confidences.append(confidence)
                ids.append(id)
            elif det.move_coords == 1:
                id = self.name_to_id(det.name)
                if len(det.world_coords) > 0:
                    bboxes.append([
                        det.world_coords[0],
                        det.world_coords[1],
                        det.world_coords[2],
                        0.5,
                        0.5,
                        0.5,
                        det.world_yaw
                    ])
                    confidences.append(det.extra[0] if len(det.extra)>0 else 0.5)
                    ids.append(id)

                rays_2d.append(det.object_ray)
                confidences_2d.append(det.extra[0] if len(det.extra)>0 else 0.5)
                ids_2d.append(id)

        t1 = time.time()
        if len(ids) > 0:
            # self.get_logger().info("process 3d")
            self.tracker.update(np.array(bboxes),
                                np.array(confidences),
                                np.array(ids))
        # self.get_logger().info(f"update_3d: {time.time() - t1}")
        t2 = time.time()
        if len(ids_2d) > 0:
            # self.get_logger().info("process 2d")
            self.tracker.update_2d(np.array(rays_2d),
                                   np.array(confidences_2d),
                                   np.array(ids_2d))
        # self.get_logger().info(f"update_2d: {time.time() - t2}")

        # Publish tracked objects
        output = DetectedObjectsStamped()
        output.header = msg.header

        markers = MarkerArray()
        header = Header()
        header.frame_id = "world_ned"
        header.stamp = msg.header.stamp
        marker = Marker()
        marker.action = Marker.DELETEALL
        marker.header = header
        marker.id = 0
        markers.markers.append(marker)
        t3 = time.time()
        for i, (id, track) in enumerate(self.tracker.tracks.items()):
            self.track_counts[track.id][track.class_id]+=1

            # print(track.id, track.age)

            if track.age <= self.min_age or len(track.identities)==0:
                continue

            max_id, conf = max(track.identities.items(), key=lambda x: x[1])
            if conf < 0.2:
                self.get_logger().info("conf too low, skipping")
                continue
            conf /= track.identities_count[max_id]
            last_point = track.track_hist[-1]
            tracked_obj_msg = DetectedObject()
            tracked_obj_msg.name = self.id_to_name(max_id)
            tracked_obj_msg.world_coords = [last_point[0],
                                            last_point[1],  last_point[2]]
            tracked_obj_msg.tracker_confidence = [int(conf * 100)]
            tracked_obj_msg.tracker_match_id = [int(track.id)]
            tracked_obj_msg.real_dims = [max(0.01, track.bbox[3]), max(0.01, track.bbox[4]), max(0.01, track.bbox[5])]
            tracked_obj_msg.world_yaw = track.bbox[6]
            tracked_obj_msg.extra = [int(track.frame_id)]
            tracked_obj_msg.move_coords = 2
            tracked_obj_msg.header = output.header
            output.detected.append(tracked_obj_msg)
            # frame, id, bb_left, bb_top, bb_width, bb_height, confidence, class_id, trunc, occ  = track
            # tracked_obj_msg = DetectedObject()
            # tracked_obj_msg.name = self.id_to_name[class_id]
            # tracked_obj_msg.world_coords = [bb_left + bb_width/2,
            #                                 bb_top + bb_height/2, 0.0]
            # tracked_obj_msg.tracker_confidence = [int(confidence)]
            # tracked_obj_msg.extra = [int(frame)]
            # output.detected.append(tracked_obj_msg)
            marker = Marker()
            marker.header = header
            marker.id = i + 1
            marker.action = Marker.ADD
            marker.type = Marker.LINE_STRIP
            marker.scale.x = 0.1
            marker.scale.y = 0.0
            marker.scale.z = 0.0
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 1.0
            marker.lifetime.sec = 5
            for point in track.track_hist:
                marker.points.append(Point(x=point[0], y=point[1], z=point[2]))
            markers.markers.append(marker)
        # self.get_logger().info(f"generate_tracks: {time.time() - t3}")
        self.tracked_objects_pub.publish(output)
        self.tracks_vis_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = LidarTrackerFilter()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
