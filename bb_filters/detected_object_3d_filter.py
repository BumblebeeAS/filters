import rclpy
from rclpy.node import Node
from collections import defaultdict
import numpy as np
from bb_msgs.msg import DetectedObject, DetectedObjectsStamped
from motrackers import IOUTracker, SORT



class TrackerFilter(Node):
    def __init__(self):
        super().__init__('motracker_iou_tracker_node')
        self.declare_parameter("raw_topic", "/wamv/vision/external/detected_stamped")
        self.declare_parameter("filtered_topic", "/wamv/vision/fused_detections")
        self.raw_topic = self.get_parameter("raw_topic")\
                                        .get_parameter_value().string_value
        self.filtered_topic = self.get_parameter("filtered_topic")\
                                        .get_parameter_value().string_value
        self.tracked_objects_pub = self.create_publisher(
            DetectedObjectsStamped, self.filtered_topic, 10)
        self.object_list_sub = self.create_subscription(
            DetectedObjectsStamped, self.raw_topic,
            self.object_list_callback, 10)
        self.tracker = SORT(
            max_lost=10,
            # min_detection_confidence=0.2,
            # max_detection_confidence=1.0,
            iou_threshold=0.001,
            tracker_output_format="visdrone_challenge")
        self.latest_header = None
        self.min_age = 1
        self.track_counts = defaultdict(lambda: np.zeros(6))

    @property
    def name_to_id(self):
        return {
            "mb_round_buoy_black": 0,
            "mb_round_buoy_orange": 1,
            "mb_marker_buoy_red": 2,
            "mb_marker_buoy_green": 3,
            "mb_marker_buoy_black": 4,
            "mb_marker_buoy_white": 5,
        }

    @property
    def id_to_name(self):
        return {v: k for k, v in self.name_to_id.items()}

    def object_list_callback(self, msg: DetectedObjectsStamped):
        self.latest_header = msg.header
        bboxes, confidences, ids = [], [], []

        for det in msg.detected:
            if len(det.world_coords) == 0:
                continue
            if det.name not in self.name_to_id.keys():
                continue
            bboxes.append(
                [
                    det.world_coords[0] - 0.2,
                    det.world_coords[1] - 0.2,
                    det.world_coords[0] + 0.2,
                    det.world_coords[1] + 0.2,
                ]
            )
            confidences.append(det.extra[0])
            ids.append(self.name_to_id[det.name])

        tracked_objects = self.tracker.update(
            np.array(bboxes), np.array(confidences), np.array(ids)
        )

        # Publish tracked objects
        output = DetectedObjectsStamped()
        output.header = msg.header

        for track in tracked_objects:
            (
                frame,
                id,
                bb_left,
                bb_top,
                bb_width,
                bb_height,
                confidence,
                class_id,
                trunc,
                occ,
            ) = track
            tracked_obj_msg = DetectedObject()
            tracked_obj_msg.name = self.id_to_name[class_id]
            tracked_obj_msg.world_coords = [
                bb_left + bb_width / 2,
                bb_top + bb_height / 2,
                0.0,
            ]
            tracked_obj_msg.tracker_confidence = [int(confidence)]
            tracked_obj_msg.extra = [int(frame)]
            output.detected.append(tracked_obj_msg)
        self.tracked_objects_pub.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = TrackerFilter()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
