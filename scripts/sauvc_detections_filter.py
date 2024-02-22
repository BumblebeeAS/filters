import rospy
import traceback
import sys

from bb_msgs.msg import DetectedObjects
from sensor_msgs.msg import CameraInfo
from importlib import import_module
from tf2_ros import Buffer, TransformListener
from bb_filters.utils import *
from bb_filters.filter import Filter, CameraInfos
from typing import List
class SauvcDetectionsFilter:
    def __init__(self):
        self.NODE_NAME = "sauvc_detections_filter"
        rospy.init_node(self.NODE_NAME)
        self.filters: List[Filter]
        self.processed_detections_pub = rospy.Publisher(
            "/auv4/vision/external/detected_filtered",
            DetectedObjects
        )
        self.buffer = Buffer(rospy.Duration(10))
        self.listener = TransformListener(self.buffer, 10)
        self.camera_info = CameraInfos(self.buffer, "map_ned")
        
        self.camera_info_topics = {
            288: "/auv4/front_cam/camera_info",
            289: "/auv4/bot_cam/camera_info",
        }
        for id, topic in self.camera_info_topics.items():
            msg = rospy.wait_for_message(topic, CameraInfo)
            self.camera_info.set_info(id, msg)
            print(id, msg)
        self.import_modules()

        self.init_filters()
        self.raw_detections_sub = rospy.Subscriber(
            "/auv4/vision/external/detected",
            DetectedObjects,
            self.process,
            queue_size=10
        )
    
    def process(self, detected: DetectedObjects):
        output = DetectedObjects()
        for filter in self.filters:
            try:
                result = filter.process(detected).detected
                output.detected.extend(result)
            except Exception as e:
                traceback.print_exc(file=sys.stdout)
                rospy.logerr(f"Error processing {filter.__name__}: {e}")
        output.node_name = self.NODE_NAME
        self.processed_detections_pub.publish(output)

    #--------------------#
    #    Init Helpers    #
    #--------------------#

    def import_modules(self):
        self.ns = rospy.get_namespace().split('/')[1]
        import_module('bb_filters.sauvc_objects', package=__name__)

    def init_filters(self):
        self.filters = []
        self.configs = get_config_files("bb_filters", "sauvc_objects")
        for config_file in self.configs:
            config = get_config(config_file, "bb_filters", "sauvc_objects")
            self.filters.append(
                getattr(import_module(
                    f"bb_filters.sauvc_objects.{config_file.split('.')[0]}"),
                    "Filter")(config, self.camera_info))

def main():
    detector = SauvcDetectionsFilter()
    rospy.spin()

if __name__ == "__main__":
    main()