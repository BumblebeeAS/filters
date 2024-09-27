# input
# colour to encircle (ie. colour to filter)
# decide clockwise or anticlockwise
# filter for coloured bouy

# output
# pub to ros topic with the pose of the  filtered bouy and the direction to encircle


from rclpy.node import Node
import rclpy
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from bb_perception_msgs.msg import (
    DetectedObject3DArray,
)
from ml_detector.schema_validator import get_config, load_schema


class EncirclementTask(Node):
    def __init__(self):
        super().__init__('encirclement_task')
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
        self.green_buoy_id = self.name_to_id["green_cylinder"]
        self.red_buoy_id = self.name_to_id["red_cylinder"]
        self.unknown_id = self.name_to_id["unknown"]
        self.gate_id = self.name_to_id["gate"]
        self.subscription = self.create_subscription(
            DetectedObject3DArray,
            # "/asv4/vision/lidar_small_objects/dets_3d/labelled",
            "/asv4/vision/detections_2d/projected/filtered",
            self.detected_objects_callback,
            10,
        )

    def detected_objects_callback(self, msg):
        self.buoys = {}
        if len(msg.objects) != 0:
            self.is_ned = msg.objects[0].hypothesis.kinematics.header.frame_id.endswith(
                "ned"
            )
            self.header = msg.objects[0].hypothesis.kinematics.header
        for det in msg.objects:
            is_green_red_buoy = (
                det.hypothesis.class_id == self.red_buoy_id
                # or det.hypothesis.class_id == self.green_buoy_id
                # or det.hypothesis.class_id == self.unknown_id
            )
            if not is_green_red_buoy:
                continue
            print(f"det.hypothesis.track_id: {det.hypothesis.class_id}")
            pose = det.hypothesis.kinematics.pose_with_covariance.pose
            print(f"pose: {pose}")
            self.buoys[det.hypothesis.track_id] = [
                pose.position.x,
                pose.position.y,
                [0, 0],  # red
            ]
            for class_ in det.hypothesis.classes:
                if class_.class_id == self.red_buoy_id:  # red
                    self.buoys[det.hypothesis.track_id][2][0] += class_.score
                elif class_.class_id == self.green_buoy_id:  # green
                    self.buoys[det.hypothesis.track_id][2][1] += class_.score
                else:
                    print(f"Unknown class id: {class_.class_id}")


def main(args=None):
    rclpy.init(args=args)
    prequali_detection = EncirclementTask()
    rclpy.spin(prequali_detection)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
