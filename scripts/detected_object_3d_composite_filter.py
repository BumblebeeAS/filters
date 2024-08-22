#!/usr/bin/env python3
from pathlib import Path
from operator import attrgetter
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from bb_perception_msgs.msg import (
    DetectedObject3D,
    DetectedObject3DArray,
)
from geometry_msgs.msg import Vector3, Quaternion, Pose
from ml_detector.schema_validator import get_config, load_schema
from datetime import datetime, timedelta
from stonesoup.dataassociator.neighbour import (
    GNNWith2DAssignment,
)
from rclpy.node import Node
from transforms3d.euler import quat2euler, euler2quat
from stonesoup.types.detection import (
    Detection,
    CompositeDetection,
    CategoricalDetection,
)
from stonesoup.updater.categorical import HMMUpdater
from stonesoup.predictor.categorical import HMMPredictor
from stonesoup.updater.composite import CompositeUpdater
from stonesoup.initiator.simple import SimpleMeasurementInitiator
from stonesoup.initiator.categorical import SimpleCategoricalMeasurementInitiator
from stonesoup.initiator.composite import CompositeUpdateInitiator
from stonesoup.models.measurement.categorical import MarkovianMeasurementModel
from stonesoup.types.state import GaussianState, CategoricalState
from stonesoup.dataassociator.probability import JPDA
from stonesoup.deleter.time import UpdateTimeDeleter
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.deleter.multi import CompositeDeleter
from stonesoup.initiator.wrapper import StatesLengthLimiter
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    RandomWalk,
)
from stonesoup.hypothesiser.categorical import HMMHypothesiser
from stonesoup.hypothesiser.composite import CompositeHypothesiser
from stonesoup.predictor.kalman import UnscentedKalmanPredictor
from stonesoup.updater.kalman import UnscentedKalmanUpdater
from stonesoup.initiator.composite import CompositeUpdateInitiator
from stonesoup.tracker.simple import MultiTargetTracker
from stonesoup.types.groundtruth import CategoricalGroundTruthState
from stonesoup.models.transition.categorical import MarkovianTransitionModel
from stonesoup.measures import Mahalanobis
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.types.hypothesis import SingleProbabilityHypothesis
from stonesoup.types.multihypothesis import MultipleHypothesis


class ProbabilityHypothesiser(DistanceHypothesiser):
    def hypothesise(self, track, detections, timestamp, **kwargs):
        multi_hypothesis = super().hypothesise(track, detections, timestamp, **kwargs)
        single_hypotheses = multi_hypothesis.single_hypotheses
        prob_single_hypotheses = list()
        for hypothesis in single_hypotheses:
            prob_hypothesis = SingleProbabilityHypothesis(
                hypothesis.prediction,
                hypothesis.measurement,
                1 / hypothesis.distance,
                hypothesis.measurement_prediction,
            )
            prob_single_hypotheses.append(prob_hypothesis)
        return MultipleHypothesis(
            prob_single_hypotheses, normalise=False, total_weight=1
        )


class CompositeTracker(Node):
    def __init__(self):
        super().__init__("composite_tracker_node")
        # self.declare_parameter("dets_3d_topic", "/robotx/detections")
        # self.declare_parameter("filtered_topic", "/robotx/detections/filtered")
        self.declare_parameter("dets_3d_topic", "/asv4/vision/lidar_small_objects/dets_3d")
        self.declare_parameter("filtered_topic", "/asv4/vision/lidar_small_objects/dets_3d/filtered")
        self.declare_parameter("objects_config", "robotx.yaml")
        objects_schema_path = (
            Path(get_package_share_directory("ml_detector"))
            / "configs"
            / "objects_schema.json"
        )
        self.objects_schema = load_schema(objects_schema_path)
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

        self.dets_3d_topic = (
            self.get_parameter("dets_3d_topic").get_parameter_value().string_value
        )
        self.filtered_topic = (
            self.get_parameter("filtered_topic").get_parameter_value().string_value
        )
        self.tracked_objects_pub = self.create_publisher(
            DetectedObject3DArray, self.filtered_topic, 10
        )

        self.min_prob = 0.7

        self.transition_model = CombinedLinearGaussianTransitionModel(
            [
                RandomWalk(0.7),
                RandomWalk(0.7),
                RandomWalk(0.7),
                RandomWalk(0.7),
                RandomWalk(0.7),
                RandomWalk(0.7),
                RandomWalk(0.7),
            ]  # x   # y  # yaw
        )

        # Object Detection Sensor Model
        self.measurement_model = LinearGaussian(
            ndim_state=7,
            mapping=[0, 1, 2, 3, 4, 5, 6],
            noise_covar=np.diag([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
        )

        self.hidden_classes = [
            "unknown",
            "red_cylinder",
            "green_cylinder",
            "black_sphere",
            "dock",
            "light_tower",
        ]
        self.class_map = {
            self.name_to_id[i]: j for j, i in enumerate(self.hidden_classes)
        }
        self.inverse_class_map = {v: k for k, v in self.class_map.items()}

        self.category_measurement_model = MarkovianMeasurementModel(
            emission_matrix=np.array(
                [
                    [0.5, 0.1, 0.1, 0.1, 0.1, 0.1],
                    [0.1, 0.8, 0.05, 0.05, 0.05, 0.05],
                    [0.1, 0.05, 0.8, 0.05, 0.05, 0.05],
                    [0.1, 0.05, 0.05, 0.8, 0.05, 0.05],
                    [0.1, 0.05, 0.05, 0.05, 0.8, 0.05],
                    [0.1, 0.05, 0.04, 0.04, 0.04, 0.8],
                ]
            ),
            measurement_categories=self.hidden_classes,
        )

        # gt_kwargs = {"timestamp": 0, "categories": self.hidden_classes}
        # self.category_state1 = CategoricalGroundTruthState([1, 0, 0, 0, 0], **gt_kwargs)
        # self.category_state2 = CategoricalGroundTruthState([0, 1, 0, 0, 0], **gt_kwargs)
        # self.category_state3 = CategoricalGroundTruthState([0, 0, 1, 0, 0], **gt_kwargs)
        self.category_transition = MarkovianTransitionModel(transition_matrix=np.eye(6))
        # UKF Predictor
        self.kinematic_predictor = UnscentedKalmanPredictor(self.transition_model)

        # UKF Updater
        self.kinematic_updater = UnscentedKalmanUpdater(self.measurement_model)
        self.category_updater = HMMUpdater(self.category_measurement_model)
        self.updater = CompositeUpdater(
            sub_updaters=[self.kinematic_updater, self.category_updater]
        )
        # Track Deleter # doesn't work with category
        # covar_deleter = CovarianceBasedDeleter(
        #     covar_trace_thresh=1.5
        # )  # Higher Value Longer Persistence

        self.time_deleter = UpdateTimeDeleter(
            timedelta(seconds=5)
        )  # Time Since Last Update
        self.deleter = CompositeDeleter(
            [self.time_deleter], intersect=False
        )  # Any fail will cause deletion

        # Hypothesiser
        # prob_detect = 0.9
        # self.pda_hypothesiser = PDAHypothesiser(
        #     predictor=self.kinematic_predictor,
        #     updater=self.kinematic_updater,
        #     clutter_spatial_density=0.125,
        #     prob_detect=prob_detect,
        # )
        self.kinematic_hypothesiser = ProbabilityHypothesiser(
            predictor=self.kinematic_predictor,
            updater=self.kinematic_updater,
            measure=Mahalanobis(),
        )
        self.category_predictor = HMMPredictor(self.category_transition)
        self.category_hypothesiser = HMMHypothesiser(
            predictor=self.category_predictor, updater=self.category_updater
        )
        self.composite_hypothesiser = CompositeHypothesiser(
            sub_hypothesisers=[self.kinematic_hypothesiser, self.category_hypothesiser]
        )

        # JPDA Associator
        # self.data_associator = JPDA(hypothesiser=self.composite_hypothesiser)
        self.associator = GNNWith2DAssignment(self.composite_hypothesiser)

        # Detection Initiator
        self.kinematic_prior = GaussianState(
            [0, 0, 0, 0.1, 0.1, 0.1, 0], np.diag([0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5])
        )

        self.categorical_prior = CategoricalState(
            [1 / len(self.hidden_classes)] * len(self.hidden_classes),
            categories=self.hidden_classes,
        )
        self.kinematic_initiator = SimpleMeasurementInitiator(
            prior_state=self.kinematic_prior, measurement_model=None
        )
        self.category_initiator = SimpleCategoricalMeasurementInitiator(
            prior_state=self.categorical_prior, updater=self.category_updater
        )

        self.data_associator = GNNWith2DAssignment(self.composite_hypothesiser)
        # self.initiator = MultiMeasurementInitiator(
        #     prior_state=CompositeState([self.kinematic_prior, self.categorical_prior]),
        #     measurement_model=self.measurement_model,
        #     deleter=self.deleter,
        #     data_associator=self.data_associator,
        #     updater=self.updater,
        #     min_points=1,
        #     initiator=CompositeUpdateInitiator(
        #         sub_initiators=[self.kinematic_initiator, self.category_initiator]
        #     ),
        # )

        self.initiator = CompositeUpdateInitiator(
            [self.kinematic_initiator, self.category_initiator]
        )

        self.initiator = StatesLengthLimiter(self.initiator, 100)
        self.all_measurements = list()
        self.tracker = MultiTargetTracker(
            self.initiator,
            self.deleter,
            self.all_measurements,
            self.data_associator,
            self.updater,
        )

        # Detections # dict[track id, tracks]
        self.tracks, self.all_tracks = set(), set()
        self.latest_header = None

        self.detections = []
        self.classes = []
        self.stamps = []

        self.detection_sub = self.create_subscription(
            DetectedObject3DArray,
            self.dets_3d_topic,
            self.detection_callback,
            10,
        )
        self.timer = self.create_timer(0.2, self.update_callback)

    def update_callback(self):
        if self.latest_header is None:
            return
        self.all_measurements.extend(self.detections)
        self.tracker.update_tracker(
            datetime.fromtimestamp(self.latest_header.stamp.sec),
            set(self.all_measurements),
        )

        current_tracks = set()
        current_tracks |= self.tracker.tracks

        # kinematics: list(current_tracks)[0].state.hypothesis.sub_hypotheses[0]
        # category: list(current_tracks)[0].state.hypothesis.sub_hypotheses[1]
        # track id: list(current_tracks)[0].id # uuid string, set detection.id
        detections = DetectedObject3DArray()
        for track in list(current_tracks):
            detection = DetectedObject3D()
            detection.id = track.id
            detection.hypothesis.class_id = self.inverse_class_map[
                np.argmax(track.state.sub_states[1].state_vector)
            ]
            pose = Pose()
            pose.position.x = track.state.sub_states[0].state_vector[0]
            pose.position.y = track.state.sub_states[0].state_vector[1]
            pose.position.z = track.state.sub_states[0].state_vector[2]
            q = euler2quat(0, 0, track.state.sub_states[0].state_vector[6])
            pose.orientation = Quaternion(**dict(zip("wxyz", q)))
            detection.hypothesis.kinematics.pose_with_covariance.pose = pose
            detection.hypothesis.kinematics.header = self.latest_header
            dimensions = Vector3(
                x=track.state.sub_states[0].state_vector[3],
                y=track.state.sub_states[0].state_vector[4],
                z=track.state.sub_states[0].state_vector[5],
            )
            detection.hypothesis.shape.dimensions = dimensions
            # self.get_logger().info(f"({track.state.sub_states[0].covar.diagonal()})")
            detections.objects.append(detection)

        detections.header = self.latest_header
        self.tracked_objects_pub.publish(detections)

    def detection_callback(self, msg: DetectedObject3DArray):
        self.latest_header = msg.header
        if len(msg.objects) > 0:
            self.latest_header = msg.objects[0].hypothesis.kinematics.header
        valid_objects = 0
        for det in msg.objects:
            if det.hypothesis.class_id not in self.class_map:
                continue
            valid_objects += 1
            pose = det.hypothesis.kinematics.pose_with_covariance.pose
            x = pose.position.x
            y = pose.position.y
            z = pose.position.z
            dx = det.hypothesis.shape.dimensions.x
            dy = det.hypothesis.shape.dimensions.y
            dz = det.hypothesis.shape.dimensions.z
            yaw = quat2euler(attrgetter("w", "x", "y", "z")(pose.orientation))[2]
            # self.detections.append([x, y, z, dx, dy, dz, yaw])

            category = np.zeros(len(self.class_map))
            category[self.class_map[det.hypothesis.class_id]] = 1


            prob = det.hypothesis.probability
            if prob < self.min_prob:
                prob = 0.6
                # continue
            # TODO: initialize category probabilities based on classes
            self.detections.append(
                CompositeDetection(
                    sub_states=[
                        Detection(
                            [x, y, z, dx, dy, dz, yaw],
                            measurement_model=self.measurement_model,
                        ),
                        CategoricalDetection(
                            category, measurement_model=self.category_measurement_model
                        ),
                    ],
                    default_timestamp=datetime.fromtimestamp(
                        self.latest_header.stamp.sec
                    ),
                )
            )

        self.stamps.extend([self.latest_header.stamp.sec] * valid_objects)


def main(args=None):
    rclpy.init(args=args)
    composite_tracker = CompositeTracker()
    rclpy.spin(composite_tracker)
    composite_tracker.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
