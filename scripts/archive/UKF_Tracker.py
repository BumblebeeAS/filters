from copy import deepcopy
from datetime import datetime, timedelta
import time
from operator import attrgetter

import matplotlib.markers
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from stonesoup.dataassociator.neighbour import (
    NearestNeighbour,
    GlobalNearestNeighbour,
    GNNWith2DAssignment,
)
from stonesoup.dataassociator.probability import PDA, JPDA
from stonesoup.deleter.time import UpdateTimeDeleter
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.deleter.multi import CompositeDeleter
from stonesoup.functions import gm_reduce_single
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.initiator.wrapper import StatesLengthLimiter
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    RandomWalk,
)
import numpy as np
from stonesoup.predictor.kalman import UnscentedKalmanPredictor
from stonesoup.types.array import StateVector, StateVectors
from stonesoup.types.detection import Detection
from stonesoup.types.state import GaussianState
from stonesoup.types.update import GaussianStateUpdate
from stonesoup.updater.kalman import UnscentedKalmanUpdater
import rclpy
from geometry_msgs.msg import Vector3, Quaternion, Point
from rclpy.duration import Duration
from rclpy.node import Node
from bb_msgs.msg import DetectedObject, DetectedObjectsStamped
from stonesoup.plotter import Plotter

from std_msgs.msg import ColorRGBA
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray
from tf2_geometry_msgs import PoseStamped as TF2PoseStamped

from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError


class UKF_Tracker(Node):

    def __init__(self, node_name: str) -> None:
        # Object Movement Model
        super().__init__(node_name)
        self.get_logger().info("Starting: " + node_name)
        self.declare_parameter(
            "filtered_topic", "/asv4/vision/external/fused_detections"
        )

        self.transition_model = CombinedLinearGaussianTransitionModel(
            [RandomWalk(0.1), RandomWalk(0.1), RandomWalk(0.1)]  # x  # y  # z
        )

        # Object Detection Sensor Model
        self.measurement_model = LinearGaussian(
            ndim_state=3, mapping=[0, 1, 2], noise_covar=np.diag([0.975, 0.975, 0.1])
        )

        # UKF Predictor
        self.predictor = UnscentedKalmanPredictor(self.transition_model)

        # UKF Updater
        self.updater = UnscentedKalmanUpdater(self.measurement_model)

        # Track Deleter
        covar_deleter = CovarianceBasedDeleter(
            covar_trace_thresh=1.5
        )  # Higher Value Longer Persistence
        time_deleter = UpdateTimeDeleter(timedelta(seconds=5))  # Time Since Last Update
        self.deleter = CompositeDeleter(
            [covar_deleter, time_deleter], intersect=False
        )  # Any fail will cause deletion

        # Hypothesiser
        prob_detect = 0.9
        pda_hypothesiser = PDAHypothesiser(
            predictor=self.predictor,
            updater=self.updater,
            clutter_spatial_density=0.125,
            prob_detect=prob_detect,
        )

        # JPDA Associator
        self.data_associator = JPDA(hypothesiser=pda_hypothesiser)

        # Stricter Hypothesiser for Initiator
        init_hypothesiser = PDAHypothesiser(
            predictor=self.predictor,
            updater=self.updater,
            clutter_spatial_density=0.45,
            prob_detect=0.9,
        )

        # Detection Initiator
        self.initiator = MultiMeasurementInitiator(
            prior_state=GaussianState([0, 0, 0], np.diag([0.1, 0.1, 0.1])),
            measurement_model=self.measurement_model,
            deleter=self.deleter,
            data_associator=GNNWith2DAssignment(init_hypothesiser),
            updater=self.updater,
            min_points=20,
        )

        self.initiator = StatesLengthLimiter(self.initiator, 20)

        # Detections
        self.tracks, self.all_tracks = dict(), dict()

        # Detected Object Sub
        self.detection_sub = self.create_subscription(
            DetectedObjectsStamped,
            "/asv4/vision/lidar/detected_stamped",
            self.detection_callback,
            10,
        )

        plt.ioff()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot()

        self.plotter = Plotter()
        self.plotCount = 0

        # Marker Pub
        self.publisher_ = self.create_publisher(MarkerArray, "/stereo_visualization", 1)

        # Plot Pub
        self.publisher_plot = self.create_publisher(
            CompressedImage, "/filter_visualisation/image_raw/compressed", 10
        )
        self.filtered_topic = (
            self.get_parameter("filtered_topic").get_parameter_value().string_value
        )
        self.tracked_objects_pub = self.create_publisher(
            DetectedObjectsStamped, self.filtered_topic, 10
        )
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.bridge = CvBridge()

    def detection_callback(self, detections: DetectedObjectsStamped):
        # self.get_logger().info("Det")
        start = time.time()
        if len(detections.detected) > 0:
            # Process Detections
            names = set([obj.name for obj in detections.detected])
            for name in names:
                if self.tracks.get(name) is None:
                    self.get_logger().info(
                        "Creating new Track for {} objects".format(name)
                    )
                    self.tracks[name] = set()
                    self.all_tracks[name] = set()

            grouped = {
                x: [y for y in detections.detected if y.name == x] for x in names
            }
            for name, objects in grouped.items():
                measurement_set = set()

                # Create Detections for Objects
                for obj in objects:
                    coords = np.array(obj.world_coords)
                    if len(coords) != 0:
                        timestamp = datetime.fromtimestamp(obj.header.stamp.sec)
                        measurement_set.add(
                            Detection(
                                state_vector=StateVector(coords),
                                timestamp=timestamp,
                                measurement_model=self.measurement_model,
                            )
                        )

                # Process Detections with JPDA UKF
                if len(measurement_set) > 0:
                    hypotheses = self.data_associator.associate(
                        self.tracks[name], measurement_set, timestamp
                    )

                    associated_measurements = set()
                    for track in self.tracks[name]:
                        track_hypotheses = hypotheses[track]

                        posterior_states = []
                        posterior_state_weights = []
                        for hypothesis in track_hypotheses:
                            if not hypothesis:
                                posterior_states.append(hypothesis.prediction)
                            else:
                                posterior_state = self.updater.update(hypothesis)
                                posterior_states.append(posterior_state)
                                associated_measurements.add(hypothesis.measurement)
                            posterior_state_weights.append(hypothesis.probability)

                        means = StateVectors([state.mean for state in posterior_states])
                        covars = np.stack(
                            [state.covar for state in posterior_states], axis=2
                        )
                        weights = np.asarray(posterior_state_weights)

                        # Reduce mixture of states to one posterior estimate Gaussian.
                        post_mean, post_covar = gm_reduce_single(means, covars, weights)

                        # Add a Gaussian state approximation to the track.
                        track.append(
                            GaussianStateUpdate(
                                post_mean,
                                post_covar,
                                track_hypotheses,
                                track_hypotheses[0].measurement.timestamp,
                            )
                        )

                        # Deletion and Initiation
                    self.tracks[name] -= self.deleter.delete_tracks(self.tracks[name])
                    self.tracks[name] |= self.initiator.initiate(
                        measurement_set - associated_measurements, timestamp
                    )

                    # self.all_tracks[name] |= self.tracks[name]

            self.get_logger().info("Processed in %.4f seconds" % (time.time() - start))

            # Plots
            # Marker
            # id=0
            markers = MarkerArray()
            marker = Marker()
            marker.header = obj.header
            marker.scale = Vector3(x=1.0, y=1.0, z=1.0)
            marker.pose.orientation = Quaternion(w=0.707, x=0.0, y=0.0, z=0.0)
            marker.action = Marker.DELETEALL
            markers.markers.append(deepcopy(marker))
            marker.action = Marker.ADD
            plot_marker = None
            plot_color = None
            id = 1

            output = DetectedObjectsStamped()
            output.header = detections.header

            for name in names:
                for track in self.tracks[name]:
                    coords = track.state_vector.flatten()

                    # Using back of Track id
                    self.get_logger().info(
                        f"Tracking {coords} {name}@{track.id.split('-')[-1]}"
                    )

                    color = ColorRGBA()
                    if "round" in name:
                        marker.type = Marker.SPHERE
                        plot_marker = "o"
                    else:
                        marker.type = Marker.CYLINDER
                        plot_marker = "^"

                    if "green" in obj.name:
                        color.r = 0.0
                        color.g = 1.0
                        color.b = 0.0
                        plot_color = "green"
                    elif "red" in obj.name:
                        color.r = 1.0
                        color.g = 0.0
                        color.b = 0.0
                        plot_color = "red"
                    elif "black" in obj.name:
                        color.r = 0.0
                        color.g = 0.0
                        color.b = 0.0
                        plot_color = "black"
                    elif "orange" in obj.name:
                        color.r = 1.0
                        color.g = 0.64
                        color.b = 0.0
                        plot_color = "orange"
                    elif "white" in obj.name:
                        color.r = 1.0
                        color.g = 1.0
                        color.b = 1.0
                        plot_color = "white"
                    elif "rgb" in obj.name:
                        color.r = 0.0
                        color.g = 0.0
                        color.b = 1.0
                        plot_color = "blue"
                    else:
                        color.r = 1.0
                        color.g = 1.0
                        color.b = 1.0
                        plot_color = "gray"
                    color.a = 1.0

                    p = Point()
                    p.x = coords[0]
                    p.y = coords[1]
                    p.z = coords[2]

                    marker.color = color
                    marker.pose.position = p
                    marker.id = id
                    id += 1
                    marker.header.frame_id = "map_ned"

                    tracked_obj_msg = DetectedObject()
                    tracked_obj_msg.name = obj.name
                    tracked_obj_msg.world_coords = [
                        coords[0],
                        coords[1],
                        0.0,
                    ]
                    tracked_obj_msg.real_dims = [0.5, 0.5, 0.5]
                    tracked_obj_msg.move_coords = 2
                    tracked_obj_msg.tracker_confidence = [1]
                    # tracked_obj_msg.extra = [int(track.id.split('-')[-1])]
                    output.detected.append(tracked_obj_msg)

                    # id += 1

                    # TODO verify uncertainty marker
                    #                   uncertainty_marker, text = self.get_uncertainty_marker(track, [0, 1, 2], color)
                    #                   uncertainty_marker.id = id
                    #                   uncertainty_marker.header.frame_id = "map_ned"
                    #                   id += 1
                    #                   text.header.frame_id = "map_ned"
                    #                   text.id = id
                    #                   text.pose.position = marker.pose.position
                    #                   id += 1

                    markers.markers.append(deepcopy(marker))
                    #                    markers.markers.append(deepcopy(uncertainty_marker))
                    #                    markers.markers.append(deepcopy(text))
                    #                    try:
                    #                        tf2_pose = TF2PoseStamped()
                    #                        tf2_pose.header = marker.header
                    #                        tf2_pose.header.frame_id = 'map_ned'
                    #                        tf2_pose.pose = marker.pose
                    #                        target_pose = "asv4/base_link"
                    #                        base_link = self._tf_buffer.transform(
                    #                            tf2_pose, target_pose, Duration(seconds=0.0))
                    #                        # base_link_coords = list(
                    #                        #     attrgetter("x", "y", "z")(base_link.pose.position))
                    #                        marker.pose.position = base_link.pose.position
                    #                        marker.header.frame_id = target_pose
                    #
                    #                        # Plot Uncertainty Markers
                    #                        uncertainty_marker.pose.position = base_link.pose.position
                    #                        uncertainty_marker.header.frame_id = target_pose
                    #                        text.pose.position = base_link.pose.position
                    #                        text.header.frame_id = target_pose
                    #
                    #                        # print(marker)
                    #                        markers.markers.append(deepcopy(marker))
                    #                        markers.markers.append(deepcopy(uncertainty_marker))
                    #                        markers.markers.append(deepcopy(text))
                    #                    except Exception as e:
                    #                        self.get_logger().warn(
                    #                            f"Failed to convert to base_link {e}")
                    #
                    # Plot on 2d map
                    self.draw_uncertainty(
                        track, [0, 1], self.ax, plot_color, plot_marker
                    )

            self.tracked_objects_pub.publish(output)

            # Prepare Plot Image
            self.fig.canvas.draw()
            graph_image = np.array(self.fig.canvas.get_renderer()._renderer)
            img_msg = self.bridge.cv2_to_compressed_imgmsg(graph_image)
            self.publisher_plot.publish(img_msg)
            self.ax.cla()
            # TODO: Add Plot - CV - ROS Pub

            self.publisher_.publish(markers)

    def draw_uncertainty(self, track, mapping, ax, color, marker):
        state, data, min_ind, max_ind, orient, w, v = self.calculate_uncertainty(
            track, mapping
        )
        ax.scatter(data[0], data[1], color=color, marker=marker)
        ax.text(data[0], data[1], track.id.split("-")[-1], fontsize="x-small")
        ellipse = Ellipse(
            xy=state.mean[mapping[:2], 0],
            width=2 * np.sqrt(w[max_ind]),
            height=2 * np.sqrt(w[min_ind]),
            angle=np.rad2deg(orient),
            alpha=0.5,
            color=color,
        )
        ax.add_patch(ellipse)

    def get_uncertainty_marker(self, track, mapping, color):
        state, data, min_ind, max_ind, orient, w, v = self.calculate_uncertainty(
            track, mapping
        )
        marker = Marker()
        marker.scale = Vector3(
            x=2 * np.sqrt(w[max_ind]), y=2 * np.sqrt(w[min_ind]), z=0.0
        )

        # TODO fix angle for marker
        marker.pose.orientation = Quaternion(w=0.707, x=0.707, y=0.0, z=0.0)
        marker.action = Marker.DELETEALL
        marker.type = Marker.SPHERE
        p = Point()
        p.x = state.mean[mapping[0]]
        p.y = state.mean[mapping[1]]
        p.z = state.mean[mapping[2]]
        color.a = 0.5
        marker.color = color
        marker.pose.position = p

        text = Marker(text=track.id.split("-")[-1])
        text.type = Marker.TEXT_VIEW_FACING
        text.pose.position = marker.pose.position
        text.pose.position.z += 1.0
        text.scale.z = 5.0

        return marker, text

    def calculate_uncertainty(self, track, mapping):
        HH = np.eye(track.ndim)[mapping, :]  # Get position mapping matrix
        state = track[-1]
        data = state.state_vector[mapping[:2], :]
        w, v = np.linalg.eig(HH @ state.covar @ HH.T)
        v = np.real_if_close(v, tol=1)
        w = np.real_if_close(w, tol=1)
        max_ind = np.argmax(w)
        min_ind = np.argmin(w)
        try:
            orient = np.arctan2(v[1, max_ind], v[0, max_ind])
        except TypeError:
            orient = np.array([0])
        return state, data, min_ind, max_ind, orient, w, v


def main(args=None):
    rclpy.init(args=args)

    ukf_tracker = UKF_Tracker(node_name="ukf_tracker")
    rclpy.spin(ukf_tracker)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
