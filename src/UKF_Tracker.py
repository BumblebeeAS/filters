from copy import deepcopy
from datetime import datetime, timedelta
import time
from operator import attrgetter

import matplotlib.markers
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from stonesoup.dataassociator.neighbour import NearestNeighbour, GlobalNearestNeighbour, GNNWith2DAssignment
from stonesoup.dataassociator.probability import PDA, JPDA
from stonesoup.deleter.time import UpdateTimeDeleter
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.deleter.multi import CompositeDeleter
from stonesoup.functions import gm_reduce_single
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.initiator.wrapper import StatesLengthLimiter
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, RandomWalk
import numpy as np
from stonesoup.predictor.kalman import UnscentedKalmanPredictor
from stonesoup.types.array import StateVector, StateVectors
from stonesoup.types.detection import Detection
from stonesoup.types.state import GaussianState
from stonesoup.types.update import GaussianStateUpdate
from stonesoup.updater.kalman import UnscentedKalmanUpdater
import rospy
from geometry_msgs.msg import Vector3, Quaternion, Point
from bb_msgs.msg import DetectedObject, DetectedObjects
from stonesoup.plotter import Plotter
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Quaternion, TransformStamped
import tf2_ros
from transforms3d.euler import euler2quat

class UKF_Tracker():

    def __init__(self, node_name: str) -> None:
        rospy.init_node(node_name)

        rospy.loginfo("Starting: " + node_name)
        rospy.set_param("filtered_topic",
                        "/auv4/vision/external/fused_detections")

        # Object Movement Model
        self.transition_model = CombinedLinearGaussianTransitionModel(
            [
                RandomWalk(0.1),  # x
                RandomWalk(0.1),  # y
                RandomWalk(0.1)  # z
            ]
        )

        # Object Detection Sensor Model
        self.measurement_model = LinearGaussian(
            ndim_state=3,
            mapping=[0, 1, 2],
            noise_covar=np.diag([0.5, 0.5, 0.1])
        )

        # UKF Predictor
        self.predictor = UnscentedKalmanPredictor(self.transition_model)

        # UKF Updater
        self.updater = UnscentedKalmanUpdater(self.measurement_model)

        # Track Deleter
        covar_deleter = CovarianceBasedDeleter(
            covar_trace_thresh=1.5)  # Higher Value Longer Persistence
        time_deleter = UpdateTimeDeleter(
            timedelta(seconds=30))  # Time Since Last Update
        # Any fail will cause deletion
        self.deleter = CompositeDeleter(
            [covar_deleter, time_deleter], intersect=False)

        # Hypothesiser
        prob_detect = 0.9
        pda_hypothesiser = PDAHypothesiser(predictor=self.predictor,
                                           updater=self.updater,
                                           clutter_spatial_density=0.125,
                                           prob_detect=prob_detect)

        # JPDA Associator
        self.data_associator = JPDA(hypothesiser=pda_hypothesiser)

        # Stricter Hypothesiser for Initiator
        init_hypothesiser = PDAHypothesiser(predictor=self.predictor,
                                            updater=self.updater,
                                            clutter_spatial_density=0.45,
                                            prob_detect=0.9)

        # Detection Initiator
        self.initiator = MultiMeasurementInitiator(
            prior_state=GaussianState([0, 0, 0], np.diag([0.1, 0.1, 0.1])),
            measurement_model=self.measurement_model,
            deleter=self.deleter,
            data_associator=GNNWith2DAssignment(init_hypothesiser),
            updater=self.updater,
            min_points=5
        )

        self.initiator = StatesLengthLimiter(self.initiator, 20)

        # Detections
        self.tracks, self.all_tracks, self.obj_meta = dict(), dict(), dict()

        # Detected Object Sub
        self.detection_sub = rospy.Subscriber(
            "/auv4/vision/external/detected_filtered", DetectedObjects, self.detection_callback)
        
        # Detected Object Pub
        self.detection_pub = rospy.Publisher(
            "/auv4/vision/UKF/detected", 
            DetectedObjects,
            queue_size=10
        )

        # Pub Timer
        self.pub_timer = rospy.Timer(
            rospy.Duration(1),
            self.pub_detections
        )

        # Plotting
        plt.ioff()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot()

        self.plotter = Plotter()
        self.plotCount = 0

        # Plot Pub
        self.publisher_plot = rospy.Publisher(
            "/filter_visualisation/image_raw/compressed", 
            CompressedImage)

        self.bridge = CvBridge()

        self.br = tf2_ros.TransformBroadcaster()

    def detection_callback(self, detections: DetectedObjects):
        # self.get_logger().info("Det")
        start = time.time()
        if len(detections.detected) > 0:
            # Process Detections
            names = set([obj.name for obj in detections.detected])
            for name in names:
                if self.tracks.get(name) is None:
                    rospy.loginfo(
                        "Creating new Track for {} objects".format(name))
                    self.tracks[name] = set()
                    self.all_tracks[name] = set()

            grouped = {x: [y for y in detections.detected if y.name == x]
                       for x in names}
            for name, objects in grouped.items():
                measurement_set = set()

                # Create Detections for Objects
                for obj in objects:
                    coords = np.array(obj.world_coords)
                    if len(coords) != 0:
                        timestamp = datetime.fromtimestamp(
                            obj.header.stamp.to_sec())
                        measurement_set.add(Detection(
                            state_vector=StateVector(coords),
                            timestamp=timestamp,
                            measurement_model=self.measurement_model,
                            metadata={
                                "frame":obj.header.frame_id,
                                "world_yaw":obj.world_yaw
                            }))
                        self.obj_meta[obj.name] = {"frame":obj.header.frame_id, "world_yaw":obj.world_yaw}

                # Process Detections with JPDA UKF
                if len(measurement_set) > 0:
                    hypotheses = self.data_associator.associate(self.tracks[name],
                                                                measurement_set,
                                                                timestamp)

                    associated_measurements = set()
                    for track in self.tracks[name]:
                        track_hypotheses = hypotheses[track]

                        posterior_states = []
                        posterior_state_weights = []
                        for hypothesis in track_hypotheses:
                            if not hypothesis:
                                posterior_states.append(hypothesis.prediction)
                            else:
                                posterior_state = self.updater.update(
                                    hypothesis)
                                posterior_states.append(posterior_state)
                                associated_measurements.add(
                                    hypothesis.measurement)
                            posterior_state_weights.append(
                                hypothesis.probability)

                        means = StateVectors(
                            [state.mean for state in posterior_states])
                        covars = np.stack(
                            [state.covar for state in posterior_states], axis=2)
                        weights = np.asarray(posterior_state_weights)

                        # Reduce mixture of states to one posterior estimate Gaussian.
                        post_mean, post_covar = gm_reduce_single(
                            means, covars, weights)

                        # Add a Gaussian state approximation to the track.
                        track.append(GaussianStateUpdate(
                            post_mean, post_covar,
                            track_hypotheses,
                            track_hypotheses[0].measurement.timestamp))

                        # Deletion and Initiation
                    self.tracks[name] -= self.deleter.delete_tracks(
                        self.tracks[name])
                    self.tracks[name] |= self.initiator.initiate(measurement_set - associated_measurements,
                                                                 timestamp)

                    # self.all_tracks[name] |= self.tracks[name]

            rospy.loginfo("Processed in %.4f seconds" % (time.time() - start))

            plot_marker = None
            plot_color = None

            for name in names:
                for track in self.tracks[name]:
                    coords = track.state_vector.flatten()

                    # Using back of Track id
                    #rospy.loginfo(
                    #    f"Tracking {coords} {name}@{track.id.split('-')[-1]}")
                    
                    plot_marker = "^"

                    if "green" in obj.name:
                        plot_color = "green"
                    elif "red" in obj.name:
                        plot_color = "red"
                    elif "black" in obj.name:
                        plot_color = "black"
                    elif "orange" in obj.name:
                        plot_color = "orange"
                    elif "white" in obj.name:
                        plot_color = "white"
                    elif "rgb" in obj.name:
                        plot_color = "blue"
                    else:
                        plot_color = "gray"
                    # Plot on 2d map
                    self.draw_uncertainty(
                        track, [0, 1], self.ax, plot_color, plot_marker, obj.name)

            # Prepare Plot Image
            self.fig.canvas.draw()
            graph_image = np.array(self.fig.canvas.get_renderer()._renderer)
            img_msg = self.bridge.cv2_to_compressed_imgmsg(graph_image)
            self.publisher_plot.publish(img_msg)
            self.ax.cla()

    def pub_detections(self, timer):
        pub_objects = DetectedObjects()
        objs = []

        for name, tracks_by_class in self.tracks.items():
            # Clear Old
            self.tracks[name] -= self.deleter.delete_tracks(
                self.tracks[name])
            for track in tracks_by_class:
                try:
                    tf_msg = TransformStamped()
                    tf_msg.header.stamp = rospy.Time.now()
                    tf_msg.header.frame_id = "map_ned"  # Assuming the map frame as reference
                    tf_msg.child_frame_id = (
                        f"{name}/ukf_tracker"  # Replace with desired TF frame ID
                    )
                    coords = track.state_vector.flatten()
                    tf_msg.transform.translation.x = coords[0]
                    tf_msg.transform.translation.y = coords[1]
                    tf_msg.transform.translation.z = coords[2]
                    w, x, y, z = euler2quat(0, 0, np.deg2rad(self.obj_meta[name]["world_yaw"]))
                    tf_msg.transform.rotation = Quaternion(x, y, z, w)
                    self.br.sendTransform(tf_msg)

                    obj = DetectedObject()
                    obj.header.frame_id = self.obj_meta[name]["frame"]
                    obj.header.stamp.from_sec(track.timestamp.timestamp())
                    obj.world_coords = track.state_vector.flatten()
                    obj.name = name
                    objs.append(obj)

                    # Using back of Track id
                    rospy.loginfo(
                        f"Tracking {coords} {name}@{track.id.split('-')[-1]}")
                except:
                    print(track)
                    continue
                break
        rospy.loginfo(f"Publishing {len(objs)} Objects")
        pub_objects.detected = objs
        self.detection_pub.publish(pub_objects)
                

    def draw_uncertainty(self, track, mapping, ax, color, marker, name):
        state, data, min_ind, max_ind, orient, w, v = self.calculate_uncertainty(
            track, mapping)
        ax.scatter(data[0], data[1], color=color, marker=marker)
        ax.text(data[0], data[1], name + "@" + track.id.split('-')[-1], fontsize="x-small")
        ellipse = Ellipse(xy=state.mean[mapping[:2], 0],
                          width=2 * np.sqrt(w[max_ind]),
                          height=2 * np.sqrt(w[min_ind]),
                          angle=np.rad2deg(orient), alpha=0.5,
                          color=color)
        ax.add_patch(ellipse)

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
    # rclpy.init(args=args)

    ukf_tracker = UKF_Tracker(node_name="ukf_tracker")
    rospy.spin()

    # rclpy.shutdown()


if __name__ == '__main__':
    main()
