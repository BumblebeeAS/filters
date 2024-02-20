import numpy as np
import time
from collections import Counter, defaultdict
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from motrackers.utils.misc import iou
from motrackers.track import Track
from motrackers.kalman_tracker import KFTrackerConstantAcceleration
# from motrackers.sort_tracker import assign_tracks2detection_iou
from motrackers.centroid_kf_tracker import CentroidKF_Tracker
from collections import OrderedDict
from scipy.spatial import distance


def iou_xywh(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
    Source: https://github.com/bochinski/iou-tracker/blob/master/util.py

    Args:
        bbox1 (numpy.array or list[floats]): bounding box of length 4 containing ``(x, y, z, dx, dy, dz, yaw)``.
        bbox2 (numpy.array or list[floats]): bounding box of length 4 containing ``(x, y, z, dx, dy, dz, yaw)``.

    Returns:
        float: intersection-over-onion of bbox1, bbox2.
    """
    b1 = bbox1[0]-bbox1[3]*0.5, bbox1[1]-bbox1[4]*0.5, bbox1[0]+bbox1[3]*0.5, bbox1[1]+bbox1[4]*0.5
    b2 = bbox2[0]-bbox2[3]*0.5, bbox2[1]-bbox2[4]*0.5, bbox2[0]+bbox2[3]*0.5, bbox2[1]+bbox2[4]*0.5

    iou_ = iou(b1, b2)

    return iou_


def assign_tracks2detection_iou(bbox_tracks, bbox_detections, dist_threshold=1.5):
    """
    Assigns detected bounding boxes to tracked bounding boxes using IoU as a distance metric.

    Args:
        bbox_tracks (numpy.ndarray): Bounding boxes of shape `(N, 4)` where `N` is number of objects already being tracked.
        bbox_detections (numpy.ndarray): Bounding boxes of shape `(M, 4)` where `M` is number of objects that are newly detected.
        dist_threshold (float): IOU threashold.

    Returns:
        tuple: Tuple contains the following elements in the given order:
            - matches (numpy.ndarray): Array of shape `(n, 2)` where `n` is number of pairs formed after matching tracks to detections. This is an array of tuples with each element as matched pair of indices`(track_index, detection_index)`.
            - unmatched_detections (numpy.ndarray): Array of shape `(m,)` where `m` is number of unmatched detections.
            - unmatched_tracks (numpy.ndarray): Array of shape `(k,)` where `k` is the number of unmatched tracks.
    """

    if (bbox_tracks.size == 0) or (bbox_detections.size == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(bbox_detections), dtype=int), np.empty((0,), dtype=int)

    if len(bbox_tracks.shape) == 1:
        bbox_tracks = bbox_tracks[None, :]

    if len(bbox_detections.shape) == 1:
        bbox_detections = bbox_detections[None, :]

    # iou_matrix = np.zeros((bbox_tracks.shape[0], bbox_detections.shape[0]), dtype=np.float32)
    # for t in range(bbox_tracks.shape[0]):
    #     for d in range(bbox_detections.shape[0]):
    #         iou_matrix[t, d] = iou_xywh(bbox_tracks[t, :], bbox_detections[d, :])
    dist_matrix = cdist(bbox_tracks, bbox_detections)

    assigned_tracks, assigned_detections = linear_sum_assignment(dist_matrix)
    unmatched_detections, unmatched_tracks = [], []

    for d in range(bbox_detections.shape[0]):
        if d not in assigned_detections:
            unmatched_detections.append(d)

    for t in range(bbox_tracks.shape[0]):
        if t not in assigned_tracks:
            unmatched_tracks.append(t)

    # filter out matched with low IOU
    matches = []
    for t, d in zip(assigned_tracks, assigned_detections):
        if dist_matrix[t, d] > dist_threshold:
            unmatched_detections.append(d)
            unmatched_tracks.append(t)
        else:
            matches.append((t, d))

    if len(matches):
        matches = np.array(matches)
    else:
        matches = np.empty((0, 2), dtype=int)

    return matches, np.array(unmatched_detections), np.array(unmatched_tracks)

class KFTracker7D(KFTrackerConstantAcceleration):
    def __init__(self, initial_measurement=np.array([0., 0., 0., 0., 0., 0., 0.]),
                 time_step=1, process_noise_scale=1.0,
                 measurement_noise_scale=1.0):
        assert initial_measurement.shape[0] == 7, initial_measurement.shape
        super().__init__(
            initial_measurement=initial_measurement, time_step=time_step, process_noise_scale=process_noise_scale,
            measurement_noise_scale=measurement_noise_scale
        )

class Track3D:
    """
    Track containing attributes to track various objects.

    Args:
        frame_id (int): Camera frame id.
        track_id (int): Track Id
        bbox (numpy.ndarray): Bounding box pixel coordinates as (x, y, z, dx, dy, dz, yaw) of the track.
        detection_confidence (float): Detection confidence of the object (probability).
        class_id (str or int): Class label id.
        lost (int): Number of times the object or track was not tracked by tracker in consecutive frames.
        iou_score (float): Intersection over union score.
        kwargs (dict): Additional key word arguments.

    """

    count = 0

    def __init__(
        self,
        track_id,
        frame_id,
        bbox,
        detection_confidence,
        class_id=None,
        lost=0,
        iou_score=0.,
        **kwargs
    ):
        Track.count += 1
        self.id = track_id

        self.detection_confidence_max = 0.
        self.lost = 0
        self.age = 0

        self.identities = defaultdict(float)
        self.identities_count = defaultdict(int)
        self.track_hist = []

        self.update(frame_id, bbox, detection_confidence, class_id=class_id, lost=lost, iou_score=iou_score, **kwargs)

        self.output = self.get_output

    def update(self, frame_id, bbox, detection_confidence, class_id=None, lost=0, iou_score=0., **kwargs):
        """
        Update the track.

        Args:
            frame_id (int): Camera frame id.
            bbox (numpy.ndarray): Bounding box pixel coordinates as (x, y, z, dx, dy, dz, yaw) of the track.
            detection_confidence (float): Detection confidence of the object (probability).
            class_id (int or str): Class label id.
            lost (int): Number of times the object or track was not tracked by tracker in consecutive frames.
            iou_score (float): Intersection over union score.
            kwargs (dict): Additional key word arguments.
        """
        self.class_id = class_id
        self.bbox = np.array(bbox)
        self.detection_confidence = detection_confidence
        self.frame_id = frame_id
        self.iou_score = iou_score

        if lost == 0:
            self.lost = 0
        else:
            self.lost += lost

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.detection_confidence_max = max(self.detection_confidence_max, detection_confidence)
        self.track_hist.append((bbox[0], bbox[1], bbox[2]))

        self.age += 1

    def update_2d(self, detection_confidence, class_id=None, distance=100, **kwargs):
        for c in self.identities.keys():
            self.identities[c]*=0.8
            self.identities_count[c]*=0.8
        self.identities[class_id] += detection_confidence / 100 * min(1, 1/(distance+1e-9))
        self.identities_count[class_id] += 1

    @property
    def centroid(self):
        """
        Return the centroid of the bounding box.

        Returns:
            numpy.ndarray: Centroid (x, y, z) of bounding box.

        """
        return np.array((self.bbox[0], self.bbox[1], self.bbox[2]))

    @property
    def get_output(self):
        """
        Track data output in VISDRONE Challenge format with tuple as
        `(frame_index, target_id, bbox_left, bbox_top, bbox_width, bbox_height, score, object_category,
        truncation, occlusion)`.

        References:
            - Website : http://aiskyeye.com/
            - Paper : https://arxiv.org/abs/2001.06303
            - GitHub : https://github.com/VisDrone/VisDrone2018-MOT-toolkit
            - GitHub : https://github.com/VisDrone/

        Returns:
            tuple: Tuple containing the elements as `(frame_index, target_id, x, y, z, dx, dy, dz, yaw,
            score, object_category, truncation, occlusion)`.
        """
        return (
            self.frame_id, self.id,
            self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3], self.bbox[4], self.bbox[5], self.bbox[6],
            self.detection_confidence, self.class_id, -1, -1
        )

    def predict(self):
        """
        Implement to prediction the next estimate of track.
        """
        raise NotImplemented

class KFTrack7DSORT(Track3D):
    """
    Track based on Kalman filter tracker used for SORT MOT-Algorithm.
    x y z w l h angle
    Args:
        track_id (int): Track Id
        frame_id (int): Camera frame id.
        bbox (numpy.ndarray): Bounding box pixel coordinates as (x_center, y_center, width, height) of the track.
        detection_confidence (float): Detection confidence of the object (probability).
        class_id (str or int): Class label id.
        lost (int): Number of times the object or track was not tracked by tracker in consecutive frames.
        iou_score (float): Intersection over union score.
        process_noise_scale (float): Process noise covariance scale or covariance magnitude as scalar value.
        measurement_noise_scale (float): Measurement noise covariance scale or covariance magnitude as scalar value.
        kwargs (dict): Additional key word arguments.

    """
    def __init__(self, track_id, frame_id, bbox, detection_confidence, class_id=None, lost=0, iou_score=0.,
                 process_noise_scale=1.0, measurement_noise_scale=1.0,
                 kf_time_step=1, **kwargs):
        self.kf = KFTracker7D(
            bbox.copy(),
            process_noise_scale=process_noise_scale, measurement_noise_scale=measurement_noise_scale,
            time_step=kf_time_step)
        super().__init__(track_id, frame_id, bbox, detection_confidence, class_id=class_id, lost=lost,
                         iou_score=iou_score, **kwargs)

    def get_bbox(self):
        # x = self.kf.x
        return self.bbox.copy()

    def predict(self):
        x = self.kf.predict()
        bb = np.array([x[0], x[3], x[6], x[9], x[12], x[15], x[18]])
        return bb

    def update(self, frame_id, bbox, detection_confidence, class_id=None, lost=0, iou_score=0., **kwargs):
        super().update(
            frame_id, bbox, detection_confidence, class_id=class_id, lost=lost, iou_score=iou_score, **kwargs)
        self.kf.update(bbox.copy())

def get_centroid3d(bboxes):
    """
    Calculate centroids for multiple bounding boxes.

    Args:
        bboxes (numpy.ndarray): Array of shape `(n, 7)` or of shape `(7,)` where
            each row contains `(x, y, z, dx, dy, dz, yaw)`.

    Returns:
        numpy.ndarray: Centroid (x, y, z) coordinates of shape `(n, 2)` or `(2,)`.

    """

    one_bbox = False
    if len(bboxes.shape) == 1:
        one_bbox = True
        bboxes = bboxes[None, :]

    x = bboxes[:, 0]
    y = bboxes[:, 1]
    z = bboxes[:, 2]

    output = np.hstack([x[:, None], y[:, None], z[:, None]])

    if one_bbox:
        output = output.flatten()
    return output

class Tracker3D:
    """
    Greedy Tracker with tracking based on ``centroid`` location of the bounding box of the object.
    This tracker is also referred as ``CentroidTracker`` in this repository.

    Args:
        max_lost (int): Maximum number of consecutive frames object was not detected.
    """

    def __init__(self, max_lost=5):
        self.next_track_id = 0
        self.tracks = OrderedDict()
        self.max_lost = max_lost
        self.frame_count = 0

    def _add_track(self, frame_id, bbox, detection_confidence, class_id, **kwargs):
        """
        Add a newly detected object to the queue.

        Args:
            frame_id (int): Camera frame id.
            bbox (numpy.ndarray): Bounding box pixel coordinates as (x, y, z, dx, dy, dz, yaw) of the track.
            detection_confidence (float): Detection confidence of the object (probability).
            class_id (str or int): Class label id.
            kwargs (dict): Additional key word arguments.
        """

        self.tracks[self.next_track_id] = Track3D(
            self.next_track_id, frame_id, bbox, detection_confidence, class_id=class_id,
            **kwargs
        )
        self.next_track_id += 1

    def _remove_track(self, track_id):
        """
        Remove tracker data after object is lost.

        Args:
            track_id (int): track_id of the track lost while tracking.
        """

        del self.tracks[track_id]

    def _update_track(self, track_id, frame_id, bbox, detection_confidence, class_id, lost=0, iou_score=0., **kwargs):
        """
        Update track state.

        Args:
            track_id (int): ID of the track.
            frame_id (int): Frame count.
            bbox (numpy.ndarray or list): Bounding box coordinates as `(xmin, ymin, width, height)`.
            detection_confidence (float): Detection confidence (a.k.a. detection probability).
            class_id (int): ID of the class (aka label) of the object being tracked.
            lost (int): Number of frames the object was lost while tracking.
            iou_score (float): Intersection over union.
            kwargs (dict): Additional keyword arguments.
        """

        self.tracks[track_id].update(
            frame_id, bbox, detection_confidence, class_id=class_id, lost=lost, iou_score=iou_score, **kwargs
        )

    def _update_track_2d(self, track_id, detection_confidence, class_id, distance=0., **kwargs):
        """
        Update track state.

        Args:
            track_id (int): ID of the track.
            frame_id (int): Frame count.
            bbox (numpy.ndarray or list): Bounding box coordinates as `(xmin, ymin, width, height)`.
            detection_confidence (float): Detection confidence (a.k.a. detection probability).
            class_id (int): ID of the class (aka label) of the object being tracked.
            lost (int): Number of frames the object was lost while tracking.
            iou_score (float): Intersection over union.
            kwargs (dict): Additional keyword arguments.
        """

        self.tracks[track_id].update_2d(
            detection_confidence, class_id=class_id, distance=distance, **kwargs
        )

    @staticmethod
    def _get_tracks(tracks):
        """
        Output the information of tracks.

        Args:
            tracks (OrderedDict): Tracks dictionary with (key, value) as (track_id, corresponding `Track` objects).

        Returns:
            list: List of tracks being currently tracked by the tracker.
        """

        outputs = []
        for trackid, track in tracks.items():
            if not track.lost:
                outputs.append(track.output)
        return outputs

    @staticmethod
    def preprocess_input(bboxes, class_ids, detection_scores):
        """
        Preprocess the input data.

        Args:
            bboxes (list or numpy.ndarray): Array of bounding boxes with each bbox as a tuple containing `(xmin, ymin, width, height)`.
            class_ids (list or numpy.ndarray): Array of Class ID or label ID.
            detection_scores (list or numpy.ndarray): Array of detection scores (a.k.a. detection probabilities).

        Returns:
            detections (list[Tuple]): Data for detections as list of tuples containing `(bbox, class_id, detection_score)`.
        """

        new_bboxes = np.array(bboxes, dtype='float')
        new_class_ids = np.array(class_ids, dtype='int')
        new_detection_scores = np.array(detection_scores)

        new_detections = list(zip(new_bboxes, new_class_ids, new_detection_scores))
        return new_detections

    def update(self, bboxes, detection_scores, class_ids):
        """
        Update the tracker based on the new bounding boxes.

        Args:
            bboxes (numpy.ndarray or list): List of bounding boxes detected in the current frame. Each element of the list represent
                coordinates of bounding box as tuple `(top-left-x, top-left-y, width, height)`.
            detection_scores(numpy.ndarray or list): List of detection scores (probability) of each detected object.
            class_ids (numpy.ndarray or list): List of class_ids (int) corresponding to labels of the detected object. Default is `None`.

        Returns:
            list: List of tracks being currently tracked by the tracker. Each track is represented by the tuple with elements `(frame_id, track_id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z)`.
        """

        self.frame_count += 1

        if len(bboxes) == 0:
            lost_ids = list(self.tracks.keys())

            for track_id in lost_ids:
                self.tracks[track_id].lost += 1
                if self.tracks[track_id].lost > self.max_lost:
                    self._remove_track(track_id)

            outputs = self._get_tracks(self.tracks)
            return outputs

        detections = Tracker3D.preprocess_input(bboxes, class_ids, detection_scores)

        track_ids = list(self.tracks.keys())

        updated_tracks, updated_detections = [], []

        if len(track_ids):
            track_centroids = np.array([self.tracks[tid].centroid for tid in track_ids])
            detection_centroids = get_centroid3d(np.asarray(bboxes))

            centroid_distances = distance.cdist(track_centroids, detection_centroids)

            track_indices = np.amin(centroid_distances, axis=1).argsort()

            for idx in track_indices:
                track_id = track_ids[idx]

                remaining_detections = [
                    (i, d) for (i, d) in enumerate(centroid_distances[idx, :]) if i not in updated_detections]

                if len(remaining_detections):
                    detection_idx, detection_distance = min(remaining_detections, key=lambda x: x[1])
                    bbox, class_id, confidence = detections[detection_idx]
                    self._update_track(track_id, self.frame_count, bbox, confidence, class_id=class_id)
                    updated_detections.append(detection_idx)
                    updated_tracks.append(track_id)

                if len(updated_tracks) == 0 or track_id is not updated_tracks[-1]:
                    self.tracks[track_id].lost += 1
                    if self.tracks[track_id].lost > self.max_lost:
                        self._remove_track(track_id)

        for i, (bbox, class_id, confidence) in enumerate(detections):
            if i not in updated_detections:
                self._add_track(self.frame_count, bbox, confidence, class_id=class_id)

        outputs = self._get_tracks(self.tracks)
        return outputs


class KFTrackCentroid3D(Track3D):
    """
    Track based on Kalman filter used for Centroid Tracking of bounding box in MOT.

    Args:
        track_id (int): Track Id
        frame_id (int): Camera frame id.
        bbox (numpy.ndarray): Bounding box pixel coordinates as (xmin, ymin, width, height) of the track.
        detection_confidence (float): Detection confidence of the object (probability).
        class_id (str or int): Class label id.
        lost (int): Number of times the object or track was not tracked by tracker in consecutive frames.
        iou_score (float): Intersection over union score.
        process_noise_scale (float): Process noise covariance scale or covariance magnitude as scalar value.
        measurement_noise_scale (float): Measurement noise covariance scale or covariance magnitude as scalar value.
        kwargs (dict): Additional key word arguments.
    """
    def __init__(self, track_id, frame_id, bbox, detection_confidence, class_id=None, lost=0, iou_score=0.,
                 process_noise_scale=1.0, measurement_noise_scale=1.0, **kwargs):
        c = np.array((bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], bbox[6]))
        self.kf = KFTracker7D(c, process_noise_scale=process_noise_scale, measurement_noise_scale=measurement_noise_scale)
        super().__init__(track_id, frame_id, bbox, detection_confidence, class_id=class_id, lost=lost,
                         iou_score=iou_score, **kwargs)

    def predict(self):
        """
        Predicts the next estimate of the bounding box of the track.

        Returns:
            numpy.ndarray: Bounding box pixel coordinates as (xmin, ymin, width, height) of the track.

        """
        s = self.kf.predict()
        xmid, ymid, zmid = s[0], s[3], s[6]
        dx, dy, dz = self.bbox[3], self.bbox[4], self.bbox[5]
        yaw = self.bbox[6]
        
        return np.array([xmid, ymid, zmid, dx, dy, dz, yaw]).astype(int)

    def update(self, frame_id, bbox, detection_confidence, class_id=None, lost=0, iou_score=0., **kwargs):
        super().update(
            frame_id, bbox, detection_confidence, class_id=class_id, lost=lost, iou_score=iou_score, **kwargs)
        self.kf.update(self.centroid)


def assign_tracks2detection_centroid_distances_3d(bbox_tracks, bbox_detections, distance_threshold=0.3):
    """
    Assigns detected bounding boxes to tracked bounding boxes using IoU as a distance metric.

    Args:
        bbox_tracks (numpy.ndarray): Tracked bounding boxes with shape `(n, 4)`
            and each row as `(xmin, ymin, width, height)`.
        bbox_detections (numpy.ndarray): detection bounding boxes with shape `(m, 4)` and
            each row as `(xmin, ymin, width, height)`.
        distance_threshold (float): Minimum distance between the tracked object
            and new detection to consider for assignment.

    Returns:
        tuple: Tuple containing the following elements:
            - matches (numpy.ndarray): Array of shape `(n, 2)` where `n` is number of pairs formed after matching tracks to detections. This is an array of tuples with each element as matched pair of indices`(track_index, detection_index)`.
            - unmatched_detections (numpy.ndarray): Array of shape `(m,)` where `m` is number of unmatched detections.
            - unmatched_tracks (numpy.ndarray): Array of shape `(k,)` where `k` is the number of unmatched tracks.

    """

    if (bbox_tracks.size == 0) or (bbox_detections.size == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(bbox_detections), dtype=int), np.empty((0,), dtype=int)

    if len(bbox_tracks.shape) == 1:
        bbox_tracks = bbox_tracks[None, :]

    if len(bbox_detections.shape) == 1:
        bbox_detections = bbox_detections[None, :]

    estimated_track_centroids = get_centroid3d(bbox_tracks)
    detection_centroids = get_centroid3d(bbox_detections)
    centroid_distances = distance.cdist(estimated_track_centroids, detection_centroids)

    assigned_tracks, assigned_detections = linear_sum_assignment(centroid_distances)

    unmatched_detections, unmatched_tracks = [], []

    for d in range(bbox_detections.shape[0]):
        if d not in assigned_detections:
            unmatched_detections.append(d)

    for t in range(bbox_tracks.shape[0]):
        if t not in assigned_tracks:
            unmatched_tracks.append(t)

    # filter out matched with high distance between centroids
    matches = []
    for t, d in zip(assigned_tracks, assigned_detections):
        if centroid_distances[t, d] > distance_threshold:
            unmatched_detections.append(d)
            unmatched_tracks.append(t)
        else:
            matches.append((t, d))

    if len(matches):
        matches = np.array(matches)
    else:
        matches = np.empty((0, 2), dtype=int)

    return matches, np.array(unmatched_detections), np.array(unmatched_tracks)


def assign_tracks2rays_centroid_distances_3d(bbox_tracks, rays_detections, distance_threshold=10.):
    """
    Assigns detected bounding boxes to tracked bounding boxes using IoU as a distance metric.

    Args:
        bbox_tracks (numpy.ndarray): Tracked bounding boxes with shape `(n, 4)`
            and each row as `(xmin, ymin, width, height)`.
        bbox_detections (numpy.ndarray): detection bounding boxes with shape `(m, 4)` and
            each row as `(xmin, ymin, width, height)`.
        distance_threshold (float): Minimum distance between the tracked object
            and new detection to consider for assignment.

    Returns:
        tuple: Tuple containing the following elements:
            - matches (numpy.ndarray): Array of shape `(n, 2)` where `n` is number of pairs formed after matching tracks to detections. This is an array of tuples with each element as matched pair of indices`(track_index, detection_index)`.
            - unmatched_detections (numpy.ndarray): Array of shape `(m,)` where `m` is number of unmatched detections.
            - unmatched_tracks (numpy.ndarray): Array of shape `(k,)` where `k` is the number of unmatched tracks.

    """

    if (bbox_tracks.size == 0) or (rays_detections.size == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(rays_detections), dtype=int), np.empty((0,), dtype=int)

    if len(bbox_tracks.shape) == 1:
        bbox_tracks = bbox_tracks[None, :]

    if len(rays_detections.shape) == 1:
        rays_detections = rays_detections[None, :]
    plucker_vecs = np.hstack((rays_detections[:,:3], np.cross(rays_detections[:,3:], rays_detections[:,3:] + rays_detections[:,:3])))
    estimated_track_centroids = get_centroid3d(bbox_tracks)
    # TODO: get plucker vector of rays, compute and assign based on distances to tracks
    # detection_centroids = get_centroid3d(rays_detections)

    centroid_distances = np.vstack([np.linalg.norm(np.cross(estimated_track_centroids.T, ray[:3], axis=0).T - ray[None, 3:], axis=1) for ray in plucker_vecs]).T

    masks = np.vstack([np.dot(estimated_track_centroids - ray[3:], ray[:3])>0 for ray in rays_detections]).T

    distances = centroid_distances + 50*centroid_distances*~masks # inflates distance of objects behind the camera

    assigned_tracks, assigned_detections = linear_sum_assignment(distances)

    unmatched_detections, unmatched_tracks = [], []

    for d in range(rays_detections.shape[0]):
        if d not in assigned_detections:
            unmatched_detections.append(d)

    for t in range(bbox_tracks.shape[0]):
        if t not in assigned_tracks:
            unmatched_tracks.append(t)

    # filter out matched with high distance between centroids
    matches = []
    for t, d in zip(assigned_tracks, assigned_detections):
        if centroid_distances[t, d] > distance_threshold:
            unmatched_detections.append(d)
            unmatched_tracks.append(t)
        else:
            matches.append((t, d))

    if len(matches):
        matches = np.array(matches)
    else:
        matches = np.empty((0, 2), dtype=int)

    return matches, np.array(unmatched_detections), np.array(unmatched_tracks)


class Centroid3DKF_Tracker(Tracker3D):
    """
    Kalman filter based tracking of multiple detected objects.

    Args:
        max_lost (int): Maximum number of consecutive frames object was not detected.
        process_noise_scale (float or numpy.ndarray): Process noise covariance matrix of shape (3, 3) or
            covariance magnitude as scalar value.
        measurement_noise_scale (float or numpy.ndarray): Measurement noise covariance matrix of shape (1,)
            or covariance magnitude as scalar value.
        time_step (int or float): Time step for Kalman Filter.
    """

    def __init__(
            self,
            max_lost=1,
            centroid_distance_threshold=30.,
            process_noise_scale=1.0,
            measurement_noise_scale=1.0,
            time_step=1
    ):
        self.time_step = time_step
        self.process_noise_scale = process_noise_scale
        self.measurement_noise_scale = measurement_noise_scale
        self.centroid_distance_threshold = centroid_distance_threshold
        self.kalman_trackers = OrderedDict()
        super().__init__(max_lost)

    def _add_track(self, frame_id, bbox, detection_confidence, class_id, **kwargs):
        self.tracks[self.next_track_id] = KFTrackCentroid3D(
            self.next_track_id, frame_id, bbox, detection_confidence, class_id=class_id,
            process_noise_scale=self.process_noise_scale,
            measurement_noise_scale=self.measurement_noise_scale, **kwargs
        )
        self.next_track_id += 1

    def update_2d(self, rays, detection_scores, class_ids):
        rays_detections = np.array(rays, dtype='float')
        # rays_detections = np.hstack((rays_detections[:,:3], np.cross(rays_detections[:,:3], rays_detections[:,3:])))

        track_ids = list(self.tracks.keys())
        bbox_tracks = []
        for track_id in track_ids:
            bbox_tracks.append(self.tracks[track_id].get_bbox())
        bbox_tracks = np.array(bbox_tracks)

        if len(rays) > 0:
            matches, unmatched_detections, unmatched_tracks = assign_tracks2rays_centroid_distances_3d(
                bbox_tracks, rays_detections, distance_threshold=1.0 # at most 10 cm from ray to detection
            )

            for i in range(matches.shape[0]):
                t, d = matches[i, :]
                track_id = track_ids[t]
                bbox = rays_detections[d, :]
                cid = class_ids[d]
                confidence = detection_scores[d]
                self._update_track_2d(track_id, confidence, cid, lost=0)

            # for d in unmatched_detections: # ignore unmatched
            #     bbox = bboxes[d, :]
            #     cid = class_ids[d]
            #     confidence = detection_scores[d]
            #     self._add_track(self.frame_count, bbox, confidence, cid)

        outputs = self._get_tracks(self.tracks)
        return outputs

    def update(self, bboxes, detection_scores, class_ids):
        self.frame_count += 1
        bbox_detections = np.array(bboxes, dtype='int')

        track_ids = list(self.tracks.keys())
        bbox_tracks = []
        for track_id in track_ids:
            bbox_tracks.append(self.tracks[track_id].predict())
        bbox_tracks = np.array(bbox_tracks)

        if len(bboxes) == 0:
            for i in range(len(bbox_tracks)):
                track_id = track_ids[i]
                bbox = bbox_tracks[i, :]
                confidence = self.tracks[track_id].detection_confidence
                cid = self.tracks[track_id].class_id
                self._update_track(track_id, self.frame_count, bbox, detection_confidence=confidence, class_id=cid, lost=1)
                if self.tracks[track_id].lost > self.max_lost:
                    self._remove_track(track_id)
        else:
            matches, unmatched_detections, unmatched_tracks = assign_tracks2detection_centroid_distances_3d(
                bbox_tracks, bbox_detections, distance_threshold=self.centroid_distance_threshold
            )

            for i in range(matches.shape[0]):
                t, d = matches[i, :]
                track_id = track_ids[t]
                bbox = bboxes[d, :]
                cid = class_ids[d]
                confidence = detection_scores[d]
                self._update_track(track_id, self.frame_count, bbox, confidence, cid, lost=0)

            for d in unmatched_detections:
                bbox = bboxes[d, :]
                cid = class_ids[d]
                confidence = detection_scores[d]
                self._add_track(self.frame_count, bbox, confidence, cid)

            for t in unmatched_tracks:
                track_id = track_ids[t]
                bbox = bbox_tracks[t, :]
                confidence = self.tracks[track_id].detection_confidence
                cid = self.tracks[track_id].class_id
                self._update_track(track_id, self.frame_count, bbox, confidence, cid, lost=1)

                if self.tracks[track_id].lost > self.max_lost:
                    self._remove_track(track_id)

        outputs = self._get_tracks(self.tracks)
        return outputs

class SORT3D(Centroid3DKF_Tracker):
    """
    SORT - Multi object tracker.

    Args:
        max_lost (int): Max. number of times a object is lost while tracking.
        dist_threshold (float): Intersection over union minimum value.
        process_noise_scale (float or numpy.ndarray): Process noise covariance matrix of shape (3, 3)
            or covariance magnitude as scalar value.
        measurement_noise_scale (float or numpy.ndarray): Measurement noise covariance matrix of shape (1,)
            or covariance magnitude as scalar value.
        time_step (int or float): Time step for Kalman Filter.
    """

    def __init__(
            self, max_lost=0,
            dist_threshold=1.0,
            process_noise_scale=1.0,
            measurement_noise_scale=1.0,
            time_step=1
    ):
        self.dist_threshold = dist_threshold

        super().__init__(
            max_lost=max_lost,
            process_noise_scale=process_noise_scale,
            measurement_noise_scale=measurement_noise_scale, time_step=time_step
        )

    def _add_track(self, frame_id, bbox, detection_confidence, class_id, **kwargs):
        self.tracks[self.next_track_id] = KFTrack7DSORT(
            self.next_track_id, frame_id, bbox, detection_confidence, class_id=class_id,
            process_noise_scale=self.process_noise_scale,
            measurement_noise_scale=self.measurement_noise_scale, kf_time_step=1, **kwargs)
        self.next_track_id += 1

    def update(self, bboxes, detection_scores, class_ids):
        self.frame_count += 1

        bbox_detections = np.array(bboxes, dtype='int')

        # track_ids_all = list(self.tracks.keys())
        # bbox_tracks = []
        # track_ids = []
        # for track_id in track_ids_all:
        #     bb = self.tracks[track_id].predict()
        #     if np.any(np.isnan(bb)):
        #         self._remove_track(track_id)
        #     else:
        #         track_ids.append(track_id)
        #         bbox_tracks.append(bb)

        track_ids = list(self.tracks.keys())
        bbox_tracks = []
        for track_id in track_ids:
            bbox_tracks.append(self.tracks[track_id].predict())

        bbox_tracks = np.array(bbox_tracks)

        if len(bboxes) == 0:
            for i in range(len(bbox_tracks)):
                track_id = track_ids[i]
                bbox = bbox_tracks[i, :]
                confidence = self.tracks[track_id].detection_confidence
                cid = self.tracks[track_id].class_id
                self._update_track(track_id, self.frame_count, bbox, detection_confidence=confidence, class_id=cid, lost=1)
                if self.tracks[track_id].lost > self.max_lost:
                    self._remove_track(track_id)
        else:
            t1 = time.time()
            matches, unmatched_detections, unmatched_tracks = assign_tracks2detection_iou(
                bbox_tracks, bbox_detections, dist_threshold=self.dist_threshold)
            t2 = time.time()

            for i in range(matches.shape[0]):
                t, d = matches[i, :]
                track_id = track_ids[t]
                bbox = bboxes[d, :]
                cid = class_ids[d]
                confidence = detection_scores[d]
                self._update_track(track_id, self.frame_count, bbox, confidence, cid, lost=0)
            t3 = time.time()

            for d in unmatched_detections:
                bbox = bboxes[d, :]
                cid = class_ids[d]
                confidence = detection_scores[d]
                self._add_track(self.frame_count, bbox, confidence, cid)
            t4 = time.time()

            for t in unmatched_tracks:
                track_id = track_ids[t]
                bbox = bbox_tracks[t, :]
                confidence = self.tracks[track_id].detection_confidence
                cid = self.tracks[track_id].class_id
                self._update_track(track_id, self.frame_count, bbox, detection_confidence=confidence, class_id=cid, lost=1)
                if self.tracks[track_id].lost > self.max_lost:
                    self._remove_track(track_id)

            print(f"assign_tracks2detection_iou: iou assign: {t2-t1}, update: {t3-t2},"
                  f" add: {t4-t3}, remove: {time.time()-t4}, # tracks {len(track_ids)} # bboxes {len(bboxes)}"
                  f" # remove: {len(unmatched_tracks)} # add: {len(unmatched_detections)}")
        outputs = self._get_tracks(self.tracks)
        return outputs
