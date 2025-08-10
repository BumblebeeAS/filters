#!/usr/bin/env python3
import numpy as np
from bb_filters.cluster import assign_to_centroids
from numpy.typing import ArrayLike
from scipy.optimize import Bounds, minimize


def get_slalom_centroids(
    data: ArrayLike, num_centroids: int, D: float = 2.0, d_limit: float = 1.0
):
    def get_next_layer_centroid(x_0, y_0, d, D, theta):
        x_1 = x_0 + d * np.cos(theta) - D * np.sin(theta)
        y_1 = y_0 + d * np.sin(theta) + D * np.cos(theta)
        return x_1, y_1

    def get_centroids(x_0, y_0, ds, theta):
        centroids = [(x_0, y_0)]
        for d in ds:
            curr_x, curr_y = centroids[-1]
            next_x, next_y = get_next_layer_centroid(curr_x, curr_y, d, D, theta)
            centroids.append((next_x, next_y))
        return np.array(centroids)

    def objective_function(params, args):
        x_0, y_0, theta = params[:3]
        ds = params[3:]
        centroids = get_centroids(x_0, y_0, ds, theta)
        assigned = assign_to_centroids(data, centroids)
        return np.sum(np.linalg.norm(data - centroids[assigned], axis=1))

    num_params = 3 + num_centroids - 1
    bounds = Bounds(
        [-np.inf, -np.inf, -np.pi / 2] + [-d_limit] * (num_centroids - 1),
        [np.inf, np.inf, np.pi / 2] + [d_limit] * (num_centroids - 1),
    )

    # TODO: Handle minimize failure
    result = minimize(
        objective_function,
        x0=np.zeros(num_params),
        args=[],
        bounds=bounds,
        method="Nelder-Mead",
    )
    x_0, y_0, theta = result.x[:3]
    ds = result.x[3:]
    centroids = get_centroids(x_0, y_0, ds, theta)

    return theta, centroids
