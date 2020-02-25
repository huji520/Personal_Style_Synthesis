import numpy as np
import matplotlib.pyplot as plt


def l2(p1, p2):
    """
    calc l2 metric between two points
    :param p1: 2D point [x,y]
    :param p2: 2D point [x,y]
    :return: l2 metric between the two given points
    """
    return np.linalg.norm(np.array(p1) - np.array(p2))


def calc_error(p1, p2):
    """
    @TODO: if one array got much more points than the other
    calc error between two groups of 2D points, by OUR metric
    :param p1: array of 2D points
    :param p2: array of 2D points
    :return: the error (float)
    """
    error1 = 0
    for point1 in p1:
        min_dist = 1000000
        for point2 in p2:
            min_dist = min(min_dist, l2(point1, point2))
        error1 += min_dist

    error2 = 0
    for point2 in p2:
        min_dist = 10000000
        for point1 in p1:
            min_dist = min(min_dist, l2(point1, point2))
        error2 += min_dist

    return (error1 + error2) / (len(p1) + len(p2))


def find_nearest_neighbor(p1, neighbors):
    """
    @TODO: translation (not sure about rotation)
    find the closest array-points to the given array-points (p1)
    :param p1: array of 2D points
    :param neighbors: array of arrays of 2D points 2D
    :return: the nearest neighbor (array of 2D points)
    """
    index = 0
    p1, x_shift, y_shift = normalize_points(p1)
    n_x_shift, n_y_shift = 0, 0
    nearest_neighbor = None
    min_score = 10000000
    for i, p in enumerate(neighbors):
        normalize_p, temp_n_x_shift, temp_n_y_shift = normalize_points(p.copy())
        error = calc_error(p1, normalize_p)
        if error < min_score:
            min_score = error
            nearest_neighbor = normalize_p
            index = i
            n_x_shift, n_y_shift = temp_n_x_shift, temp_n_y_shift

    return nearest_neighbor, index, x_shift - n_x_shift, y_shift - n_y_shift


def normalize_points(points):
    """
    normalize the points to the axis
    :param points: array of 2D points
    :return: the normalize points
    """
    points = np.array(points)
    x_shift = np.min(points[:, 0])
    y_shift = np.min(points[:, 1])
    points[:, 0] -= x_shift
    points[:, 1] -= y_shift
    return points, x_shift, y_shift
