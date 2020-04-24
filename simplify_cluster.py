import numpy as np


def distance(pt_1, pt_2):
    """
    calc l2 metric between two points
    :param pt_1: 2D point [x,y]
    :param pt_2: 2D point [x,y]
    :return: l2 metric between the two given points
    """
    pt_1 = np.array((pt_1[0], pt_1[1]))
    pt_2 = np.array((pt_2[0], pt_2[1]))
    return np.linalg.norm(pt_1-pt_2)


def closest_points(start_point, points, dist):
    """
    find all the points in 'points' in range of 'dist' to the start_point
    :param start_point: the given start point 2D point
    :param points: array of 2D points
    :param dist: int
    :return: array of all the closest points
    """
    pt = []
    for point in points:
        if distance(start_point, point) <= dist:
            pt.append(point)
    return pt


def add_next_point(result, curve):
    """
    Add the next point to the curve and return all the points that should be removed
    :param result: the points that found in closest_points function
    :param curve: the current curve
    :return: array of all the points that should be removed
    """
    rmv = []
    next_point = [0, 0]
    lenP = len(result)
    for i in range(lenP):
        next_point[0] += result[i][0]
        next_point[1] += result[i][1]
        rmv.append(result[i])

    next_point[0] /= lenP
    next_point[1] /= lenP
    curve.append(next_point)
    return rmv, next_point


def remove_points(points, points_to_remove):
    """
    Remove points from the original array of points
    :param points: the original array of points (array of 2D points)
    :param points_to_remove: points that should be removed (array of 2D points)
    :return: the original array of points after removing
    """
    for i in range(len(points_to_remove)):
        j = 0
        while j < len(points):
            if points_to_remove[i][0] == points[j][0] and points_to_remove[i][1] == points[j][1]:
                points = points[:j] + points[j + 1:]
                j = len(points) - 1
            j += 1
    return points


def calc_curve(points, dist):
    """

    :param points:
    :param dist:
    :return:
    """
    all_curve = []  # list of lists
    pt = points.pop(0)
    curve = []  # list of control points of the current curve we work on
    while len(points) > 0:
        result = closest_points(pt, points, dist)

        if len(result) > 0:
            points_to_remove, point = add_next_point(result, curve)
            pt = point
            points = remove_points(points, points_to_remove)

            # if we are done append the end point
            if len(points) == 0:
                curve.append(pt)
                all_curve.append(curve)

        else:
            # no close points are found and the curve is still empty
            if len(curve) == 0:
                points = points[1:]
                if len(points) == 0:
                    break
                pt = points.pop(0)
                continue

            curveSartpoint = curve[-1]
            result = closest_points(curveSartpoint, points, 2 * dist)
            if len(result) > 0:
                # curve.reverse()
                pt = result[0]
                continue
            else:
                all_curve.append(curve)
                if len(points) == 0:
                    break
                pt = points.pop(0)
                curve = []
                continue
    return all_curve


def simplify_cluster(x, y, dist):
    """

    :param x:
    :param y:
    :param dist:
    :return:
    """
    points = np.stack((x, y), axis=1)
    points = list(points)
    curves = np.array(calc_curve(points, dist))
    points = []
    for curve in curves:
        for point in curve:
            points.append(point)

    return points


