import numpy as np
import matplotlib.pyplot as plt
import os

def distance(pt_1, pt_2):
    pt_1 = np.array((pt_1[0], pt_1[1]))
    pt_2 = np.array((pt_2[0], pt_2[1]))
    return np.linalg.norm(pt_1-pt_2)


def closest_points(start_point, points, dist):
    pt = []
    for point in points:
        if distance(start_point, point) <= dist:
            pt.append(point)
    return pt


def add_next_point(result, curve):
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
    return rmv


def remove_points(points, points_to_remove):
    for i in range(len(points_to_remove)):
        j = 0
        while j < len(points):
            if points_to_remove[i][0] == points[j][0] and points_to_remove[i][1] == points[j][1]:
                points = points[:j] + points[j + 1:]
                j = len(points) - 1
            j += 1
    return points


def calc_curve(points, dist):

    all_curve = []  # list of lists
    pt = points.pop(0)
    curve = []  # list of control points of the current curve we work on
    while len(points) > 0:
        result = closest_points(pt, points, dist)
        if len(result) > 0:
            points_to_remove = add_next_point(result, curve)
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

            curveSartpoint = curve[0]
            result = closest_points(curveSartpoint, points, 2 * dist)
            if len(result) > 0:
                curve.reverse()
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


def show_simplifican(x, y, i):
    points = np.stack((x, y), axis=1)
    points = list(points)
    dist = 14
    curves = np.array(calc_curve(points, dist))
    points = []
    for curve in curves:
        for point in curve:
            points.append(point)

    points = np.array(points)
    plt.figure(i)
    plt.subplot(121)
    plt.plot(x, y, 'o', lw=0.1, ms=2, c='b')
    plt.subplot(122)
    plt.plot(points[:, 0], points[:, 1], 'o', lw=0.5, ms=2, c='r')
    plt.savefig(os.path.join('simplify_clusters_dist14', '{0}.png'.format(i)))


