import numpy as np
import matplotlib.pyplot as plt

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


def calc_curve(points, dist):

    all_curve = []  # list of lists
    pt = points.pop(0)
    curve = []  # list of control points of the current curve we work on
    while len(points) > 0:
        result = closest_points(pt, points, dist)
        if len(result) > 0:
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

            # remove all the closest points
            for i in range(lenP):
                print(rmv[i])
                print(points)
                print()
                points.remove(rmv[i])

            # if we are done append the end point
            if len(points) == 0:
                curve.append(pt)
                all_curve.append(curve)

        else:
            if len(curve) == 0:
                points.remove(pt)
                pt = points.pop(0)
                continue

            curveSartpoint = curve[0]
            result = closest_points(curveSartpoint, points, 2 * dist)
            if len(result):
                curve.reverse()
                pt = result[0]
                continue
            else:
                print("no close points found so save b and continue from wherever")
                all_curve.append(curve)
                if len(points) == 0:
                    break
                pt = points.pop(0)
                curve = []
                continue
    return all_curve


def remove_duplicates(points):
    new_points = np.unique(np.array(points), axis=0)
    return list(new_points)



x = [670, 670, 671, 672, 673, 675, 677, 680, 682, 683, 685, 686, 688, 689, 690, 690, 691, 691, 690]
y = [283, 283, 282, 282, 283, 284, 285, 287, 289, 291, 296, 299, 300, 303, 305, 308, 313, 316, 320]
#
# points = np.stack((x, y), axis=1)
# points = list(points)
# dist = 5
# curves = np.array(calc_curve(points, dist))
# points = []
# for curve in curves:
#     for point in curve:
#         points.append(point)
#
# points = np.array(points)
# plt.plot(x, y, 'o', lw=0.1, ms=2, c='b')
# plt.plot(points[:, 0], points[:, 1], 'o', lw=0.2, ms=2, c='r')
# plt.show()

def show_simplifican(x, y):
    points = np.stack((x, y), axis=1)
    points = list(points)
    print(points)
    # points = remove_duplicates(points)
    dist = 5
    curves = np.array(calc_curve(points, dist))
    print(curves)
    points = []
    for curve in curves:
        for point in curve:
            points.append(point)

    points = np.array(points)

    print(points)
    plt.plot(x, y, 'o', lw=0.1, ms=2, c='b')
    plt.plot(points[:, 0], points[:, 1], 'o', lw=0.2, ms=2, c='r')
    plt.show()


