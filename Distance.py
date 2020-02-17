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
    p1 = normalize_points(p1)
    nearest_neighbor = None
    min_score = 10000000
    for p in neighbors:
        normalize_p = normalize_points(p.copy())
        error = calc_error(p1, normalize_p)
        if error < min_score:
            min_score = error
            nearest_neighbor = normalize_p

    return nearest_neighbor


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
    return points





# p1 = np.array([[1,2],[2,6],[3,10],[5,11]])
# p1 = np.array([[1,2, 3, 5, 7, 8, 9, 10],[2, 6, 10, 11, 9, 8, 5, 3]])
# p2 = np.array([[2,3],[4,5], [6,7],[8,9]])


# curve1 = bezier.Curve(p1, degree=2)
# curve = bezier.Curve(p1, degree=2)
# curve = bezier.Curve.from_nodes(p1)
# curve.plot(100)

# s_vals = np.linspace(0.0, 1.0, 5)
# points = curve.evaluate_multi(s_vals)
# print(points)


# print(Distance.calc_error(p1,p2))
# plt.plot(p1[0], p1[1], 'o', lw=0.5, ms=2, c='r')
# plt.plot(p1[:, 0], p1[:, 1], 'o', lw=0.5, ms=2, c='r')
# plt.plot(p2[:, 0], p2[:, 1], 'o', lw=0.5, ms=2, c='b')
# plt.show()