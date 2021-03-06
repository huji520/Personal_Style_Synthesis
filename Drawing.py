import numpy as np
import copy
import matplotlib.pyplot as plt
import scipy.signal as signal
from Stroke import Stroke


class Drawing:
    def __init__(self, data, pic_path, stroke_size=None, is_cluster=False):
        """
        :param data: list of Strokes objects
        """
        self._data = data
        self._pic_path = pic_path
        self.set_strokes_size(stroke_size)
        self._is_cluster = is_cluster

    def set_strokes_size(self, stroke_size):
        if stroke_size:
            if stroke_size >= 1:
                for stroke in self._data:
                    if not stroke.is_pause():
                        stroke.set_size(stroke_size)
            else:
                for stroke in self._data:
                    if not stroke.is_pause():
                        stroke.set_size(int(stroke_size * stroke.size()))

    def get_pic_path(self):
        """
        :return: path to the drawing picture
        """
        return self._pic_path

    def get_data(self):
        """
        :return: list of Stroke objects
        """
        return self._data

    def size(self):
        """
        :return: number of strokes in the file
        """
        if self._is_cluster:
            return len(self._data)
        return int(len(self._data) / 2)  # half is pauses

    def total_length(self):
        """
        :return: first value is the total geometric length of a file (unit -> pixel),
                 second value is array of all the strokes lengths
        """
        dist = 0.0
        arr = []
        for stroke in self._data:
            if stroke.is_pause():
                continue
            arr.append(stroke.total_length())
            dist += stroke.total_length()

        return dist, np.asarray(arr)

    def average_length_of_stroke(self):
        """
        :return: average of geometric length of a single stroke
        """
        return self.total_length()[0] / float(self.size())

    def calc_center_of_mass(self):
        x = 0
        y = 0
        number_of_points_x = 0
        number_of_points_y = 0
        for stroke in self._data:
            if not stroke.is_pause():
                x += np.sum(stroke.get_feature('x'))
                y += np.sum(stroke.get_feature('y'))
                number_of_points_x += stroke.get_feature('x').size
                number_of_points_y += stroke.get_feature('y').size

        return x/number_of_points_x, y/number_of_points_y

    def length_std(self):
        """
        :return: std of the geometric length of the strokes
        """
        return self.total_length()[1].std()

    def get_strokes_as_one(self):
        """
        @TODO - maybe not necessary function
        @TODO - this is not efficient implements
        :return: all the strokes in the file, merged into one stroke, without pauses
        """
        map = {0: 'time', 1: 'x', 2: 'y', 3: 'pressure', 4: 'tiltX', 5: 'tiltY', 6: 'azimuth', 7: 'sidePressure',
               8: 'rotation'}
        copy_data = copy.deepcopy(self._data)
        new_stroke = []
        lst = []
        for j in range(len(map)):
            for i in range(self.size()):
                lst += list(copy_data[i].get_feature(map[j]))
            lst = []
            new_stroke.append(lst)

        return Stroke(np.asarray(new_stroke))

    def feature_vs_time(self, feature, unit, pause, save = False, path = ""):
        """
        plot graph, given feature as function of time
        :param feature: given feature for 'y' axis
        :param unit:
        :param pause: true iff pauses strokes will shown in the plot
        :param save: true for save the graph instead of plot it
        :param path: the path the graph will be saved
        """
        time = []
        y = []
        for stroke in self.get_data():
            if not pause:
                if stroke.is_pause():
                    continue
            time.append(stroke.get_feature("time")[0])
            if feature == "length":
                y.append(stroke.length())
            else:
                y.append(stroke.average(feature))

        time = np.asarray(time)
        y = np.asarray(y)
        plt.scatter(time, y, s=2)
        y_mean = y.mean()
        y_std = y.std()
        path_arr = path.split("/")
        plt.title(path_arr[1] + "-" + path_arr[2] + "\n" + feature + " mean " \
                    "is: " + str(y_mean) + "\n" + feature + " " +
                  "std is: " + str(y_std))
        plt.xlabel('time [sec]')
        plt.ylabel(feature + " [" + unit + "]")
        if not save:
            plt.show()
        else:
            plt.savefig(path)
        plt.close()
        return y_mean, y_std

    def speed_vs_time(self, pause=False, save = False, path = ""):
        """
        plot graph of speed vs time
        :param pause: true iff pauses strokes will shown in the plot
        :param save: true for save the graph instead of plot it
        :param path: the path the graph will be saved
        """
        return self.feature_vs_time("speed", "pixel/sec", pause, save, path)

    def pressure_vs_time(self, pause=False, save = False, path = ""):
        """
        plot graph of pressure vs time
        :param pause: true iff pauses strokes will shown in the plot
        :param save: true for save the graph instead of plot it
        :param path: the path the graph will be saved
        """
        return self.feature_vs_time("pressure", "?", pause, save, path)

    def length_vs_time(self, pause=False, save = False, path = ""):
        """
        plot graph of length vs time
        :param pause: true iff pauses strokes will shown in the plot
        :param save: true for save the graph instead of plot it
        :param path: the path the graph will be saved
        """
        return self.feature_vs_time("length", "pixel", pause, save, path)

    def get_active_image_sizes(self):
        """
        Calculates the locations of the pixels that border the image.
        :return: [start x, start y, end x, end y]
        """
        start_x = 20000
        start_y = 20000
        end_x = 0
        end_y = 0

        for stroke in self.get_data():
            if stroke.is_pause():
                continue
            start_x = min(min(stroke.get_feature('x')), start_x)
            end_x = max(max(stroke.get_feature('x')), end_x)
            start_y = min(min(stroke.get_feature('y')), start_y)
            end_y = max(max(stroke.get_feature('y')), end_y)

        return [start_x, start_y, end_x, end_y]

    def plot_picture(self, show_reference=False, show_clusters=False, title="", num=1, plt_show=True, show_by_labels=False, save=False):
        """
        plot the reference picture and the actual drawing.
        @TODO: pictures should be in the same resolution.
        """
        w = 6
        h = 4
        f = 3

        # drawing
        plt.figure(num=num, figsize=(f * 2.5, f * h))
        for stroke in self._data:
            avg_pressure = stroke.average('pressure') if 0 < stroke.average('pressure') < 0.15 else 0.15
            if show_clusters:
                if show_by_labels:
                    plt.plot(stroke.get_feature('x'), stroke.get_feature('y'),
                             linewidth=7 * avg_pressure,
                             label=f'{stroke._group}')
                else:
                    if stroke._color == 0:
                        plt.plot(stroke.get_feature('x'),stroke.get_feature('y'),
                                 linewidth=5 * avg_pressure,
                                 color='black')
                    elif stroke._color == 1:
                        plt.plot(stroke.get_feature('x'),stroke.get_feature('y'),
                                 linewidth=5 * avg_pressure,
                                 color='red')
                    elif stroke._color == 2:
                        plt.plot(stroke.get_feature('x'),stroke.get_feature('y'),
                                 linewidth=3 * avg_pressure,
                                 color='blue')
                    elif stroke._color == 3:
                        plt.plot(stroke.get_feature('x'),stroke.get_feature('y'),
                                 linewidth=5 * avg_pressure,
                                 color='green')
                    else:
                        plt.plot(stroke.get_feature('x'),stroke.get_feature('y'),
                                 linewidth=5 * avg_pressure,
                                 color='purple')
            else:
                plt.plot(stroke.get_feature('x'), stroke.get_feature('y'),
                         linewidth= 0.5 * (avg_pressure / (0.15 + avg_pressure)), color='black')

        plt.gca().invert_yaxis()
        if title:
            plt.title(title)

        if save:
            plt.savefig("out2.png")

        if plt_show:
            plt.show()

    def strokes_correlation(self, stroke_1, stroke_2):
        return signal.correlate2d(stroke_1, stroke_2)

    def strokes_euc_distance(self, stroke_1, stroke_2):
        """
        calculate the minimal euclidean distance between two strokes
        :param stroke_1: first 2D numpy array of (x,y) coordinates
        :param stroke_2: second 2D numpy array of (x,y) coordinates
        :return: euclidean distance
        """
        len_1, len_2 = len(stroke_1), len(stroke_2)
        # dist = np.sqrt(((stroke_1[: len_2] - stroke_2) ** 2).sum(-1)).sum() / min(len_1, len_2)
        # print(dist)
        if len_1 >= len_2:
            dist = np.sqrt(((stroke_1[: len_2] - stroke_2) ** 2).sum(-1)).sum() / len_2
            for i in range(1, len_1 - len_2):
                temp_dist = np.sqrt(((stroke_1[i: i + len_2] - stroke_2) ** 2).sum(-1)).sum() / len_2
                if temp_dist < dist: dist = temp_dist
        else:
            dist = np.sqrt(((stroke_1 - stroke_2[:len_1]) ** 2).sum(-1)).sum() / len_1
            for i in range(1, len_2 - len_1):
                temp_dist = np.sqrt(((stroke_2[i: i + len_1] - stroke_1) ** 2).sum(-1)).sum() / len_1
                if temp_dist < dist: dist = temp_dist
        return dist


    def strokes_distance(self, stroke_1, stroke_2, dist_threshold):
        x1_1, y1_1 = stroke_1[0][0], stroke_1[0][1]
        x1_2, y1_2 = stroke_1[-1][0], stroke_1[-1][1]
        x2_1, y2_1 = stroke_2[0][0], stroke_2[0][1]
        x2_2, y2_2 = stroke_2[-1][0], stroke_2[-1][1]
        if np.sqrt((x1_2 - x2_1) ** 2 + (y1_2 - y2_1) ** 2) <= dist_threshold:
            return 1
        if np.sqrt((x1_2 - x2_2) ** 2 + (y1_2 - y2_2) ** 2) <= dist_threshold:
            return 2
        if np.sqrt((x1_1 - x2_2) ** 2 + (y1_1 - y2_2) ** 2) <= dist_threshold:
            return 3
        if np.sqrt((x1_1 - x2_1) ** 2 + (y1_1 - y2_1) ** 2) <= dist_threshold:
            return 4
        return 0

    def strokes_angle_difference(self, stroke_1, stroke_2, dots_orientation):
        """
        calculate the angle between the end of one stroke and beginning of the other
        :param stroke_1: first 2D numpy array of (x,y) coordinates
        :param stroke_2: second 2D numpy array of (x,y) coordinates
        :param dots_orientation: which dots are close enough
        :return: abs of the difference between strokes angles
        """
        if dots_orientation == 1:
            x1_1, y1_1 = stroke_1[-1][0], stroke_1[-1][1]
            x1_2, y1_2 = stroke_1[-2][0], stroke_1[-2][1]
            x2_1, y2_1 = stroke_2[0][0], stroke_2[0][1]
            x2_2, y2_2 = stroke_2[1][0], stroke_2[1][1]
        elif dots_orientation == 2:
            x1_1, y1_1 = stroke_1[-1][0], stroke_1[-1][1]
            x1_2, y1_2 = stroke_1[-2][0], stroke_1[-2][1]
            x2_1, y2_1 = stroke_2[-1][0], stroke_2[-1][1]
            x2_2, y2_2 = stroke_2[-2][0], stroke_2[-2][1]
        elif dots_orientation == 3:
            x1_1, y1_1 = stroke_1[0][0], stroke_1[0][1]
            x1_2, y1_2 = stroke_1[1][0], stroke_1[1][1]
            x2_1, y2_1 = stroke_2[-1][0], stroke_2[-1][1]
            x2_2, y2_2 = stroke_2[-2][0], stroke_2[-2][1]
        elif dots_orientation == 4:
            x1_1, y1_1 = stroke_1[0][0], stroke_1[0][1]
            x1_2, y1_2 = stroke_1[1][0], stroke_1[1][1]
            x2_1, y2_1 = stroke_2[0][0], stroke_2[0][1]
            x2_2, y2_2 = stroke_2[1][0], stroke_2[1][1]
        else:
            return 0
        if x1_1 - x1_2 == 0:
            ang_1 = np.pi / 2
        else:
            ang_1 = np.arctan((y1_1 - y1_2) / (x1_1 - x1_2))
        if x2_1 - x2_2 == 0:
            ang_2 = np.pi / 2
        else:
            ang_2 = np.arctan((y2_1 - y2_2) / (x2_1 - x2_2))
        # print(np.arctan(abs((x2_1 - x2_2) / (y2_1 - y2_2))))
        return np.abs(abs(ang_1) - abs(ang_2))

    def group_strokes(self, euc_dist_threshold, dist_threshold, ang_threshold, max_num_of_strokes=0,
        limit_strokes_num=False, fixed_size_of_strokes=False):
        """
        divide all strokes in the drawing to groups according to distance and angle properties
        :param euc_dist_threshold: the maximal distance between strokes of which they'll belong to the same group
        :param ang_threshold: the maximal angle between strokes of which they'll belong to the same group
        :return: a list of drawing objects: each object is a part of this drawing with one group of strokes
        """
        print("Start clustering with group_strokes function")
        new_draws = []
        data = [stroke for stroke in self._data if not stroke.is_pause() and 30 <= stroke.length()]
        counter = 0
        while len(data) != 0:
            # group = [stroke for stroke in data[1:] if self.strokes_euc_distance(np.array(data[0].get_data()[1:3]).T,
            #          np.array(stroke.get_data()[1:3]).T) <= euc_dist_threshold
            #          or
            #          (self.strokes_distance(np.array(data[0].get_data()[1:3]).T, np.array(stroke.get_data()[1:3]).T,
            #          dist_threshold)
            #          and
            #          self.strokes_angle_difference(np.array(data[0].get_data()[1:3]).T,
            #          np.array(stroke.get_data()[1:3]).T, self.strokes_distance(np.array(data[0].get_data()[1:3]).T,
            #          np.array(stroke.get_data()[1:3]).T, dist_threshold)) <= ang_threshold)]
            # group.append(data[0])
            group = [data[0]]
            i = 0
            data[0]._group = counter + 1
            for stroke in data[1:]:
                if self.strokes_euc_distance(np.array(data[0].get_data()[1:3]).T, np.array(stroke.get_data()[1:3]).T) <= euc_dist_threshold \
                        and self.calc_error(np.array(data[0].get_data()[1:3]).T, np.array(stroke.get_data()[1:3]).T) < 30:
                     # or
                     # (self.strokes_distance(np.array(data[i].get_data()[1:3]).T, np.array(stroke.get_data()[1:3]).T,
                     # dist_threshold)
                     # and
                     # self.strokes_angle_difference(np.array(data[i].get_data()[1:3]).T,
                     # np.array(stroke.get_data()[1:3]).T, self.strokes_distance(np.array(data[i].get_data()[1:3]).T,
                     # np.array(stroke.get_data()[1:3]).T, dist_threshold)) <= ang_threshold)):
                    stroke._group = counter + 1
                    group.append(stroke)
                    i = i + 1
                if limit_strokes_num:
                    if len(group) == max_num_of_strokes:
                        break
            for stroke in group:
                stroke._color = counter % 5
                data = list(filter(lambda x: not np.array_equal(x, stroke), data))
            if (fixed_size_of_strokes):
                if len(group) < max_num_of_strokes:
                    for j in range(max_num_of_strokes - len(group)):
                        group.append(copy.deepcopy(group[0]))
                        group[-1]._color = counter % 5

            new_draws.append(Drawing(group, self._pic_path, is_cluster=True))
            # print(f"{counter}: Number of strokes in current cluster: {len(group)}")
            counter = counter + 1
        print(f"Number of clusters: {len(new_draws)}")
        print("End clustering with group_strokes function\n")
        return self, new_draws

    def shift_x(self, shift):
        for stroke in self._data:
            stroke.set_x(shift)

    def shift_y(self, shift):
        for stroke in self._data:
            stroke.set_y(shift)

    def rotate(self, angle):
        for stroke in self._data:
            stroke.rotate(angle)

    def __str__(self):
        s = f"The total number of strokes (without pause strokes) is: {self.size()}\n\n"
        for i, stroke in enumerate(self._data):
            if not stroke.is_pause():
                if self._is_cluster:
                    s += f"Current index: {i}\n"
                else:
                    s += f"Current index: {i//2}\n"
                s += f"{stroke.__str__()}\n"
        return s

    @staticmethod
    def calc_error(p1, p2):
        """
        calc error between two groups of 2D points, by OUR metric
        :param p1: array of 2D points
        :param p2: array of 2D points
        :return: the error (float)
        """
        pt1 = np.array(p1)  # NxD, here D=2
        pt2 = np.array(p2)  # MxD
        d = pt1[:, None, :] - pt2[None, :, :]  # pairwise subtraction, NxMxD
        d = np.sum(d ** 2, axis=2).min(axis=1)  # min square distance, N
        error1 = np.sqrt(d).sum()  # output, scalar

        d = pt2[:, None, :] - pt1[None, :, :]  # pairwise subtraction, NxMxD
        d = np.sum(d ** 2, axis=2).min(axis=1)  # min square distance, N
        error2 = np.sqrt(d).sum()  # output, scalar

        return (error1 + error2) / (len(p1) + len(p2))