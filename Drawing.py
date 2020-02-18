import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread, imshow, imsave
from PIL import Image
import scipy.signal as signal
import scipy.spatial.distance as distance
import Constants
from Stroke import Stroke


class Drawing:
    def __init__(self, data, pic_path, ref_path):
        """
        :param data: list of Strokes objects
        """
        self._data = data
        self._ref_path = ref_path
        self._pic_path = pic_path

    def get_ref_path(self):
        """
        :return: path to reference picture
        """
        return self._ref_path

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

    def plot_crop_image(self, return_without_plot = False):
        """

        :param return_without_plot: true for only return IMG ref without plot
        :return: IMG ref
        """
        img = Image.open(self.get_pic_path())
        locations = self.get_active_image_sizes()  # 0 -> start_x, 1 -> start_y, 2 -> end_x, 3 -> end_y
        img = img.crop((locations[0], locations[1], locations[2], locations[3]))
        img.thumbnail([locations[2] - locations[0], locations[3] - locations[1]], Image.ANTIALIAS)
        if not return_without_plot:
            plt.figure()
            plt.imshow(img)

        img_ref = Image.open(self._ref_path)
        ref_width = float(img_ref.size[0])
        ref_height = float(img_ref.size[1])
        new_width = img.size[0]  # the required width
        ratio = new_width / ref_width
        new_height = int(ref_height * float(ratio))
        img_ref = img_ref.resize((new_width, new_height), Image.ANTIALIAS)
        if not return_without_plot:
            plt.figure()
            plt.imshow(img_ref)
            plt.show()

        return img, img_ref

    def plot_picture(self):
        """
        plot the reference picture and the actual drawing.
        @TODO: pictures should be in the same resolution.
        """
        w = 6
        h = 4
        f = 3

        # drawing
        plt.figure(figsize=(f * 2.5, f * h))
        for stroke in self._data:
            if stroke._color == 0:
                plt.plot(stroke.get_feature('x'),stroke.get_feature('y'),
                         linewidth=3 * stroke.average('pressure'),
                         color='black')
            elif stroke._color == 1:
                plt.plot(stroke.get_feature('x'),stroke.get_feature('y'),
                         linewidth=3 * stroke.average('pressure'),
                         color='red')
            elif stroke._color == 2:
                plt.plot(stroke.get_feature('x'),stroke.get_feature('y'),
                         linewidth=3 * stroke.average('pressure'),
                         color='blue')
            elif stroke._color == 3:
                plt.plot(stroke.get_feature('x'),stroke.get_feature('y'),
                         linewidth=3 * stroke.average('pressure'),
                         color='green')
            else:
                plt.plot(stroke.get_feature('x'),stroke.get_feature('y'),
                         linewidth=3 * stroke.average('pressure'),
                         color='purple')
        plt.gca().invert_yaxis()
        # reference
        plt.figure(figsize=(w, h))
        image = Image.open(self._ref_path)
        ref_width = float(image.size[0])
        ref_height = float(image.size[1])
        new_width = 800  # the required width
        ratio = new_width / ref_width
        new_height = int(ref_height * float(ratio))
        image = image.resize((new_width, new_height), Image.ANTIALIAS)
        plt.imshow(image)

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

    def group_strokes(self, euc_dist_threshold, dist_threshold, ang_threshold):
        """
        divide all strokes in the drawing to groups according to distance and angle properties
        :param euc_dist_threshold: the maximal distance between strokes of which they'll belong to the same group
        :param ang_threshold: the maximal angle between strokes of which they'll belong to the same group
        :return: a list of drawing objects: each object is a part of this drawing with one group of strokes
        """
        new_draws = []
        data = [stroke for stroke in self._data if not stroke.is_pause() and 30 <= stroke.length() <= 250]
        # print(data)
        counter = 0
        while len(data) is not 0:
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
            group = []
            group.append(data[0])
            i = 0
            data[0]._group = counter + 1
            for stroke in data[1:]:
                if (self.strokes_euc_distance(np.array(data[0].get_data()[1:3]).T,
                     np.array(stroke.get_data()[1:3]).T) <= euc_dist_threshold
                     or
                     (self.strokes_distance(np.array(data[0].get_data()[1:3]).T, np.array(stroke.get_data()[1:3]).T,
                     dist_threshold)
                     and
                     self.strokes_angle_difference(np.array(data[i].get_data()[1:3]).T,
                     np.array(stroke.get_data()[1:3]).T, self.strokes_distance(np.array(data[i].get_data()[1:3]).T,
                     np.array(stroke.get_data()[1:3]).T, dist_threshold)) <= ang_threshold)):
                    stroke._group = counter + 1
                    group.append(stroke)
                    i = i + 1
            for stroke in group:
                stroke._color = counter % 5
                data = list(filter(lambda x: not np.array_equal(x, stroke), data))
            new_draws.append(Drawing(group, self._ref_path, self._pic_path))
            counter = counter + 1
            # print(group)

        return self, new_draws