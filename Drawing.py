import numpy as np
import copy
from Stroke import Stroke
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Drawing:
    def __init__(self, data, pic_path, ref_path):
        """
        :param data: list of Strokes objects
        """
        self._data = data
        self._ref_path = ref_path
        self._pic_path = pic_path

    def get_ref_path(self):
        return self._ref_path

    def get_pic_path(self):
        return self._pic_path

    def get_data(self):
        """
        :return: list of Stroke objects
        """
        return self._data

    def size(self):
        """
        :return: number of stroke in the file
        """
        return len(self._data)

    def total_length(self):
        """
        :return: first value is the total geometric length of a file (unit -> pixel),
                 second value is array of all the strokes lengths
        """
        dist = 0.0
        arr = []
        for stroke in self._data:
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

    def active_time(self):
        """
        :return: total active time (without pauses time)
        """
        return self._data[-1].get_feature('time')[-1]

    def get_strokes_as_one(self):
        """
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

    def feature_vs_time(self, feature, unit):
        """
        :param feature:
        :param unit:
        :return:
        """
        time = []
        y = []
        for stroke in self.get_data():
            time.append(stroke.get_feature("time")[0])
            if feature == "length":
                y.append(stroke.length())
            else:
                y.append(stroke.average(feature))

        time = np.asarray(time)
        y = np.asarray(y)
        plt.scatter(time, y, s=2)
        plt.title(feature + " mean is: " + str(y.mean()) + "\n" + feature + " std is: " + str(y.std()))
        plt.xlabel('time [sec]')
        plt.ylabel(feature + " [" + unit + "]")
        plt.show()

    def speed_vs_time(self):
        """
        plot graph of speed vs time
        """
        self.feature_vs_time("speed", "pixel/sec")

    def pressure_vs_time(self):
        """
        plot graph of pressure vs time
        """
        self.feature_vs_time("pressure", "?")

    def length_vs_time(self):
        """
        plot graph of length vs time
        """
        self.feature_vs_time("length", "pixel")

    def plot_picture(self):
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        for stroke in self._data:
            plt.plot(stroke.get_feature('x'), -stroke.get_feature('y'),
                     linewidth=10*stroke.average('pressure'),
                     color='black')

        plt.subplot(1, 2, 2)
        plt.imshow(mpimg.imread(self._ref_path))

        plt.show()


        # plt.figure(figsize=(24,8))
        # plt.subplot(1, 3, 1)
        # for stroke in self._data:
        #     plt.plot(stroke.get_feature('x'), -stroke.get_feature('y'), linewidth=1, color='black')
        #
        # plt.subplot(1, 3, 2)
        # plt.imshow(mpimg.imread(self._pic_path))
        #
        # plt.subplot(1, 3, 3)
        # plt.imshow(mpimg.imread(self._ref_path))
        #
        # plt.show()
        
        def plot_picture(self):
        w=6
        h=4
        f=3
        plt.figure(figsize=(f*2.5, f*h))
       
        for stroke in self._data:
            plt.plot(stroke.get_feature('x'), -stroke.get_feature('y'),
                     linewidth=3*stroke.average('pressure'),
                     color='black')

        plt.figure(figsize=(w, h))
        plt.imshow(mpimg.imread(self._ref_path))

        plt.show()
        
        plt.figure(figsize=(w, h))
        plt.imshow(mpimg.imread(self._pic_path))
