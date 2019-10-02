#######################
# for now, not in use #
#######################

import numpy as np
import math


class StrokePause:
    def __init__(self, data, pause=False):
        """
        :param data: is pause==False, then data tuple of start and end time of the pause
        :param pause: boolean - the stroke is pause-stroke ot not
        """
        self._data = data
        self._pause = pause

    def is_pause(self):
        return self._pause

    def get_data(self):
        return self._data

    def size(self):
        """
        :return: number of stamps in a stroke
        """
        if self._pause is True:
            return int((float(self._data[1]) - float(self._data[0])) / 0.17)
        else:
            return len(self._data['time'])

    def time(self):
        """
        :return: numpy array of time values of the stroke
        """
        return self._get_array('time')

    def x(self):
        """
        :return: numpy array of x values of the stroke, in case of pause-stroke, fill -1
        """
        return self._get_array('x')

    def y(self):
        """
        :return: numpy array of y values of the stroke, in case of pause-stroke, fill -1
        """
        return self._get_array('y')

    def pressure(self):
        """
        :return: numpy array of pressure values of the stroke, in case of pause-stroke, fill 0
        """
        return self._get_array('pressure')

    def tiltX(self):
        """
        :return: numpy array of tiltX values of the stroke, in case of pause-stroke, fill -1
        """
        return self._get_array('tiltX')

    def tiltY(self):
        """
        :return: numpy array of tiltY values of the stroke, in case of pause-stroke, fill -1
        """
        return self._get_array('tiltY')

    def azimuth(self):
        """
        :return: numpy array of azimuth values of the stroke, in case of pause-stroke, fill -1
        """
        return self._get_array('azimuth')

    def sidePressure(self):
        """
        :return: numpy array of sidePressure values of the stroke, in case of pause-stroke, fill -1
        """
        return self._get_array('sidePressure')

    def rotation(self):
        """
        :return: numpy array of rotation values of the stroke, in case of pause-stroke, fill -1
        """
        return self._get_array('rotation')

    def total_length(self):
        """
        :return: the total geometric length of a stroke (unit -> pixel)
        """
        x_array = self.x()
        y_array = self.y()
        dist = 0.0
        for i in range(1, len(x_array)):
            dist += self.calc_dist_2D(x_array[i - 1], x_array[i], y_array[i - 1], y_array[i])

        return dist

    def total_time(self):
        """
        :return: the total time of a stroke
        """
        time_array = self.time()
        time = 0.0
        for i in range(1, len(time_array)):
            time += (time_array[i] - time_array[i-1])

        return time

    def average_speed(self):
        """
        :return: the average speed of a stroke
        """
        return self.total_length() / self.total_time()

    @staticmethod
    def calc_dist_2D(old_x, new_x, old_y, new_y):
        """
        :return: the distance between two points - 2D
        """
        return math.sqrt(float(pow(new_x - old_x, 2)) + float(pow(new_y - old_y, 2)))

    @staticmethod
    def _string2float_array(string_array):
        """
        :param string_array: array with numbers represents by strings
        :return: numpy array with float values instead of strings
        """
        array = []
        for value in string_array:
            array.append(float(value))
        return np.asarray(array)

    def _get_array(self, feature):
        """
        :param feature: string represent the feature of the data
        :return: numpy array with the values of the given feature
        """
        without_pauses_strokes = True  # just for now
        if without_pauses_strokes:
            return self._string2float_array(self._data[feature])

        else:
            if self._pause is False:
                return self._string2float_array(self._data[feature])
            elif feature == 'time':
                time_array = []
                for i in range(self.size()):
                    time_array.append(float(self._data[0]) + (i * 0.17))
                return np.asarray(time_array)
            elif feature == 'pressure':
                return np.full(self.size(), 0)
            else:
                return np.full(self.size(), -1)




