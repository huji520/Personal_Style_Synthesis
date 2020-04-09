import Constants
import numpy as np
import Analyzer


class Stroke:
    def __init__(self, data, pause=False):
        """
        :param data: list of np.array, every array represent a feature
        :param pause: true iff this is pause stroke
        """
        self._data = data
        self._pause = pause
        self._group = 0
        self._color = 0

    def is_pause(self):
        """
        :return: true iff this is pause stroke
        """
        return self._pause

    def get_data(self):
        """
        :return: list of np.array, every array represent a feature
        """
        return self._data

    def size(self):
        """
        :return: number of stamps in a stroke
        """
        return len(self._data[0])

    def get_feature(self, feature):
        """
        :param feature: string
        :return: np.array of the given feature
        """
        if feature == "time":
            return self._data[Constants.TIME]
        elif feature == "x":
            return self._data[Constants.X]
        elif feature == "y":
            return self._data[Constants.Y]
        elif feature == "pressure":
            return self._data[Constants.PRESSURE]
        elif feature == "tiltX":
            return self._data[Constants.TILT_X]
        elif feature == "tiltY":
            return self._data[Constants.TILT_Y]
        elif feature == "azimuth":
            return self._data[Constants.AZIMUTH]
        elif feature == "sidePressure":
            return self._data[Constants.SIDE_PRESSURE]
        elif feature == "rotation":
            return self._data[Constants.ROTATION]
        else:
            print("Not a valid feature for this function\n"
                  "Try: 'time', 'x', 'y', 'pressure', 'tiltX', 'tiltY', 'azimuth', 'sidePressure', 'rotation'")
            return

    def average(self, feature):
        """
        :param feature: string
        :return: average of the given feature
        """
        if feature == "speed":
            return self.length() / self.time()
        elif feature == "pressure":
            return self.get_feature("pressure").mean()
        elif feature == "tiltX":
            return self.get_feature("tiltX").mean()
        elif feature == "tiltY":
            return self.get_feature("tiltY").mean()
        elif feature == "azimuth":
            return self.get_feature("azimuth").mean()
        elif feature == "sidePressure":
            return self.get_feature("sidePressure").mean()
        elif feature == "rotation":
            return self.get_feature("rotation").mean()
        else:
            print("Not a valid feature for this function\n"
                  "Try: 'speed', 'pressure', 'tiltX', 'tiltY', 'azimuth', 'sidePressure', 'rotation'")
            return

    def length(self):
        """
        :return: the total geometric length of a stroke (unit -> pixel)
        """
        x_array = self.get_feature("x")
        y_array = self.get_feature("y")
        dist = 0.0
        for i in range(1, len(x_array)):
            dist += Analyzer.Analyzer.calc_dist_2D(x_array[i - 1], x_array[i], y_array[i - 1], y_array[i])

        return dist

    def time(self):
        """
        :return: the total time of a stroke
        """
        time_array = self.get_feature("time")
        time = 0.0
        for i in range(1, len(time_array)):
            time += (time_array[i] - time_array[i-1])

        return time

    def set_x(self, shift):
        """
        Shifting the x values of the stroke
        :param shift: the value to shift the x values
        """
        self._data[Constants.X] += shift

    def set_y(self, shift):
        """
        Shifting the y values of the stroke
        :param shift: the value to shift the y values
        """
        self._data[Constants.Y] += shift

    def rotate(self, angle):
        """
        rotate the stroke in the given angle
        :param angle: degrees (int)
        """
        angle = np.deg2rad(angle)
        points = np.stack((self._data[Constants.X], self._data[Constants.Y]), axis=1)
        for i, point in enumerate(points):
            points[i] = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) @ point

        self._data[Constants.X] = points[:, 0]
        self._data[Constants.Y] = points[:, 1]

    def insert(self, idx, idx1=0, idx2=0):
        """
        Insert a new sample to the stroke
        :param idx: the index to insert sample
        """
        # Set this values to -1 since they are not in use
        self._data[Constants.TILT_X] = np.insert(self._data[Constants.TILT_X], idx, -1)
        self._data[Constants.TILT_Y] = np.insert(self._data[Constants.TILT_Y], idx, -1)
        self._data[Constants.AZIMUTH] = np.insert(self._data[Constants.AZIMUTH], idx, -1)
        self._data[Constants.SIDE_PRESSURE] = np.insert(self._data[Constants.SIDE_PRESSURE], idx, -1)
        self._data[Constants.ROTATION] = np.insert(self._data[Constants.ROTATION], idx, -1)

        # Set the value to be averaging of the two value which is insert between them
        time = (self._data[Constants.TIME][idx - 1] + self._data[Constants.TIME][idx]) / 2
        pressure = (self._data[Constants.PRESSURE][idx - 1] + self._data[Constants.PRESSURE][idx]) / 2
        x = (self._data[Constants.X][idx - 1] + self._data[Constants.X][idx]) / 2
        y = (self._data[Constants.Y][idx - 1] + self._data[Constants.Y][idx]) / 2
        self._data[Constants.TIME] = np.insert(self._data[Constants.TIME], idx, time)
        self._data[Constants.PRESSURE] = np.insert(self._data[Constants.PRESSURE], idx, pressure)
        self._data[Constants.X] = np.insert(self._data[Constants.X], idx, x)
        self._data[Constants.Y] = np.insert(self._data[Constants.Y], idx, y)

    def remove(self, idx):
        """
        Remove the given index from the stroke
        :param idx: The index the should be deleted
        """
        self._data[Constants.TIME] = np.delete(self._data[Constants.TIME], idx)
        self._data[Constants.TILT_X] = np.delete(self._data[Constants.TILT_X], idx)
        self._data[Constants.TILT_Y] = np.delete(self._data[Constants.TILT_Y], idx)
        self._data[Constants.AZIMUTH] = np.delete(self._data[Constants.AZIMUTH], idx)
        self._data[Constants.SIDE_PRESSURE] = np.delete(self._data[Constants.SIDE_PRESSURE], idx)
        self._data[Constants.ROTATION] = np.delete(self._data[Constants.ROTATION], idx)
        self._data[Constants.PRESSURE] = np.delete(self._data[Constants.PRESSURE], idx)
        self._data[Constants.X] = np.delete(self._data[Constants.X], idx)
        self._data[Constants.Y] = np.delete(self._data[Constants.Y], idx)

    def remove_and_replace(self, idx):
        """
        Remove the samples with indexes idx and idx-1, and instead of them insert new sample (averaging of them)
        Example: [1,2,3,4,5,6], idx=2 --> [1,2.5,4,5,6]
        :param idx: idx and idx-1 will be removed, idx-1 will be insert instead
        """
        self.insert(idx)
        self.remove(idx - 1)
        self.remove(idx)

    def __str__(self):
        """
        Using for print(Stroke) outside the class
        :return: The string that will be print
        """
        s = f"time\tx\t\ty\t\tpressure\ttiltX\t\ttiltY\t\tazimuth\t\tsidePressure\trotation\n"
        for i in range(self.size() - 1):
            s += f"{format(self._data[Constants.TIME][i], '.3f')}\t" \
                 f"{self._data[Constants.X][i]}\t" \
                 f"{self._data[Constants.Y][i]}\t" \
                 f"{format(self._data[Constants.PRESSURE][i], '.3f')}\t\t" \
                 f"{format(self._data[Constants.TILT_X][i], '.3f')}\t\t" \
                 f"{format(self._data[Constants.TILT_Y][i], '.3f')}\t\t" \
                 f"{format(self._data[Constants.AZIMUTH][i], '.3f')}\t\t" \
                 f"{format(self._data[Constants.SIDE_PRESSURE][i], '.3f')}\t\t\t" \
                 f"{format(self._data[Constants.ROTATION][i], '.3f')}\n"
        return s