import Constants
import Analyzer


class Stroke:
    def __init__(self, data, pause=False):
        """
        :param data: list of np.array, every array represent a feature
        :param pause: true iff this is pause stroke
        """
        self._data = data
        self._pause = pause

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
