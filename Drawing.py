import numpy as np
import copy
from Stroke import Stroke
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import imread, imshow, imsave
import Constants
from PIL import Image


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

    # def active_time(self):
    #     """
    #     :return: total active time (without pauses time)
    #     """
    #     return self._data[-1].get_feature('time')[-1]

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

    def feature_vs_time(self, feature, unit, pause):
        """
        plot graph, given feature as function of time
        :param feature: given feature for 'y' axis
        :param unit:
        :param pause: true iff pauses strokes will shown in the plot
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
        plt.title(feature + " mean is: " + str(y.mean()) + "\n" + feature + " std is: " + str(y.std()))
        plt.xlabel('time [sec]')
        plt.ylabel(feature + " [" + unit + "]")
        plt.show()

    def speed_vs_time(self, pause=False):
        """
        plot graph of speed vs time
        """
        self.feature_vs_time("speed", "pixel/sec", pause)

    def pressure_vs_time(self, pause=False):
        """
        plot graph of pressure vs time
        """
        self.feature_vs_time("pressure", "?", pause)

    def length_vs_time(self, pause=False):
        """
        plot graph of length vs time
        """
        self.feature_vs_time("length", "pixel", pause)

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

    def plot_crop_image(self):
        """
        plot the picture with only the active pixels (cropping)
        """
        img = Image.open(self.get_pic_path())
        locations = self.get_active_image_sizes()  # 0 -> start_x, 1 -> start_y, 2 -> end_x, 3 -> end_y
        img = img.crop((locations[0], locations[1], locations[2], locations[3]))
        img.thumbnail([locations[2] - locations[0], locations[3] - locations[1]], Image.ANTIALIAS)
        imshow(img)
        plt.show()

        # imagefile = open('out.png', 'wb')
        # img.save(imagefile, "png")
        # imagefile.close()

    def plot_picture(self):
        """
        plot the reference picture and the actual drawing.
        @TODO: pictures should be in the same resolution.
        """
        # plt.figure(figsize=(20, 10))
        # plt.subplot(1, 2, 1)
        #
        # # y axis got minus to deal with the reflection.
        # for stroke in self._data:
        #     plt.plot(stroke.get_feature('x'), (-stroke.get_feature('y')),
        #              linewidth=3*stroke.average('pressure'),
        #              color='black')
        #
        # plt.subplot(1, 2, 2)
        # plt.imshow(mpimg.imread(self._ref_path))

        # plt.show()


        w = 6
        h = 4
        f = 3

        # drawing
        plt.figure(figsize=(f * 2.5, f * h))
        for stroke in self._data:
            plt.plot(stroke.get_feature('x'), Constants.DRAWING_HEIGHT - stroke.get_feature('y'),
                     linewidth=3 * stroke.average('pressure'),
                     color='black')

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



        # plt.figure(figsize=(w, h))
        # plt.imshow(mpimg.imread(self._ref_path))

        # plt.figure(figsize=(w, h))
        # plt.imshow(mpimg.imread(self._pic_path))
        # plt.show()
