import pandas as pd
from Stroke import Stroke
from Drawing import Drawing
import os
import numpy as np
import Constants
import math
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import canny_edge_detector as ced


class Analyzer:
    @staticmethod
    def get_data_file(path):
        """
        :param path: string
        :return: DataFrame object
        """
        data = pd.read_csv(path, delimiter='\t')
        return data.loc[data['time'].shift() != data['time']]  # removing duplicates pauses

    @staticmethod
    def get_pause_data(size):
        """
        :param size: number of sampling in the pause-stroke (integer)
        :return: array of features (every feature is np.array)
        """
        return [np.zeros(size), np.full(size, -1), np.full(size, -1), np.zeros(size), np.full(size, -1),
                np.full(size, -1), np.full(size, -1), np.full(size, -1), np.full(size, -1)]

    @staticmethod
    def update_time(stroke, current_time):
        """
        Updating the times to get uniform time intervals
        :param stroke: array the represent the times of the stroke
        :param current_time: the current time in the drawing
        :return: the updated time in the drawing
        """
        for j in range(len(stroke[0])):
            stroke[Constants.TIME][j] = current_time
            current_time += Constants.TIME_STAMP  # TIME_STAMP = 0.017
        return current_time

    @staticmethod
    def get_strokes(data):
        """
        :param data: DataFrame object
        :return: list of strokes that represent the data
        """
        stroke_list = []
        stroke = []
        start_stroke_location = 0  # the location of the next stroke in data
        current_time = 0.0

        for i, time in enumerate(data['time']):
            if time == 'Pause':
                # getting threshold for the size of stroke
                if i - start_stroke_location < Constants.MIN_SIZE_OF_STROKE:  # MIN_SIZE_OF_STROKE = 2
                    start_stroke_location = i + 1
                    continue

                # creating the data for the new stroke that will be create
                for index, feature in enumerate(data):
                    stroke.append(Analyzer.string2float_array(data[feature][start_stroke_location:i]))
                    if index == Constants.TIME:
                        current_time = Analyzer.update_time(stroke, current_time)

                start_stroke_location = i + 1
                stroke_list.append(Stroke(stroke))
                stroke = []

                if i + 2 == len(data['time']):  # end of file
                    break

                # creating the data for the new pause-stroke
                pause_time = float(list(data['time'])[i+1]) - float(list(data['time'])[i-1])
                pause_data = Analyzer.get_pause_data(int(pause_time / Constants.TIME_STAMP))
                current_time = Analyzer.update_time(pause_data, current_time)
                stroke_list.append(Stroke(pause_data, pause=True))

        return stroke_list

    @staticmethod
    def string2float_array(string_array):
        """
        :param string_array: array with numbers represents by strings
        :return: numpy array with float values instead of strings
        """
        array = []
        for value in string_array:
            array.append(float(value))
        return np.asarray(array)

    @staticmethod
    def get_ref_path(data):
        """
        :param data: DataFrame object
        :return: path to reference picture
        """
        ref_name = list(data['time'])[-1].split(' ')[1]
        return "ref_pics_crop/" + ref_name + ".JPG"

    @staticmethod
    def get_pic_path(path):
        """
        :param path: string
        :return: path to the drawing
        """
        ref_folder = path.split('/')[1]
        participant_name = path.split('/')[2]
        for file in os.listdir("data/" + ref_folder + "/" + participant_name):
            if "DS_Store" not in file:
                if file.endswith(".png"):
                    return "data/" + ref_folder + "/" + participant_name + "/" + file

    @staticmethod
    def create_drawing(path):
        """
        :param path: path to the data file (.txt)
        :return: Drawing object.
        """
        data = Analyzer.get_data_file(path)
        pic_path = Analyzer.get_pic_path(path)
        ref_path = Analyzer.get_ref_path(data)
        if pic_path is not None and ref_path is not None:
            return Drawing(Analyzer.get_strokes(data), pic_path, ref_path)
        else:
            print("Error: missing data")

    @staticmethod
    def calc_dist_2D(old_x, new_x, old_y, new_y):
        """
        :return: the distance between two points - 2D
        """
        return math.sqrt(float(pow(new_x - old_x, 2)) + float(pow(new_y - old_y, 2)))

    @staticmethod
    def rgb2gray(rgb):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    @staticmethod
    def canny_edge_detector1(path, lowthreshold=100, highthreshold=250, save_pic=False):
        img = cv2.imread(path, 0)
        edges = cv2.Canny(img, lowthreshold, highthreshold)
        plt.subplot(121), plt.imshow(img, cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(edges, cmap='gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        if save_pic:
            plt.savefig('out1.jpg')
        plt.show()

    @staticmethod
    def canny_edge_detector2(path, save_pic=False):
        img = mpimg.imread(path)
        img = Analyzer.rgb2gray(img)
        plt.subplot(121), plt.imshow(img, 'gray'), plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        detector = ced.cannyEdgeDetector([img],
                                         sigma=1.4,
                                         kernel_size=5,
                                         lowthreshold=0.09,
                                         highthreshold=0.22,
                                         weak_pixel=100)
        detect_image = detector.detect()
        plt.subplot(122), plt.imshow(detect_image[0], 'gray'), plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        if save_pic:
            plt.savefig('out2.jpg')
        plt.show()
