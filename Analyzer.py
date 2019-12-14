import pandas as pd
from Stroke import Stroke
from Drawing import Drawing
import os
import numpy as np
import Constants
import math
# import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import canny_edge_detector as ced
from imageio import imread, imsave
from skimage.color import rgb2gray
from PIL import Image


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
        return np.array(array)

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
    def canny_edge_detector1(path, lowthreshold=50, highthreshold=250, save_pic=False, out='out1.jpg'):
        img = cv2.imread(path, 0)
        edges = cv2.Canny(img, lowthreshold, highthreshold)
        plt.subplot(121), plt.imshow(img, cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(edges, cmap='gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        if save_pic:
            plt.savefig(out)
        plt.show()

    @staticmethod
    def canny_edge_detector2(path, save_pic=False, out='out2.jpg'):
        img = mpimg.imread(path)
        img = rgb2gray(img)
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
            plt.savefig(out)
        plt.show()

    @staticmethod
    def split_image_to_patches(path, patch_w, patch_h, img_count=1):
        print(path)
        img = imread(path)
        img = np.asarray(rgb2gray(img))

        img_h = img.shape[0]
        img_w = img.shape[1]

        for i in range(0, img_h, patch_h):
            if i + patch_h >= img_h:
                break
            for j in range(0, img_w, patch_w):
                if j + patch_w >= img_w:
                    break
                temp = np.asarray(img[i:i+patch_h, j:j+patch_w])
                if len(np.where(temp == 1)[1]) != patch_h * patch_w:
                    imsave("patches/{0}.jpg".format(img_count), temp)
                    img_count += 1
                    # imsave("patches/row_{0}_col_{1}.jpg".format(i, j), temp)

        return img_count

    @staticmethod
    def png2jpg(path):
        # inverse
        img = imread(path)
        img = img[:, :, 3]
        img = 255 - img
        # img = np.where(img < 255, 0, 255)

        # saving
        arr = path.split('/')
        name = arr[-1]
        name = name[:-4]
        name += '.jpg'
        imsave('dataset/' + name, img)

    @staticmethod
    def png2jpg_folder(folder_path):
        for img_path in os.listdir(folder_path):
            Analyzer.png2jpg(folder_path + '/' + img_path)

    @staticmethod
    def simplify_folder(folder_path):
        process_counter = 1
        for img_path in os.listdir(folder_path):
            os.system("python3 sketch_simplification/simplify.py "
                      "                                          --img {0}/{1} "
                      "                                          --out sketch_simplification/simplify/{1} "
                      "                                          --model model_gan".format(folder_path, img_path))
            if process_counter % 100 == 0:
                print(process_counter, " out of ", len(os.listdir(folder_path)))
            process_counter += 1

    @staticmethod
    def crop(img_path, w, h):
        img = imread(img_path)
        new_img = img[0:h, 0:w]
        imsave(img_path, new_img)

    @staticmethod
    def crop_folder(folder_path, w, h):
        process_counter = 1
        for img_path in os.listdir(folder_path):
            Analyzer.crop(os.path.join(folder_path, img_path), w, h)
            if process_counter % 100 == 0:
                print(process_counter, " out of ", len(os.listdir(folder_path)))
            process_counter += 1

    @staticmethod
    def renaming_files_in_folder(folder_path, even_odd):
        """
        for even numbering call with 0, for odd call with 1.
        """
        for img_path in os.listdir(folder_path):
            try:
                new_name = int((int(img_path[9:-4]) - even_odd) / 2)
            except:
                continue
            os.rename(r'{0}/{1}'.format(folder_path, img_path), r'{0}/{1}.png'.format(folder_path, new_name))

    @staticmethod
    def concat3image(path_fakeB, path_realA, path_realB, ):
        realA = imread(path_realA)
        realB = imread(path_realB)
        fakeB = imread(path_fakeB)
        realAB = np.hstack((realA, realB))
        return np.hstack((realAB, fakeB))

    @staticmethod
    def concat3image_directory(folder_path):
        """
        for even numbering call with 0, for odd call with 1.
        """
        list = sorted(os.listdir(folder_path))
        for i, img_path in enumerate(list):
            if i % 3 == 0:
                img = Analyzer.concat3image(os.path.join(folder_path, list[i]), os.path.join(folder_path, list[i+1]), os.path.join(folder_path, list[i+2]))
                plt.imsave('concat/{0}.png'.format(int(i/3)), img)
