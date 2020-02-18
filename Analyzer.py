import pandas as pd
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from imageio import imread, imsave
from skimage.color import rgb2gray
from skimage.transform import resize
import Constants
from Stroke import Stroke
from Drawing import Drawing


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
    def split_image_to_patches(path, patch_w, patch_h, img_count=1):
        """
        @ maybe can be removed
        split the image to patches
        :param path: path of the image
        :param patch_w: width of the patch
        :param patch_h: height of the patch
        :param img_count: integer
        :return: img_count
        """
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
    def simplify_folder(folder_path):
        """
        simplify all the pictures in the given folder (using sketch_simplification algorithm)
        :param folder_path:
        :return:
        """
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
    def resize(img_path, factor=2):
        """
        resize the given picture by factor
        :param img_path: path to the image
        :param factor: factor for scaling
        """
        image = imread(img_path)
        image_resized = resize(image, (image.shape[0] // factor, image.shape[1] // factor), anti_aliasing=True)
        imsave(img_path, image_resized)

    @staticmethod
    def resize_folder(folder_path, factor):
        """
        resize all the pictures in the given folder by the same factor
        :param folder_path: path to the folder
        :param factor: factor for scaling
        """
        process_counter = 1
        for img_path in os.listdir(folder_path):
            Analyzer.resize(os.path.join(folder_path, img_path), factor)
            print(process_counter, " out of ", len(os.listdir(folder_path)))
            process_counter += 1

    @staticmethod
    def concat3image(path_fakeB, path_realA, path_realB, ):
        """
        @TODO: titles for every picture
        :param path_fakeB: path to the fake picture
        :param path_realA: path to the real picture
        :param path_realB: path to the predict picture
        :return: the concat image
        """
        realA = imread(path_realA)
        realB = imread(path_realB)
        fakeB = imread(path_fakeB)
        realAB = np.hstack((realA, realB))
        return np.hstack((realAB, fakeB))

    @staticmethod
    def concat3image_directory(folder_path, new_folder_name=None):
        """
        concat every 3 images in the folder. assuming the picture names are numbers.
        :param folder_path: path to the folder
        :param new_folder_name: path to the output folder (optional)
        """
        if new_folder_name is None:
            path = folder_path.split('/')
            new_folder_name = path[-1]

        if not os.path.exists(os.path.join('concat', new_folder_name)):
            os.mkdir(os.path.join('concat', new_folder_name))

        list = sorted(os.listdir(folder_path))
        for i, img_path in enumerate(list):
            if i % 3 == 0:
                img = Analyzer.concat3image(os.path.join(folder_path, list[i]), os.path.join(folder_path, list[i+1]), os.path.join(folder_path, list[i+2]))
                plt.imsave('concat/{0}/{1}.png'.format(new_folder_name, int(i/3)), img)

    def write_draw_to_file(draw):
        """
        Tal:

        :return:
        """
        print(draw.get_ref_path())
        name = "clustered_draws/" + draw.get_ref_path().split('/')[-1].split('.')[-2] + ".txt"
        f = open(name, 'w')
        f.write("time\tx\ty\tpressure\ttiltX\ttiltY\tazimuth\tsidePressure\trotation\tcluster\n")
        for stroke in draw.get_data():
            if stroke.is_pause():
                continue
            data = stroke.get_data()
            for i in range(len(data[0])):
                for j in range(len(data)):
                    f.write(str(data[j][i]) + "\t")
                f.write(str(stroke._group) + "\n")
            # np.savetxt(name, np.array(stroke.get_data()).T)
            f.write("\n\n")
        f.close()