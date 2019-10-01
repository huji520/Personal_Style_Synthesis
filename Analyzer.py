import pandas as pd
from Personal_Style_Synthesis.Stroke import Stroke
from Personal_Style_Synthesis.Drawing import Drawing
import os
import numpy as np
import Personal_Style_Synthesis.Constants as Constants


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
        return [np.zeros(size), np.full(size, -1), np.full(size, -1), np.zeros(size), np.full(size, -1),
                np.full(size, -1), np.full(size, -1), np.full(size, -1), np.full(size, -1)]

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
                if i - start_stroke_location < Constants.MIN_SIZE_OF_STROKE:  # MIN_SIZE_OF_STROKE = 2
                    start_stroke_location = i + 1
                    continue

                for index, feature in enumerate(data):
                    stroke.append(Analyzer.string2float_array(data[feature][start_stroke_location:i]))
                    if index == Constants.TIME:
                        for j in range(len(stroke[0])):
                            stroke[index][j] = current_time
                            current_time += Constants.TIME_STAMP  # TIME_STAMP = 0.017

                start_stroke_location = i + 1
                stroke_list.append(Stroke(stroke))
                stroke = []

                if i + 2 == len(data['time']):  # end of file
                    break

                pause_time = float(list(data['time'])[i+1]) - float(list(data['time'])[i-1])
                pause_data = Analyzer.get_pause_data(int(pause_time / Constants.TIME_STAMP))
                for j in range(len(pause_data[0])):
                    pause_data[Constants.TIME][j] = current_time
                    current_time += Constants.TIME_STAMP  # TIME_STAMP = 0.017
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
        return "ref_pics/" + ref_name + ".JPG"

    @staticmethod
    def get_pic_path(path):
        ref_folder = path.split('/')[1]
        participant_name = path.split('/')[2]
        for file in os.listdir("data/" + ref_folder + "/" + participant_name):
            if "DS_Store" not in file:
                if file.endswith(".png"):
                    return "data/" + ref_folder + "/" + participant_name + "/" + file

    @staticmethod
    def create_drawing(path):
        data = Analyzer.get_data_file(path)
        pic_path = Analyzer.get_pic_path(path)
        ref_path = Analyzer.get_ref_path(data)
        if pic_path is not None and ref_path is not None:
            return Drawing(Analyzer.get_strokes(data), pic_path, ref_path)


if __name__ == "__main__":
    input1 = "data/D_01/aliza/aliza__130319_0935_D_01.txt"
    # input2 = "data/D_01/zoey/zoey__130319_1208_D_01.txt"
    draw = Analyzer.create_drawing(input1)
    draw.speed_vs_time(pause=True)
    draw.length_vs_time()
    draw.pressure_vs_time()
    # draw.plot_picture()
    # from Participant import Participant
    # person = Participant("aliza")
    # person.plot_participant_pictures()
