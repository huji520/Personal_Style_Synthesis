from Analyzer import Analyzer
from Drawing import Drawing
from Participant import Participant
from PIL import Image
from matplotlib.pyplot import imshow, imsave
import matplotlib.pyplot as plt
import numpy as np
# import cv2
from scipy import ndimage as ndi
from skimage import feature
from imageio import imread
from skimage.color import rgb2gray
import utils
import canny_edge_detector as ced

import os

if __name__ == "__main__":
    input1 = "data/D_01/aliza/aliza__130319_0935_D_01.txt"
    input2 = "data/D_01/zoey/zoey__130319_1208_D_01.txt"
    input3 = "data/F_05/aliza/aliza__040619_1842_F_05.txt"
    draw = Analyzer.create_drawing(input1)
    # draw.plot_crop_image()
    # input3 = "data/F_05/aliza/aliza__040619_1842_F_05.txt"
    # draw = Analyzer.create_drawing(input1)
    # draw.speed_vs_time(pause=True)
    # draw.length_vs_time()
    # draw.pressure_vs_time()
    # draw.plot_picture()
    # strokes = draw.get_data()
    # stroke_1, stroke_2 = strokes[0], strokes[2]
    # stroke_1, stroke_2 = np.array(stroke_1.get_data()[1:3]), np.array(stroke_2.get_data()[1:3])
    # print(draw.strokes_distance(stroke_1.T, stroke_2.T))
    # print(draw.strokes_angle_difference(stroke_1.T, stroke_2.T))

    orig_draw, draws = draw.group_strokes(50, 10, 0.5)
    strokes = []
    for draw in draws:
        strokes.extend(draw.get_data())
    Analyzer.write_draw_to_file(orig_draw)
    rebuilt_draw = Drawing(strokes, draws[0].get_ref_path(), draws[0].get_pic_path())
    print(len(draws))
    rebuilt_draw.plot_picture()
    # draws[4].plot_picture()
    # for i in range(10):
    #     draws[i].plot_picture()
    # plt.show()
    # plt.figure()
    # draws1 = draw.group_strokes(50, 10, 0.5)
    # strokes1 = []
    # for draw in draws1:
    #     strokes.extend(draw.get_data())
    # rebuilt_draw1 = Drawing(strokes, draws[0].get_ref_path(), draws[0].get_pic_path())
    # rebuilt_draw1.plot_picture()
    plt.show()
    # draws[3].plot_picture()
    # plt.show()


    # Analyzer.canny_edge_detector1('clean_refs_pics/F01_stroke.jpg', save_pic=True, out='out3.jpg')
    # Analyzer.canny_edge_detector2('clean_refs_pics/F01_stroke.jpg', save_pic=True, out='out4.jpg')

    # person = Participant("aliza")
    # person.split_all_participant_picture(50, 50)
    # person.export_data()

    # Analyzer.split_image_to_patches('out1.jpg', patch_w=40, patch_h=30)
    # Analyzer.png2jpg_folder('dataset/A/png')
    # Analyzer.split_image_to_patches('aliza/D01.jpg', 50, 50)
    #
    # img = imread('patches/1.jpg')
    # print(img.shape)
    # a = np.where(img == 255)
    # print(len(a[1]))

    # Analyzer.simplify_folder('patches')
