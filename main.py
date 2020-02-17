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
import simplify_cluster
import Distance

import os

if __name__ == "__main__":
    input1 = "data/D_01/aliza/aliza__130319_0935_D_01.txt"
    # input2 = "data/D_01/zoey/zoey__130319_1208_D_01.txt"
    # input3 = "data/F_05/aliza/aliza__040619_1842_F_05.txt"
    # draw = Analyzer.create_drawing(input1)
    # draw.plot_crop_image()
    # input3 = "data/F_05/aliza/aliza__040619_1842_F_05.txt"
    draw = Analyzer.create_drawing(input1)
    # draw.speed_vs_time(pause=True)
    # draw.length_vs_time()
    # draw.pressure_vs_time()
    # draw.plot_picture()
    # strokes = draw.get_data()
    # stroke_1, stroke_2 = strokes[0], strokes[2]
    # stroke_1, stroke_2 = np.array(stroke_1.get_data()[1:3]), np.array(stroke_2.get_data()[1:3])
    # print(draw.strokes_distance(stroke_1.T, stroke_2.T))
    # print(draw.strokes_angle_difference(stroke_1.T, stroke_2.T))

    person = Participant("aliza")
    draws = []
    for draw in person.get_data():
        draws.extend(draw.group_strokes(euc_dist_threshold=10, dist_threshold=5, ang_threshold=0.5)[1])

    # orig_draw, draws = draw.group_strokes(euc_dist_threshold=10, dist_threshold=5, ang_threshold=0.5)

    # strokes = []
    # for draw in draws:
    #     strokes.extend(draw.get_data())
    # # Analyzer.write_draw_to_file(orig_draw)
    # rebuilt_draw = Drawing(strokes, draws[0].get_ref_path(), draws[0].get_pic_path())
    # print(len(draws))
    # rebuilt_draw.plot_picture()
    # draws[0].plot_picture()

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
    # plt.show()
    # draws[3].plot_picture()
    # plt.show()

    # arr = []
    #
    # x = []
    # y = []
    # for stroke in draws[0].get_data():
    #     x.extend(stroke.get_feature('x'))
    #     y.extend(stroke.get_feature('y'))
    #
    # p = simplify_cluster.show_simplifican(x, y, 0, dist=10)
    # print(p)



    arr = []
    for i, draw in enumerate(draws):
        print("{0} out of {1}".format(i, len(draws)))
        x = []
        y = []
        for stroke in draw.get_data():
            x.extend(stroke.get_feature('x'))
            y.extend(stroke.get_feature('y'))

        p = simplify_cluster.show_simplifican(x, y, i, dist=10)
        if len(p) > 3:  # handle with very short simplify
            arr.append(p)

    p1 = np.array([[487.0, 342.5], [477.25, 337.5], [462.6, 349.4], [479.3333333333333, 348.6666666666667], [499.75, 343.875], [512.4, 343.4], [523.4285714285714, 342.42857142857144], [533.0, 344.0], [548.25, 347.75], [560.0, 352.4], [604.25, 363.75], [581.6666666666666, 367.8], [571.6666666666666, 360.3333333333333], [592.0, 366.0], [608.6666666666666, 357.0]])

    p = Distance.find_nearest_neighbor(p1, arr)

    plt.figure(1)
    plt.subplot(121)
    plt.plot(p1[:,0], p1[:,1], 'o', lw=0.5, ms=2, c='b')
    plt.subplot(122)
    plt.plot(p[:, 0], p[:, 1], 'o', lw=0.5, ms=2, c='r')
    plt.show()



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

    # Analyzer.simplify_folder('patches/B/val')
    # Analyzer.crop_folder('patches/B/val', 48, 48)

    # Analyzer.concat3image_directory('dataset/segments_overfit_20_pix2pix', 'overfit_20')

