from Analyzer import Analyzer
from Participant import Participant
from PIL import Image
from matplotlib.pyplot import imread, imshow, imsave
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage import feature
from scipy.misc import imread, imshow
from skimage.color import rgb2gray
import utils
import canny_edge_detector as ced


if __name__ == "__main__":
    # input1 = "data/D_01/aliza/aliza__130319_0935_D_01.txt"
    # input2 = "data/D_01/zoey/zoey__130319_1208_D_01.txt"
    # input3 = "data/F_05/aliza/aliza__040619_1842_F_05.txt"
    # draw = Analyzer.create_drawing(input1)
    # draw.plot_crop_image()
    # draw.speed_vs_time(pause=True)
    # draw.length_vs_time()
    # draw.pressure_vs_time()
    # draw.plot_picture()
    # person = Participant("aliza")
    # person.plot_participant_pictures()

    Analyzer.canny_edge_detector1('clean_refs_pics/F01_stroke.jpg', save_pic=True, out='out3.jpg')
    Analyzer.canny_edge_detector2('clean_refs_pics/F01_stroke.jpg', save_pic=True, out='out4.jpg')




