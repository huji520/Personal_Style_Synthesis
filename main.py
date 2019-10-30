from Analyzer import Analyzer
from Participant import Participant
from PIL import Image
from matplotlib.pyplot import imread, imshow, imsave
import matplotlib.pyplot as plt
import numpy as np
# import cv2
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

    # im = imread('ref_pics_crop/D01.JPG').astype(np.float64)
    # im = rgb2gray(im)

    # import cv2
    # import numpy as np
    # from matplotlib import pyplot as plt
    #
    # img = cv2.imread('ref_pics_crop/D01.JPG', 0)
    # edges = cv2.Canny(img, 100, 200)
    #
    # plt.subplot(121), plt.imshow(img, cmap='gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(edges, cmap='gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    #
    # plt.show()

    imgs = utils.load_data()
    utils.visualize(imgs, 'gray')
    detector = ced.cannyEdgeDetector(imgs, sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.17,
                                     weak_pixel=100)
    imgs_final = detector.detect()
    utils.visualize(imgs_final, 'gray')




