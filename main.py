from Analyzer import Analyzer
from Drawing import Drawing
from Participant import Participant
from PIL import Image
from matplotlib.pyplot import imshow, imsave
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage import feature
from imageio import imread
from skimage.color import rgb2gray
import simplify_cluster
import nearest_neighbor
import os
import pickle
import copy


def searching_match_on_person(person_name, p1, load=True, euc_dist_threshold=10, dist_threshold=5, ang_threshold=0.5):
    base_path = "{0}_{1}_{2}_{3}.p".format(person_name, euc_dist_threshold, dist_threshold, ang_threshold)
    simplify_path = os.path.join("pickle", "simplify", base_path)
    style_path = os.path.join("pickle", "style", base_path)
    person_clusters_path = os.path.join("pickle", "clusters", base_path)
    if load:
        simplify_clusters = pickle.load(open(simplify_path, "rb"))
        style_clusters = pickle.load(open(style_path, "rb"))
        person_clusters = pickle.load((open(person_clusters_path, "rb")))
    else:
        person = Participant(person_name)
        simplify_clusters, style_clusters = person.simplify_all_clusters(euc_dist_threshold, dist_threshold, ang_threshold)
        person_clusters = person.clusters
        pickle.dump(simplify_clusters, open(simplify_path, "wb"))
        pickle.dump(style_clusters, open(style_path, "wb"))
        pickle.dump(person_clusters, open(person_clusters_path, "wb"))

    p, i, x_shift, y_shift = nearest_neighbor.find_nearest_neighbor(p1, simplify_clusters)
    # Analyzer.plot_clusters(orig=p1, simplify_cluster=p, style_cluster=style_clusters[i], scatter=True)
    # person_clusters[i].shift_x(x_shift)
    # person_clusters[i].shift_y(y_shift)
    return person_clusters[i], x_shift, y_shift



def plot_clustering(input_txt_path, euc_dist_threshold=10, dist_threshold=5, ang_threshold=0.5):
    draw = Analyzer.create_drawing(input_txt_path)
    orig_draw, draws = draw.group_strokes(euc_dist_threshold, dist_threshold, ang_threshold)
    strokes = []
    for draw in draws:
        strokes.extend(draw.get_data())
    # Analyzer.write_draw_to_file(orig_draw)
    rebuilt_draw = Drawing(strokes, draws[0].get_ref_path(), draws[0].get_pic_path())
    # print(len(draws))
    rebuilt_draw.plot_picture()
    # draws[0].plot_picture()


def plot_simplify(input_txt_path, euc_dist_threshold=10, dist_threshold=5, ang_threshold=0.5):
    draw = Analyzer.create_drawing(input_txt_path)
    draw.plot_simplify_drawing(euc_dist_threshold, dist_threshold, ang_threshold)


def get_simplified_draw(clusters):
    simplified_clusters = []

    for i, draw in enumerate(clusters):
        print("{0} out of {1}".format(i, len(clusters)))
        x = []
        y = []
        for stroke in draw.get_data():
            x.extend(stroke.get_feature('x'))
            y.extend(stroke.get_feature('y'))

        p = simplify_cluster.simplify_cluster(x, y, i, dist=10, save_pairs=False)
        if len(p) > 3:  # handle with very short simplify
            simplified_clusters.append(p)

    return simplified_clusters


def replace_cluster_with_cluster(draw):
    clusters = draw.group_strokes(euc_dist_threshold=10, dist_threshold=5, ang_threshold=0.5)[1]
    clusters[0].plot_picture()
    simplified_clusters = get_simplified_draw(clusters)
    clusters[0], x_shift, y_shift = searching_match_on_person("aliza", np.array(simplified_clusters[0]), load=True)
    clusters[0].shift_x(x_shift)
    clusters[0].shift_y(y_shift)

    clusters[0].plot_picture()








if __name__ == "__main__":
    input2 = "data/D_01/zoey/zoey__130319_1208_D_01.txt"
    input3 = "data/F_05/aliza/aliza__040619_1842_F_05.txt"


    # #####################
    input1 = "data/D_01/aliza/aliza__130319_0935_D_01.txt"
    # plot_clustering(input1, euc_dist_threshold=10, dist_threshold=5, ang_threshold=0.5)
    # #####################

    # #####################
    # p1 = np.array([[487.0, 342.5], [477.25, 337.5], [462.6, 349.4], [479.3333333333333, 348.6666666666667], [499.75, 343.875], [512.4, 343.4], [523.4285714285714, 342.42857142857144], [533.0, 344.0], [548.25, 347.75], [560.0, 352.4], [604.25, 363.75], [581.6666666666666, 367.8], [571.6666666666666, 360.3333333333333], [592.0, 366.0], [608.6666666666666, 357.0]])
    # searching_match_on_person("aliza", p1, load=True)
    # #####################

    ####################
    # input1 = "data/D_01/aliza/aliza__130319_0935_D_01.txt"
    # plot_simplify(input1, euc_dist_threshold=10, dist_threshold=10, ang_threshold=0.5)
    ####################


    replace_cluster_with_cluster(Analyzer.create_drawing(input2))
    # plot_clustering(input1, euc_dist_threshold=30)

    # @TODO: map[simplify] = cluster - done (in other way)
    # @TODO: replacing function
    # @TODO: problem with lines between points (order)







