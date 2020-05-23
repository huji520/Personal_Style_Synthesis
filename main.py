from Analyzer import Analyzer
from Drawing import Drawing
import simplify_cluster
from Participant import Participant
import numpy as np
import pickle
import os
import nn
import tensorflow as tf
import matplotlib.pyplot as plt
import copy


def normalize_center_of_mass(points):
    shift_x = points[:, 0].mean()
    shift_y = points[:, 1].mean()
    points[:, 0] -= shift_x
    points[:, 1] -= shift_y
    return shift_x, shift_y


def get_simplified_draw(clusters, already_simplified=False):
    """
    Get array of clusters, making every cluster to be an array of 2D points, then simplify this array (if needed).
    :param clusters: Array of clusters (which are Drawing objects)
    :param already_simplified: If the clusters are already simplify, no need to use simplify_cluster function.
    :return: Array of 2D arrays.
    """
    print("Start simplify the given input")
    simplified_clusters = []
    indexes = []
    for i, draw2 in enumerate(clusters):
        add_orig = True
        print("{0} out of {1}".format(i, len(clusters)))
        x = []
        y = []
        for stroke in draw2.get_data():
            x.extend(stroke.get_feature('x'))
            y.extend(stroke.get_feature('y'))

        # Case which the input should be simplify, like the participants inputs
        if not already_simplified:
            p, num_of_stroke_in_simplify = simplify_cluster.simplify_cluster(x, y)
            if 3 < len(p) < 1000 and num_of_stroke_in_simplify == 1:
                indexes.append(i)
                add_orig = False

        if add_orig:
            simplified_clusters.append(np.stack((x,y), axis=1))
        else:
            simplified_clusters.append(p)
    print("End simplify the given input\n")
    return simplified_clusters, indexes


def get_participant(person_name, load_person, stroke_length=None):
    """
    Get participant object.
    :param stroke_length:
    :param person_name: The name of the participant
    :param load_person: Use pickle if True, else create a new one
    :return: Participant object
    """
    pickle_person_path = os.path.join("pickle", "participant", f"{person_name}_{stroke_length if stroke_length else ''}.p")
    if load_person:
        participant = pickle.load(open(pickle_person_path, "rb"))
    else:
        print("Start creating a new Participant")
        if stroke_length:
            participant = Participant(person_name, stroke_length=stroke_length)
        else:
            participant = Participant(person_name)
        pickle.dump(participant, open(pickle_person_path, "wb"))
        print("End creating a new Participant\n")

    return participant


def transfer_style(draw, person_name, load_person=False, load_dict=True, already_simplified=False,
                   euc_dist_threshold=10, dist_threshold=5, ang_threshold=0.5, matching_threshold=5,
                   min_length=3, max_length=1000, stroke_length=None, simplify_size=None):
    """
    Get a drawing and person name, and generate a new draw with the style of the given person name.
    :param simplify_size: length of the simplify strokes (keys of the dict)
    :param stroke_length: length of the strokes of the participant that will be in the clusters
    :param max_length: max length of stroke after simplify to add the dict
    :param min_length: min length of stroke after simplify to add the dict
    :param matching_threshold: If the error of the best match if bigger than that threshold, do not change the cluster
    :param ang_threshold: argument for group_strokes
    :param dist_threshold: argument for group_strokes
    :param euc_dist_threshold: argument for group_strokes
    :param already_simplified: True if the given drawing is already simplify
    :param load_dict: Use pickle if True, else create a new one
    :param load_person: Use pickle if True, else create a new one
    :param draw: a draw to be transfer
    :param person_name: the name of the person with the wanted style
    :return: a draw object with the wanted style
    """
    participant = get_participant(person_name, load_person, stroke_length)
    clusters = draw.group_strokes(euc_dist_threshold, dist_threshold, ang_threshold)[1]
    simplified_clusters = get_simplified_draw(clusters, already_simplified=already_simplified)[0]

    print("Start matching")
    for i in range(len(clusters)):
        print(f"{i} out of {len(clusters)}")
        matched_cluster, x_shift, y_shift, error = participant.searching_match_on_person(
            np.array(simplified_clusters[i]), load_dict, euc_dist_threshold, dist_threshold, ang_threshold,
            min_length, max_length, simplify_size)
        load_dict = True

        if error < matching_threshold:
            clusters[i] = matched_cluster
            clusters[i].shift_x(x_shift)
            clusters[i].shift_y(y_shift)
    print("End matching")

    strokes = []
    for cluster in clusters:
        strokes.extend(cluster.get_data())

    return Drawing(strokes, clusters[0].get_pic_path())


def transfer_style2(draw, already_simplified, simplify_size, euc_dist_threshold, dist_threshold, ang_threshold):

    model = nn.MyModel()
    model.load_weights(f"results/weights/200_reg.tf")
    # nn.train_model(model=model, epochs=200, loss_object=nn.loss_object_func, save=False)

    clusters = draw.group_strokes(euc_dist_threshold, dist_threshold, ang_threshold)[1]
    simplified_clusters, indexes = get_simplified_draw(clusters, already_simplified)

    # for sim in simplified_clusters:
    #     plt.plot(np.array(sim)[:, 0], np.array(sim)[:, 1])

    # for j, sim in enumerate(simplified_clusters):
    #     if j in indexes:
    #         plt.plot(np.array(sim)[:,0], np.array(sim)[:,1])
    #     else:
    #         clusters[j].plot_picture(plt_show=False)

    # for cluster in clusters:
    #     for stroke in cluster.get_data():
    #         plt.plot(stroke.get_feature('x'), stroke.get_feature('y'), color='black')
    # plt.gca().invert_yaxis()
    # plt.show()
    # exit(1)
    new_draw = []
    for i, simplify in enumerate(simplified_clusters):
        if i in indexes:
            simplify = list(simplify)
            Analyzer.set_size(simplify, simplify_size)
            simplify = np.array(simplify)
            shift_x, shift_y = normalize_center_of_mass(simplify)
            reconstruction = model(simplify[tf.newaxis, :, :]).numpy().squeeze()
            reconstruction[:, :, 0] += shift_x
            reconstruction[:, :, 1] += shift_y
            new_draw.append(reconstruction)
        else:
            clusters[i].plot_picture(plt_show=False)

    for cluster in new_draw:
        for stroke in cluster:
            plt.plot(np.array(stroke)[:,0], np.array(stroke)[:,1], c='black')

    # plt.gca().invert_yaxis()
    plt.show()


def plot_clusters(draw, euc_dist_threshold, dist_threshold, ang_threshold):
    """
    Plot the input draw after clustering
    :param ang_threshold: argument for group_strokes
    :param dist_threshold: argument for group_strokes
    :param euc_dist_threshold: argument for group_strokes
    :param draw: Drawing object
    """
    clusters = draw.group_strokes(euc_dist_threshold, dist_threshold, ang_threshold)[1]
    strokes = []
    for cluster in clusters:
        strokes.extend(cluster.get_data())
    rebuilt_draw = Drawing(strokes, clusters[0].get_pic_path())
    rebuilt_draw.plot_picture(show_clusters=True)


def save_dict_for_nn(base_path, x_output_path, y_output_path, units=0, put_zeros=False):
    simplify_path = os.path.join("pickle", "simplify", base_path)
    person_clusters_path = os.path.join("pickle", "clusters", base_path)
    simplify_clusters = pickle.load(open(simplify_path, "rb"))
    person_clusters = pickle.load(open(person_clusters_path, "rb"))

    # Rotation
    if units > 0:
        print("Start rotate simplify")
        simplify_clusters_new = []
        step = np.ceil(360/units)
        for i, sim in enumerate(simplify_clusters):
            print(f"{i} out of {len(simplify_clusters)}")
            for ang in range(0, 360, step):
                simplify_clusters_new.append(Analyzer.rotate(copy.deepcopy(sim), ang))
        print("End rotate simplify")

        print("Start rotate clusters")
        person_clusters_new = []
        for i, cluster in enumerate(person_clusters):
            print(f"{i} out of {len(person_clusters)}")
            person_clusters_new.append(copy.deepcopy(cluster))
            for ang in range(units-1):
                cluster.rotate(step)
                person_clusters_new.append(copy.deepcopy(cluster))
        print("End rotate clusters")
    else:
        simplify_clusters_new = simplify_clusters
        person_clusters_new = person_clusters

    # Fix shapes for nn
    print("Start fix the shapes")
    simplify_clusters_shape = np.zeros(shape=(len(simplify_clusters_new), 40, 2))
    for i, stroke in enumerate(simplify_clusters_new):
        for j, point in enumerate(stroke):
            simplify_clusters_shape[i][j] = point

    person_clusters_shape = np.zeros(shape=(len(person_clusters_new), 5, 20, 2))
    for i, cluster in enumerate(person_clusters_new):
        for j, stroke in enumerate(cluster.get_data()):
            pairs = np.stack((stroke.get_feature('x'), stroke.get_feature('y')), axis=1)
            for k, point in enumerate(pairs):
                person_clusters_shape[i][j][k] = point
    print("End fix the shapes")

    # normalize
    print("Start normalize (center of mass at (0,0)")
    # Put center of mass at (0,0)
    for sim in simplify_clusters_shape:
        sim[:, 0] -= sim[:, 0].mean()
        sim[:, 1] -= sim[:, 1].mean()

    for per in person_clusters_shape:
        per[:, :, 0] -= per[:, :, 0].mean()
        per[:, :, 1] -= per[:, :, 1].mean()
    print("End normalize (center of mass at (0,0)")

    # Put zeros instead of duplicates
    if put_zeros:
        print("Start pushing zeros instead of duplicates")
        for cluster in person_clusters_shape:
            for i in range(1, len(cluster)):
                if np.all(cluster[i] == cluster[0]):
                    cluster[i][:] = 0
        print("End pushing zeros instead of duplicates")

    pickle.dump(simplify_clusters_shape, open(x_output_path, "wb"))
    pickle.dump(person_clusters_shape, open(y_output_path, "wb"))


if __name__ == "__main__":

    input_banana = "example_input/testdata banana.txt"
    input_fish = "example_input/testdata fish.txt"
    input1 = "data/D_01/aliza/aliza__130319_0935_D_01.txt"

    # aliza = get_participant("aliza", load_person=False, stroke_length=20)
    # aliza.create_dict(euc_dist_threshold=40, dist_threshold=10, ang_threshold=0.5, simplify_size=40)

    draw = Analyzer.create_drawing(input1)
    # draw = Analyzer.create_drawing(input_banana, orig_data=False)
    # draw.plot_picture()
    transfer_style2(draw, already_simplified=False, simplify_size=40, euc_dist_threshold=40, dist_threshold=10, ang_threshold=0.5)
    # new_draw.plot_picture(show_clusters=False)
    # new_draw = transfer_style(draw, "aliza", load_person=False, load_dict=False, already_simplified=True, stroke_length=20, simplify_size=20)
    # new_draw.plot_picture(show_clusters=False)

    # base_path = f"aliza_40_10_0.5_stroke_length_20.p"
    # y_path = 'y/y40_10_simplify_1_40_rotation_360.p'
    # x_path = 'x/x40_10_simplify_1_40_rotation_360.p'
    # save_dict_for_nn(base_path, x_path, y_path, rotation=True)

    # aliza = Participant("aliza", stroke_length=20)
    # draws = aliza.get_all_files_of_participant()
    # pickle.dump(draws, open("aliza_draws.p", "wb"))
    # aliza_draws = pickle.load(open("aliza_draws.p", "rb"))[0]


