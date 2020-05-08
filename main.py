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
    for i, draw in enumerate(clusters):
        print("{0} out of {1}".format(i, len(clusters)))
        x = []
        y = []
        for stroke in draw.get_data():
            x.extend(stroke.get_feature('x'))
            y.extend(stroke.get_feature('y'))

        # Case of simplify input, like banana or fish
        if already_simplified:
            p = np.stack((x,y), axis=1)
        # Case which the input should be simplify, like the participants inputs
        else:
            p = simplify_cluster.simplify_cluster(x, y, dist=10)

        # This is a bug in simplify cluster, p should never be zero
        if len(p) == 0:
            simplified_clusters.append(np.stack((x,y), axis=1))
            continue
        simplified_clusters.append(p)
    print("End simplify the given input\n")
    return simplified_clusters


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
    :param simplify_size:
    :param stroke_length:
    :param max_length:
    :param min_length:
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
    simplified_clusters = get_simplified_draw(clusters, already_simplified=already_simplified)

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


def transfer_style2(draw, already_simplified=False, euc_dist_threshold=10, dist_threshold=5, ang_threshold=0.5):
    """
    Get a drawing and person name, and generate a new draw with the style of the given person name.
    :param simplify_size:
    :param stroke_length:
    :param max_length:
    :param min_length:
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
    model = nn.MyModel()
    nn.train_model(model, 100, nn.loss_object_func)
    arr = []
    simplified_clusters = []
    for stroke in draw.get_data():
        if not stroke.is_pause():
            simplified_clusters.append(np.stack((stroke.get_feature('x'), stroke.get_feature('y')), axis=1))

    for simplify in simplified_clusters:
        print(np.array(simplify).shape)
        shift_x, shift_y = normalize_center_of_mass(simplify)
        reconstruction = model(simplify[tf.newaxis, :, :]).numpy().squeeze()
        reconstruction[:, :, 0] += shift_x
        reconstruction[:, :, 1] += shift_y
        arr.extend(reconstruction)

    for stroke in arr:
        print(stroke)
        plt.plot(np.array(stroke)[:,0], np.array(stroke)[:,1])

    plt.gca().invert_yaxis()
    plt.show()

    # return Drawing(arr, draw.get_pic_path())

# def plot_clusters(draw, euc_dist_threshold=10, dist_threshold=5, ang_threshold=0.5):
#     """
#     Plot the input draw after clustering
#     :param ang_threshold: argument for group_strokes
#     :param dist_threshold: argument for group_strokes
#     :param euc_dist_threshold: argument for group_strokes
#     :param draw: Drawing object
#     """
#     clusters = draw.group_strokes(euc_dist_threshold, dist_threshold, ang_threshold)[1]
#     strokes = []
#     for cluster in clusters:
#         strokes.extend(cluster.get_data())
#     rebuilt_draw = Drawing(strokes, clusters[0].get_pic_path())
#     rebuilt_draw.plot_picture(show_clusters=True)


def l2(p1, p2):
    """
    calc l2 metric between two points
    :param p1: 2D point [x,y]
    :param p2: 2D point [x,y]
    :return: l2 metric between the two given points
    """
    return np.linalg.norm(np.array(p1) - np.array(p2))


def l2_stroke(s1, s2):
    error = 0
    for i in range(len(s1)):
        for j in range(len(s1[0])):
            error += l2(s1[i][j], s2[i][j])
    return error


if __name__ == "__main__":

    input_banana = "example_input/testdata banana.txt"
    input_fish = "example_input/testdata fish.txt"
    input1 = "data/D_01/aliza/aliza__130319_0935_D_01.txt"

    draw = Analyzer.create_drawing(input_banana, orig_data=False, stroke_size=20)
    new_draw = transfer_style2(draw, already_simplified=True)
    # new_draw.plot_picture(show_clusters=False)
    # new_draw = transfer_style(draw, "aliza", load_person=False, load_dict=False, already_simplified=True, stroke_length=20, simplify_size=20)
    # new_draw.plot_picture(show_clusters=False)

    # base_path = f"aliza_10_5_0.5_stroke_length_20.p"
    # simplify_path = os.path.join("pickle", "simplify", base_path)
    # person_clusters_path = os.path.join("pickle", "clusters", base_path)
    # simplify_clusters = pickle.load(open(simplify_path, "rb"))
    # person_clusters = pickle.load(open(person_clusters_path, "rb"))

    # simplify_clusters_shape = np.zeros(shape=(len(simplify_clusters), 20, 2))
    # for i, stroke in enumerate(simplify_clusters):
    #     for j, point in enumerate(stroke):
    #         simplify_clusters_shape[i][j] = point
    #
    # person_clusters_shape = np.zeros(shape=(len(person_clusters), 5, 20, 2))
    # for i, cluster in enumerate(person_clusters):
    #     for j, stroke in enumerate(cluster.get_data()):
    #         pairs = np.stack((stroke.get_feature('x'), stroke.get_feature('y')), axis=1)
    #         for k, point in enumerate(pairs):
    #             person_clusters_shape[i][j][k] = point
    #
    # pickle.dump(simplify_clusters_shape, open('x/x.p', "wb"))
    # pickle.dump(person_clusters_shape, open('y/y.p', "wb"))



    # import matplotlib.pyplot as plt
    # plt.subplot(121)
    # plt.title("simplify")
    # plt.plot(simplify_clusters_shape[48][:,0], simplify_clusters_shape[8][:,1])
    #
    # plt.subplot(122)
    # plt.title("cluster")
    # for l in range(5):
    #     plt.plot(person_clusters_shape[48][l][:, 0], person_clusters_shape[8][l][:, 1])
    # plt.show()



    # for simplify in simplify_clusters:
    #     print((simplify))
    # person_clusters = np.array(person_clusters)

    # print(simplify_clusters.shape)
    # print(person_clusters.shape)


    # import matplotlib.pyplot as plt
    # for i in range(200):
    #     plt.figure(i)
    #     plt.subplot(121)
    #     plt.title("simplify")
    #     plt.plot(np.array(simplify_clusters[i])[:,0], np.array(simplify_clusters[i])[:,1])
    #
    #     plt.subplot(122)
    #     plt.title("style")
    #     for stroke in person_clusters[i].get_data():
    #         plt.plot(stroke.get_feature('x'), stroke.get_feature('y'))
    #
    #     plt.savefig(f'pairs/{i}.jpg')

    points = np.array(  [[466,  434 ],
                         [466,  434 ],
                         [467,  434 ],
                         [468,  434 ],
                         [469,  434 ],
                         [470,  433 ],
                         [472,  433 ],
                         [473,  432 ],
                         [475,  432 ],
                         [476,  432 ],
                         [478,  431 ],
                         [474,  432 ],
                         [471,  433 ],
                         [467,  434 ],
                         [464,  435 ],
                         [460,  437 ],
                         [457,  438 ],
                         [453,  439 ],
                         [450,  440 ],
                         [450,  440 ]])

    points2 = np.array([[[460,   431 ],
                          [461,  432 ],
                          [462,  430 ],
                          [464,  432 ],
                          [463,  431 ],
                          [464,  430 ],
                          [464,  431 ],
                          [464,  430 ],
                          [464,  430 ],
                          [465,  430 ],
                          [465,  430 ],
                          [465,  430 ],
                          [465,  431 ],
                          [466,  430 ],
                          [467,  431 ],
                          [469,  432 ],
                          [470,  434 ],
                          [471,  434 ],
                          [472,  435 ],
                          [473,  436 ]],

                         [[461,  431 ],
                          [462,  431 ],
                          [464,  430 ],
                          [464,  430 ],
                          [464,  430 ],
                          [464,  430 ],
                          [465,  431 ],
                          [465,  429 ],
                          [466,  429 ],
                          [466,  429 ],
                          [466,  429 ],
                          [465,  429 ],
                          [466,  430 ],
                          [466,  430 ],
                          [467,  432 ],
                          [468,  433 ],
                          [468,  435 ],
                          [468,  435 ],
                          [469,  437 ],
                          [470,  438 ]],

                         [[462,  433 ],
                          [463,  433 ],
                          [464,  433 ],
                          [465,  433 ],
                          [465,  432 ],
                          [465,  432 ],
                          [465,  432 ],
                          [466,  432 ],
                          [466,  432 ],
                          [467,  430 ],
                          [466,  429 ],
                          [466,  430 ],
                          [467,  429 ],
                          [468,  430 ],
                          [468,  431 ],
                          [469,  431 ],
                          [469,  433 ],
                          [470,  433 ],
                          [471,  435 ],
                          [471,  436 ]],

                         [[460,  430 ],
                          [461,  431 ],
                          [462,  431 ],
                          [463,  432 ],
                          [463,  430 ],
                          [463,  430 ],
                          [464,  430 ],
                          [464,  430 ],
                          [464,  429 ],
                          [465,  431 ],
                          [465,  429 ],
                          [465,  429 ],
                          [465,  430 ],
                          [465,  430 ],
                          [465,  431 ],
                          [467,  433 ],
                          [468,  434 ],
                          [469,  435 ],
                          [470,  436 ],
                          [471,  437 ]],

                         [[461,  430 ],
                          [460,  430 ],
                          [461,  429 ],
                          [463,  431 ],
                          [464,  430 ],
                          [463,  430 ],
                          [465,  430 ],
                          [465,  429 ],
                          [464,  429 ],
                          [465,  430 ],
                          [465,  429 ],
                          [465,  429 ],
                          [465,  429 ],
                          [465,  430 ],
                          [467,  431 ],
                          [469,  432 ],
                          [469,  434 ],
                          [471,  435 ],
                          [472,  435 ],
                          [472,  437 ]]])

    # simplify_clusters_shape = pickle.load(open('x/x.p', "rb"))
    # person_clusters_shape = pickle.load(open('y/y.p', "rb"))
    #
    # for sim in simplify_clusters_shape:
    #     sim[:, 0] -= sim[:, 0].mean()
    #     sim[:, 1] -= sim[:, 1].mean()
    #
    # for per in person_clusters_shape:
    #     per[:, :, 0] -= per[:, :, 0].mean()
    #     per[:, :, 1] -= per[:, :, 1].mean()
    #
    # pickle.dump(simplify_clusters_shape, open('x/x_norm.p', "wb"))
    # pickle.dump(person_clusters_shape, open('y/y_norm.p', "wb"))

    # import matplotlib.pyplot as plt
    # plt.subplot(121)
    # plt.title("simplify")
    # plt.plot(simplify_clusters_shape[0][:,0], simplify_clusters_shape[0][:,1], 'o')
    #
    # plt.subplot(122)
    # plt.title("output")
    # for i in range(5):
    #     plt.plot(person_clusters_shape[0][i][:, 0], person_clusters_shape[0][i][:, 1], 'o')
    # plt.show()


    #
    # a = np.array([[[1,2], [3,6], [4,8], [11,16]], [[1,2], [3,6], [4,8], [11,16]]])
    # b = np.array([[[1,2], [3,6], [4,8], [12,17]], [[1,2], [3,6], [4,9], [12,17]]])
    # print(np.sum((np.linalg.norm(np.array(a) - np.array(b), axis=2))))
    # print(l2_stroke(a,b))
    #
    # print(np.sum((np.linalg.norm(np.array(a[0]) - np.array(b[0]), axis=1))) + np.sum((np.linalg.norm(np.array(a[1]) - np.array(b[1]), axis=1))))