from Analyzer import Analyzer
from Drawing import Drawing
import simplify_cluster
from Participant import Participant
import numpy as np
import pickle
import os


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


if __name__ == "__main__":

    input_banana = "example_input/testdata banana.txt"
    input_fish = "example_input/testdata fish.txt"
    input1 = "data/D_01/aliza/aliza__130319_0935_D_01.txt"

    # draw = Analyzer.create_drawing(input_banana, orig_data=False)
    # new_draw = transfer_style(draw, "aliza", load_person=False, load_dict=False, already_simplified=True, stroke_length=20, simplify_size=20)
    # new_draw.plot_picture(show_clusters=False)

    # base_path = f"aliza_10_5_0.5_stroke_length_20.p"
    # simplify_path = os.path.join("pickle", "simplify", base_path)
    # person_clusters_path = os.path.join("pickle", "clusters", base_path)
    # simplify_clusters = pickle.load(open(simplify_path, "rb"))
    # person_clusters = pickle.load(open(person_clusters_path, "rb"))
    #
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







