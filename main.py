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
            p = simplify_cluster.simplify_cluster(x, y, i, dist=10, save_pairs=False)

        # This is a bug in simplify cluster, p should never be zero
        if len(p) == 0:
            simplified_clusters.append(np.stack((x,y), axis=1))
            continue
        simplified_clusters.append(p)
    print("End simplify the given input\n")
    return simplified_clusters


def get_participant(person_name, load_person):
    """
    Get participant object.
    :param person_name: The name of the participant
    :param load_person: Use pickle if True, else create a new one
    :return: Participant object
    """
    pickle_person_path = os.path.join("pickle", "participant", f"{person_name}.p")
    if load_person:
        participant = pickle.load(open(pickle_person_path, "rb"))
    else:
        print("Start creating a new Participant")
        participant = Participant(person_name)
        pickle.dump(participant, open(pickle_person_path, "wb"))
        print("End creating a new Participant\n")

    return participant


def transfer_style(draw, person_name, load_person=False, load_dict=True, already_simplified=False,
                   euc_dist_threshold=10, dist_threshold=5, ang_threshold=0.5, matching_threshold=5):
    """
    Get a drawing and person name, and generate a new draw with the style of the given person name.
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
    participant = get_participant(person_name, load_person)
    clusters = draw.group_strokes(euc_dist_threshold, dist_threshold, ang_threshold)[1]
    simplified_clusters = get_simplified_draw(clusters, already_simplified=already_simplified)

    print("Start matching")
    for i in range(len(clusters)):
        print(f"{i} out of {len(clusters)}")
        matched_cluster, x_shift, y_shift, error = participant.searching_match_on_person(
            np.array(simplified_clusters[i]), load_dict, euc_dist_threshold, dist_threshold, ang_threshold)

        if error < matching_threshold:
            clusters[i] = matched_cluster
            clusters[i].shift_x(x_shift)
            clusters[i].shift_y(y_shift)
    print("End matching")

    strokes = []
    for cluster in clusters:
        strokes.extend(cluster.get_data())

    return Drawing(strokes, clusters[0].get_ref_path(), clusters[0].get_pic_path())


def plot_clusters(draw, euc_dist_threshold=10, dist_threshold=5, ang_threshold=0.5):
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
    rebuilt_draw = Drawing(strokes, clusters[0].get_ref_path(), clusters[0].get_pic_path())
    rebuilt_draw.plot_picture(show_clusters=True)


if __name__ == "__main__":
    input_banana = "example_input/testdata banana.txt"
    input_fish = "example_input/testdata fish.txt"
    input1 = "data/D_01/aliza/aliza__130319_0935_D_01.txt"

    draw = Analyzer.create_drawing(input1, orig_data=False, stroke_size=200)
    draw.plot_picture(title="output drawing (length=200)")

    draw = Analyzer.create_drawing(input1, orig_data=False)
    draw.plot_picture(title="input drawing")

    # new_draw = transfer_style(draw, "aliza", load_person=True, load_dict=True, already_simplified=True)
    # new_draw.plot_picture(show_clusters=False)
    # print(draw)

    # print(draw.get_data()[0])
    # print()
    # draw.get_data()[0].remove_and_replace(3)
    # print()
    # print(draw.get_data()[0])

    # plt.subplot(121)
    # plt.title(f"length = {draw.get_data()[0].size()}")
    # plt.plot(draw.get_data()[0].get_feature('x'), draw.get_data()[0].get_feature('y'), color='red')
    #
    # draw.get_data()[0].set_size(200)
    # print(draw.get_data()[0].size())
    # plt.subplot(122)
    # plt.title(f"length = 200")
    # plt.plot(draw.get_data()[0].get_feature('x'), draw.get_data()[0].get_feature('y'), color='blue')

    # plt.show()