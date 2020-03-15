from Analyzer import Analyzer
from Drawing import Drawing
import simplify_cluster
from Participant import Participant
import numpy as np
import pickle
import os



# def searching_match_on_person(person_name, p1, load=True, euc_dist_threshold=10, dist_threshold=5, ang_threshold=0.5):
#     base_path = "{0}_{1}_{2}_{3}.p".format(person_name, euc_dist_threshold, dist_threshold, ang_threshold)
#     simplify_path = os.path.join("pickle", "simplify", base_path)
#     style_path = os.path.join("pickle", "style", base_path)
#     person_clusters_path = os.path.join("pickle", "clusters", base_path)
#     if load:
#         simplify_clusters = pickle.load(open(simplify_path, "rb"))
#         style_clusters = pickle.load(open(style_path, "rb"))
#         person_clusters = pickle.load((open(person_clusters_path, "rb")))
#     else:
#         person = Participant(person_name)
#         simplify_clusters, style_clusters = person.simplify_all_clusters(euc_dist_threshold, dist_threshold, ang_threshold)
#         person_clusters = person.clusters
#         pickle.dump(simplify_clusters, open(simplify_path, "wb"))
#         pickle.dump(style_clusters, open(style_path, "wb"))
#         pickle.dump(person_clusters, open(person_clusters_path, "wb"))
#
#     p, i, x_shift, y_shift = nearest_neighbor.find_nearest_neighbor(p1, simplify_clusters)
#     # Analyzer.plot_clusters(orig=p1, simplify_cluster=p, style_cluster=style_clusters[i], scatter=True)
#     # person_clusters[i].shift_x(x_shift)
#     # person_clusters[i].shift_y(y_shift)
#     return simplify_clusters[i], person_clusters[i], x_shift, y_shift



def plot_simplify(input_txt_path, euc_dist_threshold=10, dist_threshold=5, ang_threshold=0.5):
    draw = Analyzer.create_drawing(input_txt_path)
    draw.plot_simplify_drawing(euc_dist_threshold, dist_threshold, ang_threshold)


# def get_simplified_draw(clusters):
#     simplified_clusters = []
#
#     for i, draw in enumerate(clusters):
#         print("{0} out of {1}".format(i, len(clusters)))
#         x = []
#         y = []
#         for stroke in draw.get_data():
#             x.extend(stroke.get_feature('x'))
#             y.extend(stroke.get_feature('y'))
#
#         p = simplify_cluster.simplify_cluster(x, y, i, dist=10, save_pairs=False)
#         if (len(p) == 0):
#             simplified_clusters.append([[0,0]])
#             continue
#         simplified_clusters.append(p)
#
#     return simplified_clusters
#
#
# def replace_cluster_with_cluster(draw):
#     clusters = draw.group_strokes(euc_dist_threshold=10, dist_threshold=5, ang_threshold=0.5)[1]
#
#
#     simplified_clusters = get_simplified_draw(clusters)
#     for i in range(len(clusters)):
#         print("{0} out of {1}".format(i, len(clusters)))
#         neighbor_simplified, clusters[i], x_shift, y_shift = searching_match_on_person("aliza",
#                                                                                    np.array(simplified_clusters[i]),
#                                                                     load=True)
#
#         clusters[i].shift_x(x_shift)
#         clusters[i].shift_y(y_shift)
#
#         # if i == 8:
#         #     clusters[i].plot_picture()
#
#     strokes = []
#     for cluster in clusters:
#         strokes.extend(cluster.get_data())
#     rebuilt_draw = Drawing(strokes, clusters[0].get_ref_path(), clusters[0].get_pic_path())
#     # pickle.dump(rebuilt_draw, open("rebuilt_draw.p", "wb"))
#     # pickle.dump(clusters, open("clusters.p", "wb"))
#     rebuilt_draw.plot_picture()


def get_simplified_draw(clusters, already_simplified=False):
    simplified_clusters = []

    for i, draw in enumerate(clusters):
        print("{0} out of {1}".format(i, len(clusters)))
        x = []
        y = []
        for stroke in draw.get_data():
            x.extend(stroke.get_feature('x'))
            y.extend(stroke.get_feature('y'))
        if already_simplified:
            p = np.stack((x,y), axis=1)
        else:
            p = simplify_cluster.simplify_cluster(x, y, i, dist=10, save_pairs=False)
        if len(p) == 0:
            simplified_clusters.append(np.stack((x,y), axis=1))
            continue
        simplified_clusters.append(p)

    return simplified_clusters


def transfer_style(draw, person_name, load_person=False, load_dict=True, already_simplified=False):
    """
    generate a draw with the style of the given person name
    :param draw: a draw of some participant
    :param person_name: the name of the person with the wanted style
    :return: a draw object with the wanted style
    """
    pickle_path = os.path.join("pickle", "participant")
    if load_person:
        participant = pickle.load(open(os.path.join(pickle_path, "{0}.p".format(person_name)), "rb"))
    else:
        participant = Participant(person_name)
        pickle.dump(participant, open(os.path.join(pickle_path, "{0}.p".format(person_name)), "wb"))

    clusters = draw.group_strokes(euc_dist_threshold=10, dist_threshold=5, ang_threshold=0.51)[1]
    simplified_clusters = get_simplified_draw(clusters, already_simplified=already_simplified)

    if not load_dict:
        participant.create_dict(ang_threshold=0.5)

    for i in range(len(clusters)):
        print("{0} out of {1}".format(i, len(clusters)))
        matched_cluster, x_shift, y_shift, match = participant.searching_match_on_person(
                                                            np.array(simplified_clusters[i]), ang_threshold=0.51)
        if match:
            clusters[i] = matched_cluster
            clusters[i].shift_x(x_shift)
            clusters[i].shift_y(y_shift)
        # @TODO: else return a simplified draw object


    strokes = []
    for cluster in clusters:
        strokes.extend(cluster.get_data())

    return Drawing(strokes, clusters[0].get_ref_path(), clusters[0].get_pic_path())








if __name__ == "__main__":
    input2 = "data/D_01/zoey/zoey__130319_1208_D_01.txt"
    input3 = "data/F_05/aliza/aliza__040619_1842_F_05.txt"
    input1 = "data/D_01/aliza/aliza__130319_0935_D_01.txt"
    input4 = "data/F_01/zoey/zoey__300519_1846_F_01.txt"
    input5 = "data/F_01/danielle/danielle__190319_2056_F_01.txt"
    input6 = "data/F_01/dudi/dudi__100419_1314_F_01.txt"
    input7 = "data/F_01/elchanan/elchanan__240319_1814_F_01.txt"
    input8 = "data/D_01/danielle/danielle__190319_2046_D_01.txt"
    input9 = "data/D_01/dudi/dudi__140519_1225_D_01.txt"
    input10 = "data/D_01/elchanan/elchanan__030419_1910_D_01.txt"

    input_banana = "example_input/testdata banana.txt"
    input_fish = "example_input/testdata fish.txt"

    inputs = [input4, input5, input6, input7, input8, input9, input10]
    new_draws = []

    # for input in inputs:
    #     draw = Analyzer.create_drawing(input)
    #     new_draws.append(transfer_style(draw, "aliza", load_person=True, load_dict=True))
    #
    # for new_draw in new_draws:
    #     new_draw.plot_picture(show_clusters=True)

    draw = Analyzer.create_drawing(input_fish, orig_data=False)
    # draw.plot_picture()
    # participant = Participant("aliza")
    # participant.create_dict(ang_threshold=0.52)
    new_draw = transfer_style(draw, "aliza", load_person=True, load_dict=True, already_simplified=True)
    new_draw.plot_picture(show_clusters=False)


    # clusters = draw.group_strokes(euc_dist_threshold=50, dist_threshold=5, ang_threshold=0.51)[1]
    # strokes = []
    # for cluster in clusters:
    #     strokes.extend(cluster.get_data())
    # rebuilt_draw = Drawing(strokes, clusters[0].get_ref_path(), clusters[0].get_pic_path())
    #
    # rebuilt_draw.plot_picture(show_clusters=True)

    # draw.plot_simplify_drawing()


    # participant = Participant("aliza")
    # participant = pickle.load(open('aliza.p', 'rb'))
    # participant.plot_participant_pictures()
