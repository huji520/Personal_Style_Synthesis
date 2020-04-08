from Analyzer import Analyzer
from Drawing import Drawing
import simplify_cluster
import os
import numpy as np
import pickle
import nearest_neighbor
import matplotlib.pyplot as plt


class Participant:
    def __init__(self, name):
        self._name = name
        self._data = self.get_all_files_of_participant()
        self.clusters = []

    def get_data(self):
        return self._data[0]

    def get_picture_list(self):
        return self._data[1]

    def get_name(self):
        return self._name

    def get_all_files_of_participant(self):
        """
        :return: list of Drawing of the participant
        """
        lst = []
        pic_list = []
        path = None
        for picture in os.listdir("data"):
            if "DS_Store" not in picture:
                for person in os.listdir("data/" + picture):
                    if "DS_Store" not in person:
                        if person == self._name:
                            for file in os.listdir("data/" + picture + "/" + person):
                                if "DS_Store" not in file:
                                    if file.endswith(".txt"):
                                        path = "data/" + picture + "/" + person + "/" + file
                                    if file.endswith(".png"):
                                        pic_list.append("data/" + picture + "/" + person + "/" + file)
                            if path is not None:
                                drawing = Analyzer.create_drawing(path)
                                if drawing is not None:
                                    lst.append(drawing)
                                else:
                                    print("Error: missing data")
                                    print(path)
                            else:
                                print("Error: missing data")

        return lst, pic_list

    def plot_participant_pictures(self):
        """
        plot all the participant pictures
        """
        for i, drawing in enumerate(self._data[0]):
            # plt.subplot(2,2,i + 1)
            drawing.plot_picture()
            plt.savefig(f"aliza_figs/{i}.png")

    def export_data(self):
        """
        export all participant draws and graphs into "participants_output_data"
        folder. calculate general means of features and write it to
        <participant>.txt in the same folder
        """
        os.mkdir("participants_output_data/" + self._name)
        text_file = open("participants_output_data/" + self._name + "/" +
        self._name + ".txt", "w")

        speed_total_mean = 0
        speed_total_std = 0
        length_total = 0
        length_total_std = 0
        pressure_total_mean = 0
        pressure_total_std = 0
        speed_counter = 0
        length_counter = 0
        pressure_counter = 0
        for draw in self._data[0]:
            img, img_ref = draw.plot_crop_image(True)
            path = "participants_output_data/" + self._name + "/" + \
                   draw.get_ref_path().split("/")[1]
            os.mkdir(path)
            img.save(path + "/" + draw.get_ref_path().split("/")[1], "PNG")
            img_ref.save(path + "/ref_" + draw.get_ref_path().split("/")[1]
                         , "PNG")
            img.close()
            img_ref.close()
            mean, std = draw.speed_vs_time(save=True,
                                           path=path+"/speed_vs_time")
            if mean is not None:
                speed_total_mean = speed_total_mean + mean
                speed_total_std = speed_total_std + std
                speed_counter = speed_counter + 1

            mean, std = draw.length_vs_time(save=True,
                                           path=path+"/length_vs_time")
            if mean is not None:
                length_total = length_total + mean
                length_total_std = length_total_std + std
                length_counter = length_counter + 1

            mean, std = draw.pressure_vs_time(save=True,
                                            path=path + "/pressure_vs_time")
            if mean is not None:
                pressure_total_mean = pressure_total_mean + mean
                pressure_total_std = pressure_total_std + std
                pressure_counter = pressure_counter + 1

        text_file.write("speed_mean = " + str(
            speed_total_mean / speed_counter) +
                        "\t\t" + "speed_std_mean = " +
                        str(speed_total_std / speed_counter) + "\n\n")

        text_file.write("length_mean = " + str(length_total /
                        length_counter) + "\t\t" + "length_std_mean = " +
                        str(length_total_std / length_counter) + "\n\n")

        text_file.write("pressure_mean = " + str(pressure_total_mean /
                        pressure_counter) + "\t\t" + "pressure_std_mean = " +
                        str(pressure_total_std / pressure_counter) + "\n\n")

        text_file.close()

    def split_all_participant_picture(self, patch_w, patch_h):
        process_counter = 1
        img_counter = 1
        for pic_path in self.get_picture_list():
            img_counter = Analyzer.split_image_to_patches(pic_path, patch_w, patch_h, img_counter)
            print(process_counter, " out of ", len(self.get_picture_list()))
            process_counter += 1

    def simplify_all_clusters(self, euc_dist_threshold, dist_threshold, ang_threshold):
        """
        Simplify all the clusters of the participant (which include all the draws)
        :param euc_dist_threshold: argument for group_stroke
        :param dist_threshold: argument for group_stroke
        :param ang_threshold: argument for group_stroke
        :return: Simplify clusters
        """
        print("Start clustering all participant draws")
        clusters = []
        for j, draw in enumerate(self.get_data()):
            print(f"{j} out of {len(self.get_data())}")
            clusters.extend(draw.group_strokes(euc_dist_threshold, dist_threshold, ang_threshold)[1])
        self.clusters = clusters
        print("End clustering all participant draws\n")

        print("Start simplify all participant clusters")
        simplify_clusters = []
        for i, draw in enumerate(clusters):
            print(f"{i} out of {len(clusters)}")
            x = []
            y = []
            for stroke in draw.get_data():
                x.extend(stroke.get_feature('x'))
                y.extend(stroke.get_feature('y'))

            p = simplify_cluster.simplify_cluster(x, y, i, dist=10, save_pairs=False)
            if len(p) > 3:  # handle with very short simplify
                simplify_clusters.append(p)
            else:
                simplify_clusters.append([[0,0], [5000,5000]])
        print("End simplify all participant clusters\n")

        return simplify_clusters

    def create_dict(self, euc_dist_threshold, dist_threshold, ang_threshold):
        """
        Creating a new dict for the participant.
        :param euc_dist_threshold: argument for simplify_all_clusters
        :param dist_threshold: argument for simplify_all_clusters
        :param ang_threshold: argument for simplify_all_clusters
        :return: Array of simplify and array of clusters, which are fit to each other (by indexes)
        """
        print(f"Start creating new dict for {self._name}\neuc_dist_threshold={euc_dist_threshold}\n"
              f"dist_threshold={dist_threshold}\nang_threshold={ang_threshold}\n")

        base_path = f"{self._name}_{euc_dist_threshold}_{dist_threshold}_{ang_threshold}.p"
        simplify_path = os.path.join("pickle", "simplify", base_path)
        person_clusters_path = os.path.join("pickle", "clusters", base_path)
        simplify_clusters = self.simplify_all_clusters(euc_dist_threshold, dist_threshold, ang_threshold)
        person_clusters = self.clusters
        pickle.dump(simplify_clusters, open(simplify_path, "wb"))
        pickle.dump(person_clusters, open(person_clusters_path, "wb"))
        print("End creating new dict")
        return simplify_clusters, person_clusters

    def searching_match_on_person(self, p1, load_dict, euc_dist_threshold, dist_threshold, ang_threshold):
        """
        Find the best match for 2D array p1 in the participant dict
        :param p1: 2D array
        :param load_dict: Use pickle if True, else create a new one
        :param euc_dist_threshold: argument for create a new dict if needed
        :param dist_threshold: argument for create a new dict if needed
        :param ang_threshold: argument for create a new dict if needed
        :return: The cluster that found (with the participant style)
        """
        base_path = f"{self._name}_{euc_dist_threshold}_{dist_threshold}_{ang_threshold}.p"
        simplify_path = os.path.join("pickle", "simplify", base_path)
        person_clusters_path = os.path.join("pickle", "clusters", base_path)
        if load_dict:
            simplify_clusters = pickle.load(open(simplify_path, "rb"))
            person_clusters = pickle.load(open(person_clusters_path, "rb"))
        else:
            simplify_clusters, person_clusters = self.create_dict(euc_dist_threshold, dist_threshold, ang_threshold)

        i, x_shift, y_shift, error = nearest_neighbor.find_nearest_neighbor(p1, simplify_clusters)
        return person_clusters[i], x_shift, y_shift, error
