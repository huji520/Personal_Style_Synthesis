from Analyzer import Analyzer
import os



class Participant:

    def __init__(self, name):
        self._name = name
        self._data = self.get_all_files_of_participant()

    def get_data(self):
        return self._data

    def get_name(self):
        return self._name

    def get_all_files_of_participant(self):
        """
        :return: list of Drawing of the participant
        """
        lst = []
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
                            if path is not None:
                                drawing = Analyzer.create_drawing(path)
                                if drawing is not None:
                                    lst.append(drawing)
                                else:
                                    print("Error: missing data")
                            else:
                                print("Error: missing data")

        return lst

    def plot_participant_pictures(self):
        """
        plot all the participant pictures
        """
        for drawing in self._data:
            drawing.plot_picture()

    def export_data(self):
        """
        export all participant draws and graphs into "participants_output_data"
        folder. calculate general means of features and write it to
        <participant>.txt in the same folder
        :return: none
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
        for draw in self._data:
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
