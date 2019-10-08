from Analyzer import Analyzer
import os


class Participant:

    def __init__(self, name):
        self._name = name
        self._data = self.get_all_files_of_participant()

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

