from Analyzer import Analyzer
from Participant import Participant


if __name__ == "__main__":
    input1 = "data/D_01/aliza/aliza__130319_0935_D_01.txt"
    # input2 = "data/D_01/zoey/zoey__130319_1208_D_01.txt"
    input3 = "data/F_05/aliza/aliza__040619_1842_F_05.txt"
    draw = Analyzer.create_drawing(input3)
    # draw.speed_vs_time(pause=True)
    # draw.length_vs_time()
    # draw.pressure_vs_time()
    draw.plot_picture()
    # person = Participant("aliza")
    # person.plot_participant_pictures()