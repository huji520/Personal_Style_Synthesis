# Personal_Style_Synthesis

###############
### imports ###
###############
the following packages are required:
1) numpy
2) pandas
3) os
4) copy
5) matplotlib

The project is build in classes, so the files are using other files in the project, so we need to import them.
The import line should be: "from FileName import ClassName" or "from Personal_Style_Synthesis.FileName import ClassName",
it's depends if you just open the git project after cloning, or creating new project in your local pc. If you wish only
to import file and not a class (like Constants) you can just wrote "import FileName".


##############
### Stroke ###
##############
This is the atomic unit it our data.
Stroke is holding information of the time, (x,y) location and pressure of the stroke. Every one of them represent as
np.array (each cell represent a different sampling in the current stroke).

Useful functions:
get_feature(feature_name): return the np.array that represent the given feature name.
average(feature_array): return the average of the given feature array.
length(): return the total geometric length of the stroke.
time(): return the total time of the stroke.
is_pause(): return true iff the stroke is "Pause stroke"

'Pause stroke':
There is two type of Stroke: "Normal stroke", and "Pause stroke".
"Pause stroke" is create to fill the empty space left by the Pause, so actually it's not a real data.
By default, Stroke is define as 'normal' stroke, for creating 'Pause stroke' you need to add pause=True in the
constructor call.
The 'Pause stroke' is taking the total time of the pause, and divide it into pieces, with time stamps of 0.017 seconds.
The pressure is define as 0, and all other fields (excepts 'time') is define as -1.


###############
### Drawing ###
###############
Represent a drawing. it holding list of Strokes, which together represent a complete drawing.
It also holding a path to the reference picture (what the participant was trying to draw), and a path to the actual
picture that he draw.

Useful functions:
total_length(): return list of size 2, the first value is the total geometric length of the drawing. The second value
is array of all the length of each stroke (real stroke, not pause) in the drawing.

feature_vs_time(feature, unit, pause): This if helper function for the following:
speed_vs_time(), pressure_vs_time(), length_vs_time() :
In these functions, we plot graph of the feature (speed, pressure, length) as function of time.
In speed_vs_time and pressure_vs_time, every point in the graph is represent a stroke: x value is the starting time of
the stroke, and the y value is the average of the feature, in the stroke.
In length_vs_time, the x value is the same, but y value is the total geometric length of the stroke.
You can choose if you want to see the pause-strokes in the graph, or not. The default is to plot the graph without them,
and if you want to see them, just call the functions with: pause=True in the parameters.

plot_picture(): In progress. Should plot the reference picture and the actual drawing, in the same resolution.


###################
### Participant ###
###################
Represent a participant. The constructor get the name of the participant (for example: 'aliza') and it holding a list
of Drawing of all the drawing of the participant.

Useful functions:
plot_participant_pictures(): plot all the drawings of the participant (by using plot_picture() of Drawing class).


################
### Analyzer ###
################
This class including static methods for analyzing the data.
The only method that is useful is create_drawing(path), which is kind of constructor for Drawing. It's return a new
Drawing object. Maybe it's should be in Drawing.


############
### main ###
############

Example 1 (Using Drawing object to plot graph):
input_path = "data/D_01/aliza/aliza__130319_0935_D_01.txt"  # path to data file
draw = Analyzer.create_drawing(input_path)  # creating Drawing object
draw.speed_vs_time(pause=True)  # plot speed vs time, with pauses.
draw.length_vs_time()  # plot length vs time, without pauses.
draw.pressure_vs_time()  # plot pressure vs time, without pauses.
draw.plot_picture()  # plot reference picture with the actual drawing (problem with resolution)

Example 2 (Using Participant object to plot all participant data):
person = Participant("aliza")  # creating Participant object, of aliza
person.plot_participant_pictures()  # plot all the picture of the participant