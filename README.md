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
6) math
7) imageio
8) skimage
9) PIL
10) scipy

directories:
data: includes all the data files (.txt) and the drawings (.png) of the participants.
ref_pics: includes all the reference pictures.
ref_pics_crop: includes all the cropped reference pictures (with only the active pixels).
sketch_simplification: includes files for implements sketch simplification.
articles: includes relevants artical for our project.
concat: includes results of the Pix2Pix network (input, label, prediction) as one picture.
dataset: includes results of the Pix2Pix network (input, label, prediction) separte.
participant_output_data: includes participant draws and graphs.

Tal: 
clustered_draws: ?



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
Tal:
add your changes

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
plot_crop_image(): plot the picture with only the active pixels area

###################
### Participant ###
###################
Represent a participant. The constructor get the name of the participant (for example: 'aliza') and it holding a list
of Drawing of all the drawing of the participant.

Useful functions:
plot_participant_pictures(): plot all the drawings of the participant (by using plot_picture() of Drawing class).
export_data():  export all participant draws and graphs into "participants_output_data" folder. 
	        calculate general means of features and write it to <participant>.txt in the same folder.
		

################
### Analyzer ###
################
This class including static methods for analyzing the data.


########################
### nearest_neighbor ###
########################
This file include the function find_nearest_neighbor(), that getting array of 2D points (p1), and list of others arrays points 2D (neighbors),
and find the closest array, according to our distance metric.
Add pictures to illustration


########################
### simplify_cluster ###
########################
This file using to simplify clusters. 
Cluster is define to be a group of Strokes, so this file getting a cluster, and make it more "simple".
Add pictures to illustration


############
### main ###
############

Example 1 (Using Drawing object to plot graph): \n
input_path = "data/D_01/aliza/aliza__130319_0935_D_01.txt"  # path to data file \n
draw = Analyzer.create_drawing(input_path)  # creating Drawing object
draw.speed_vs_time(pause=True)  # plot speed vs time, with pauses.
draw.length_vs_time()  # plot length vs time, without pauses.
draw.pressure_vs_time()  # plot pressure vs time, without pauses.
draw.plot_picture()  # plot reference picture with the actual drawing (problem with resolution)

Example 2 (Using Participant object to plot all participant data):
person = Participant("aliza")  # creating Participant object, of aliza
person.plot_participant_pictures()  # plot all the picture of the participant
