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
Stroke is holding information of the time, (x,y) location and pressure. Every one of them represent as np.array (each
cell represent a different measurement in the current stroke.

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


