"""
input:
pts: points representing  the curve cluster (curves divided into points)
dist. typical distance between points ( more like an upper bound but not really)
output:
g:  simplified curves (usually only one but sometimes can be more if the cluster is problematic)
"""
'''
#find the nearest point from a given point to a large list of points

import numpy as np

def distance(pt_1, pt_2):
    pt_1 = np.array((pt_1[0], pt_1[1]))
    pt_2 = np.array((pt_2[0], pt_2[1]))
    return np.linalg.norm(pt_1-pt_2)

def closest_node(node, nodes):
    pt = []
    dist = 9999999
    for n in nodes:
        if distance(node, n) <= dist:
            dist = distance(node, n)
            pt = n
    return pt

a = []
for x in range(50000):
    a.append((np.random.randint(0,1000),np.random.randint(0,1000)))

some_pt = (1, 2)

closest_node(some_pt, a)
'''
import rhinoscriptsyntax as rs
import Rhino

start = pt = pts.pop(0) #arbitrary point
b = []  # list of control points of the current curve we work on
c = []  # list of lists

while ((len(pts) > 0)):  # while we didn't go over all input points

    # I added a numpy version of this at the top
    result = rs.PointCloudClosestPoints(pts, [pt], dist)  # returns the points closest to pt at result[0]
    if (result and result[0]):
        # get all the closest points (actual coordinates)
        rmv = []
        pt = Rhino.Geometry.Point3d(0, 0, 0)
        lenP = len(result[0])
        for i in range(lenP):
            pt += pts[result[0][i]]
            rmv.append(pts[result[0][i]])
        # calc the mean of the points we found
        pt.X /= lenP
        pt.Y /= lenP

        b.append(pt)
        # remove all the closest points
        for i in range(lenP):
            pts.remove(rmv[i])

        # if we are done append the end point
        if (len(pts) == 0):
            b.append(pt)
            c.append(b)  # append the last curve as well


    else:
        if (len(b) == 0):
            print("len(b)==0):")
            break
        bSartpoint = b[0]
        result = rs.PointCloudClosestPoints(pts, [bSartpoint], 2 * dist)
        if (result and result[0]):
            # print("neighbors found now reverse the b and continue")
            b.reverse()
            pt = pts[result[0][0]]
            continue
        else:
            # print("no close points found so save b and continue from wherever")
            c.append(b)
            pt = pts.pop(0)
            b = []
            continue

"""
at this point in c you have a list of points for each simplified curve
c might hold more than just one curve because sometimes the cluster is simplified to a few curves not just one

"""

g = []

# generate curves out of the points we saved
#the following code is very rhino specific it is basically fitting a 3rd degree curve to the points
#3rd degree means it's a cubic function not linear or square
#we should probably replace it maybe something like ths:
# https://nurbs-python.readthedocs.io/en/latest/fitting.html
for i in range(len(c)):

    if (len(c[i]) > 4):
        knots = [0, 0]
        knots.extend(range(len(c[i]) - 2))
        knots.append(len(c[i]) - 3)
        knots.append(len(c[i]) - 3)
        g.append(rs.AddNurbsCurve(c[i], knots, 3))
    else:
        print("(len(c[i])<=4")

"""sharan 

when we remove all the neighboring points we get some kind of 
smoothing but we also get a method that is sensitive to the chosen distance
of search. 
it might be good to do this itteratively and compute a loss function 
and optimize the search distance . 
say start from a large distance and as long as there is improvement reduce it  



"""

