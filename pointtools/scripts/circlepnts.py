# -*- coding: UTF-8 -*-
"""
:Script:   createcompass.py
:Author   Dan_Patterson@carleton.ca
:Modified: 2017-04-02
: if north angle is needed, you can use this to convert
:if fromNorth:
:    ang = np.mod((450.0 - ang), 360.)
"""

# --------------------------------------------------------------------------
import sys
import numpy as np
import arcpy

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}

np.set_printoptions(edgeitems=10, linewidth=80, precision=2,
                    suppress=True, threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


def _circle(radius=10, theta=22.5, xc=0.0, yc=0.0):
    """Produce a circle/ellipse depending on parameters.
    :  radius - distance from centre
    :  theta - either a single value to form angles about a circle or
    :        - a list or tuple of the desired angles
    """
    angles = np.deg2rad(np.arange(180.0, -180.0-theta, step=-theta))
    x_s = radius*np.cos(angles) + xc    # X values
    y_s = radius*np.sin(angles) + yc    # Y values
    pnts = np.c_[x_s, y_s]
    return pnts

# --------------------------------------------------------------------------
inFC = sys.argv[1]  # r"C:\Data\points\points.gdb\fishnet_label"
outFC = sys.argv[2]  # r"C:\Data\points\pnts2.shp"
radius = float(sys.argv[3])  # radius = 2
theta = float(sys.argv[4])
a = arcpy.da.FeatureClassToNumPyArray(inFC, ["SHAPE@X", "SHAPE@Y"])

frmt = """... {} ...
:Input featureclass : {}
:Output featureclass: {}
:Radius {}, angle step {}
:Points:
{}
:
"""
args = [script, inFC, outFC, radius, theta, a]
msg = frmt.format(*args)
arcpy.AddMessage(msg)
# ---- option (1) read X and Y separately, then reset the dtype names
# ---- get the circle values, stack and set dtype
# or a list like... theta = [0, 90, 180, 270]
#    theta = [0, 90, 180, 270]

a.dtype = [('X', '<f8'), ('Y', '<f8')]
b = [_circle(radius, theta, a['X'][i], a['Y'][i])
     for i in range(len(a))]
c = np.vstack(b)
c.dtype = a.dtype
c = np.squeeze(c)

arcpy.da.NumPyArrayToFeatureClass(c, outFC, c.dtype.names)
arcpy.AddXY_management(outFC)

# --------------------------------------------------------------------
if __name__ == '__main__':
    """produce some points around a centroid at angles and distances"""
# ---- option (2) read the centroid coordinates then reset the dtype name
# ---- get the circle values, stack and set dtype
#    a = arcpy.da.FeatureClassToNumPyArray(inFC, "SHAPE@XY")
#    a.dtype = [('XY', '<f8', (2,))]
#    b = [_circle(radius, theta, a['XY'][i,0], a['XY'][i,1])
#        for i in range(len(a))]
