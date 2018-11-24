# -*- coding: UTF-8 -*-
"""
:Script:   azimuth_sequential_pnts.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-06-19
:Purpose:  tools for working with numpy arrays
:Useage:
:
:References:
:
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
x0 = 0.0
y0 = 0.0
angle = 0.0
def angle_between(shape, from_north):
    """Calculate the angle/azimuth between sequential points in a point file.
    :Use:
    : .... angle_between(!Shape!, True) ....
    """
    global x0
    global y0
    x = shape.centroid.X
    y = shape.centroid.Y
    if x0 == 0.0 and y0 == 0.0:
        x0 = x
        y0 = y
        return 0.0
    radian = math.atan2((y - y0), (x - x0))
    angle = math.degrees(radian)
    if from_north:
        angle = (450 - angle) % 360
    x0 = x
    y0 = y
    return angle
# __esri_field_calculator_splitter__  # optionally
# angle_between(!Shape!, True)
#
# Python command... expr is the code block above
# arcpy.management.CalculateField("poly_pnts", "Seq_angle",
#                                 "angle_between(!Shape!, True)", "PYTHON_9.3",
#                                 expr)
# Angle_between(!Shape!)
# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
#    _demo()

from random import uniform
def shift(val, start=-1, end=1):
    """shift within the range - start and end"""
    jiggle = uniform(start, end)
    return val + jiggle

a = 10
print(shift(10, -1, 1))