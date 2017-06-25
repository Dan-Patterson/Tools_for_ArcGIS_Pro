# -*- coding: UTF-8 -*-
"""-----------------------------------------
:Input shape field:
:returns angle between 0 and <360 from the first and last point
:azimuth_to(!Shape!,from_x, from_y, from_north=True)
ie azimuth_to(!Shape!, 300050, 5000050, True)
"""
import math

def azimuth_to(shape, from_x, from_y, from_north):
    x = shape.centroid.X
    y = shape.centroid.Y
    radian = math.atan2((y - from_y), (x - from_x))
    angle = math.degrees(radian)
    if from_north:
        angle = (450 - angle) % 360
    return angle
#__esri_field_calculator_splitter__
#azimuth_to(!Shape!, 300050, 5000050, True)