# -*- coding: utf-8 -*-
""" input shape field: returns cumulative distance between points
dist_cumu(!Shape!)    #enter into the expression box"""
import math
x0 = 0.0;  y0 = 0.0;  distance = 0.0
def dist_cumu(shape):
    global x0;  global y0;  global distance
    x = shape.firstpoint.X;  y = shape.firstpoint.Y
    if x0 == 0.0 and y0 == 0.0:
        x0 = x; y0 = y
    distance += math.sqrt((x - x0)**2 + (y - y0)**2)
    x0 = x;  y0 = y
    return distance