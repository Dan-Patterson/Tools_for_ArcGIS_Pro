# -*- coding: utf-8 -*-
def dist_cumu(shape, is_first=True):
    import arcpy
    global x0;  global y0;  global distance
    #
    arcpy.AddMessage(str(is_first))
    if is_first:
        import math
        x0 = 0.0;  y0 = 0.0;  distance = 0.0
    x = shape.centroid.X;  y = shape.centroid.Y
#    arcpy.AddMessage((str(x)+ str(y)) )
    if x0 == 0.0 and y0 == 0.0:
        x0 = x; y0 = y
    distance += math.sqrt((x - x0)**2 + (y - y0)**2)
    x0 = x;  y0 = y
    is_first = False
    arcpy.AddMessage(("{}, {} {}".format(x, y, distance) ))
    return distance, is_first