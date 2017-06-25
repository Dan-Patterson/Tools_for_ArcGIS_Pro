# -*- coding: UTF-8 -*-
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
#__esri_field_calculator_splitter__
#angle_between(!Shape!, True)
