# -*- coding: UTF-8 -*-
"""
:Script:   Field_Calculator_defs.py
:Author:   Dan.Patterson@carleton.ca
:Created:  2013-??-??
:Modified: 2016-09-17
:Purpose: demonstration functions that can be used in the field calculator
:Useage:
:  copy the function into the code block and put in the suggested format
:  into the expression box...modify that to reflect your field names and/or
:  required input values
:Tips:
:  some people use the __esri_field_calculator_splitter__ within their scripts
:  to separate the def from the expression box.
:
"""
import sys
import arcpy

# Simple no code block examples
"""
- simple comparison and reclass
  " True condition " if !some_field! == "some_value" else " False condition "
  "Agricultural 10 Acres" if !Zone! == "A-10" else "False"

- compound comparisons and reclass
  "True" if !some_field! < "some_value"
         else ("other value" if !some_field! == "final_value" else "False")
  "True" if !Random! < 5 else ("sort of" if !Random! == 5 else "False")

- geometry related
  5000m along polygons and polylines
  x = !Shape!.boundary().positionAlongLine(5000,False).centroid.X  # polygons
  y = !Shape!.boundary().positionAlongLine(5000,False).centroid.Y  # polygons

  x = !Shape!.positionAlongLine(5000,False).centroid.X  #for polylines
  y = !Shape!.positionAlongLine(5000,False).centroid.Y  #for polylines

  x = !Shape!.positionAlongLine(0.25,True).centroid.X  # polylines
  y = !Shape!.positionAlongLine(0.25,True).centroid.Y  # polylines
  pnt_num = !Shape!.pointCount - !Shape!.partCount
          # number of vertices in a polygon

"""
# Imports  normally they are handled within the def to remind people

print("\nScript location...since I always forget... {}".format(sys.argv[0]))

# ----- Code block section -----

# (1) no_nulls_allowed([!a!, !b!, !c!...])

def no_nulls_allowed(fld_list):
    """provide a list of fields"""
    good_stuff = []
    for i in fld_list:
        if i:
            good_stuff.append(str(i))
        out_str = " ".join(good_stuff)
    return out_str


def no_nulls_mini(fld_list):
    ok_flds = [str(i) for i in fld_list if i]
    return ("{} "*len(ok_flds)).format(*ok_flds)


# (2) Conversions

"""field name, extracts numbers from the beginning of a string until
a non-numeric entry is reached, otherwise None
extract_int(field_name)   extract the integer from the front of a string
"""

import itertools
def extract_int(field_name):
    try:
        return int("".join(itertools.takewhile(str.isdigit, str(field_name))))
    except:
        pass


"""-----------------------------------------
field name, extracts integers from a string
extract_nums(field_name)   extracts all numbers from a string
"""
def extract_nums(field_name):
    numbers = []
    for i in field_name:
        try:
            val = int(i)
            numbers.append(str(val))
        except ValueError:
            break
    return int("".join(numbers))


"""-----------------------------------------
field name, strips numbers from a string
strip_num(field_name)   strips all numbers from a string
"""
####add string data digits, letters, printable punctuation etc
import string
def strip_num(field_name):
    for i in string.digits:
        field_name = field_name.replace(i,"")
    return field_name


# (3) Comparisons

"""-----------------------------------------
field name, threshold value in the code block, modify expression to suite,
if_elif_else(!test_fld!,5)    modify the code within the return section
"""
def if_elif_else(field_name, value):
    if field_name <= value:
        return value     # ie (1.0 + (field_name/2.0))
    elif field_name <= 10:
        return value     # ie (1.5 + 2*((field-1)/24.0))
    else:
        return value     # ie (max(5.0, 3.5+((field-25)/15.0)))


"""-----------------------------------------
field name, threshold value in the code block, modify expression to suite,
returns 0, 1
greaterThan( !Y_UTM!, 5000000)  copy to the expression box and modify to suit
"""
def greaterThan(field_name,value):
    if field_name >= value:
        return True
    else:
        return False


"""-----------------------------------------
field name, threshold value in the code block, modify expression to suite,
returns 0, 1
lessThan( !Y_UTM!, 500000)  copy to the expression box and modify to suit
"""
def lessThan(field_name,value):
    if field_name <= value:
        return True
    else:  return False

#Others
"""return a text value for nulls in a text field
replaceNull(!some_field!,"'TextValue')   """
def replaceNull(field_name,value):
    if field_name is None:
        return value
    else:
        return field_name


"""-----------------------------------------
return a text value with a word removed by index number
remove_part(!textFld!,0," ")
"""
def remove_part(field_name,index=-1,sep=" "):
    a_list = field_name.split(sep)
    out_str = ""
    if abs(index) <= len(a_list):
        a_list.pop(index)
        for i in a_list:
            out_str += (i + " ")
    return out_str.rstrip()


"""------------------------------------------
Label grid cells like excel
"""
import string
c = -1
r = 0

def code_grid(rows=1, cols=1):
    global c, r
    c += 1
    UC = list(string.ascii_uppercase)
    if c >= cols:
        c = 0
        r += 1
    label = UC[c] + str(r)
    return label


# (4) Date-time

# arcpy.time.ParseDateTimeString(!FIELD1!) + datetime.timedelta(hours=1)
# arcpy.time.ParseDateTimeString(!FIELD1!) + datetime.timedelta(days=1)

import datetime

#datetime.datetime.now() + datetime.timedelta(days=1)

#Math related

""" -----------------------------------------
Return a random number in the range 0-1
randomNum()  #enter into the expression box
"""
import numpy
def randomNum():
    return numpy.random.random()


"""-----------------------------------------
Return a random number integer in the range start, end
randomInt(0,10)   #enter into the expression box
"""
import numpy
def randomInt(start, end):
    return numpy.random.randint(start, end)


"""-----------------------------------------
Return the cumulative sum of a field
cumulative(field_name)  # enter into the expression box
"""
old = 0   # include this line
def cumulative(new):
    '''accumulate values'''
    global old
    if old >= 0:
        old = old + new
    else:
        old = new
    return old

#  or

total = 0
def cumsum(in_field):
    global total
    total += in_field
    return total

"""-----------------------------------------
geometric mean   not done yet
"""
import operator
def geometric_mean(iterable):
    return (reduce(operator.mul, iterable)) ** (1.0/len(iterable))

#Geometry related

#--Counts
"""-----------------------------------------
Input shape field, return number of parts
count_parts(!Shape!)    #enter into the expression box
"""
def count_parts(shape):
    return shape.partCount


"""-----------------------------------------
Input shape field, return number of points in a feature
count_pnts(!Shape!)     #enter into the expression box
"""
def count_pnts(shape):
    counter = 0
    num_parts = shape.partCount
    num_pnts = 0
    while counter < num_parts:
        part = shape.getPart(counter)
        pnt = part.next()
        while pnt:
            num_pnts += 1
            pnt = part.next()
            if not pnt:
                pnt = part.next()
        counter += 1
    return num_pnts


# -- Point features ----------------------------------------
#  References
#     https://geonet.esri.com/message/557195#557195
#     https://geonet.esri.com/message/557196#557196

"""-----------------------------------------
dist_to(shape, from_x, from_y)
input:      shape field, origin x,y
returns:    distance to the specified point
expression: dist_to(!Shape!, x, y)
"""

def dist_to(shape, from_x, from_y):
    x = shape.centroid.X
    y = shape.centroid.Y
    distance = math.sqrt((x - from_x)**2 + (y - from_y)**2)
    return distance


""" -----------------------------------------
dist_between(shape)
input:      shape field
returns:    distance between successive points
expression: dist_between(!Shape!)
"""

x0 = 0.0
y0 = 0.0

def dist_between(shape):
    global x0
    global y0
    x = shape.centroid.X
    y = shape.centroid.Y
    if x0 == 0.0 and y0 == 0.0:
        x0 = x
        y0 = y
    distance = math.sqrt((x - x0)**2 + (y - y0)**2)
    x0 = x
    y0 = y
    return distance


"""-----------------------------------------
dist_cumu(shape)
input:      shape field
returns:    cumulative distance between points
expression: dist_cumu(!Shape!)
"""

x0 = 0.0
y0 = 0.0
distance = 0.0
def dist_cumu(shape):
    global x0
    global y0
    global distance
    x = shape.firstpoint.X
    y = shape.firstpoint.Y
    if x0 == 0.0 and y0 == 0.0:
        x0 = x
        y0 = y
    distance += math.sqrt((x - x0)**2 + (y - y0)**2)
    x0 = x
    y0 = y
    return distance

"""-----------------------------------------
azimuth_to(shape, from_x, from_y)
input:      shape field, from_x, from_y
returns:    angle between 0 and <360 between a specified point and others
expression: azimuth_to(!Shape!, from_x, from_y)
"""
def azimuth_to(shape, from_x, from_y):
    x = shape.centroid.X
    y = shape.centroid.Y
    radian = math.atan((x - from_x)/(y - from_y))
    degrees = math.degrees(radian)
    if degrees < 0:
        return degrees + 360.0
    else:
        return degrees

"""-----------------------------------------
angle_between(shape)
input:      shape field
returns:    angle between successive points,
            NE +ve 0 to 90, NW +ve 90 to 180,
            SE -ve <0 to -90, SW -ve <-90 to -180
expression: angle_between(!Shape!)
"""
x0 = 0.0;  y0 = 0.0;  angle = 0.0
def angle_between(shape):
    global x0
    global y0
    x = shape.centroid.X
    y = shape.centroid.Y
    if x0 == 0.0 and y0 == 0.0:
        x0 = x
        y0 = y
        return 0.0
    radian = math.atan2((y - y0),(x - x0))
    angle = math.degrees(radian)
    x0 = x
    y0 = y
    return angle

#--Polyline features

"""-----------------------------------------
Input shape field: simple shape length
poly_length(!Shape!)    #enter into the expression box
"""
def poly_length(shape):
    return shape.length

"""-----------------------------------------
Input shape field: returns cumulative length of polylines connected or not
poly_cumu_len(!Shape!)    #enter into the expression box
"""
length = 0.0
def poly_cumu_len(shape):
    global length
    length += shape.length
    return length

"""-----------------------------------------
Input shape field: returns angle between 0 and <360 based upon the first and last point
azimuth_to(!Shape!,from_x, from_y)  # ie azimuth_to(!Shape!,339200, 5025200)
"""
import math
def azimuth_to(shape, from_x, from_y):
    x = shape.centroid.X
    y = shape.centroid.Y
    radian = math.atan2((y - from_y), (x - from_x))
    degrees = math.degrees(radian)
    if degrees < 0:
        return  degrees + 360
    return degrees


"""-----------------------------------------
Input shape field, value (distance or decimal fraction, 0-1),
    use_fraction (True/False), XorY (X or Y):
Returns a point x meters or x  decimal fraction along a line,
   user specifies whether X or Y coordinates
pnt_along(!Shape!, 100, False, 'X')  # eg. X coordinate 100 m from start point
"""
def pnt_along(shape, value, use_fraction, XorY):
    XorY = XorY.upper()
    if use_fraction and (value > 1.0):
        value = value/100.0
    if shape.type == "polygon":
        shape = shape.boundary()
    pnt = shape.positionAlongLine(value, use_fraction)
    if XorY == 'X':
        return pnt.centroid.X
    else:
        return pnt.centroid.Y

# --Polygon


# --Shapes in general
##def shape_shift(shape,dX=0,dY=0):
##    """ shape_shift(!Shape!,0,0)
##    __esri_field_calculator_splitter__
##    shift/move/translate a shape by dX,dY
##    """
##    pnt = shape.firstPoint.X #.centroid
##    #pnt.X += pnt.X + dX
##    #pnt.Y = pnt.Y + dY
##    return pnt
import arcpy

def shift_features(in_features, x_shift=None, y_shift=None):
    """
    Shifts features by an x and/or y value. The shift values are in
    the units of the in_features coordinate system.

    Parameters:
    in_features: string
        An existing feature class or feature layer.  If using a
        feature layer with a selection, only the selected features
        will be modified.

    x_shift: float
        The distance the x coordinates will be shifted.

    y_shift: float
        The distance the y coordinates will be shifted.
    """

    with arcpy.da.UpdateCursor(in_features, ['SHAPE@XY']) as cursor:
        for row in cursor:
            cursor.updateRow([[row[0][0] + (x_shift or 0),
                               row[0][1] + (y_shift or 0)]])

    return


# shape_shift(!SHAPE@!,0,0)

# formatting
"""partition a string or number into parts in a text field
!field_name!   in the expression box"""
def partition(field_name):
    a = str(field_name)
    return "{0[0]}+{0[1]}".format(a.partition(a[3:]))


#field_name=602300
#print frmt(field_name)
