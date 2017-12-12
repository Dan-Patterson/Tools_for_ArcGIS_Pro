# -*- coding: UTF-8 -*-
"""
:Script:   sampling_grid.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-12-07
:Purpose:  tools for working with numpy arrays
:Useage:
:
:References:
:----------
:Phish_Nyet.py
:  https://community.esri.com/blogs/dan_patterson/2016/09/09/
:        numpy-snippets-3-phishnyet-creating-sampling-grids-using-numpy
:n-gons....
:  https://community.esri.com/blogs/dan_patterson/2016/09/09/
:        n-gons-regular-polygonal-shape-generation
:
:Purpose:
:-------
: - Produce a sampling grid with user defined parameters.
: - create hexagon shapes in two forms, flat-topped and pointy-topped
:
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
import numpy as np
import arcpy

arcpy.overwriteOutputs = True

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}

np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


# ----------------------------------------------------------------------
# ---- main functions ----
def code_grid(cols=1, rows=1, zero_based=False, shaped=True, bottom_up=False):
    """produce spreadsheet like labelling, either zero or 1 based
    :  zero - A0,A1  or ones - A1, A2..
    :  dig = list('0123456789')  # string.digits
    : import string .... string.ascii_uppercase
    """
    UC = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    z = [1, 0][zero_based]
    rc = [1, 0][zero_based]
    c = [UC[c] + str(r)                # pull in the column heading
         for r in range(z, rows + rc)  # label in the row letter
         for c in range(cols)]         # label in the row number
    c = np.asarray(c)
    if shaped:
        c = c.reshape(rows, cols)
        if bottom_up:
            c = np.flipud(c)
    return c


def rotate(pnts, angle=0):
    """rotate points about the origin in degrees, (+ve for clockwise) """
    angle = np.deg2rad(angle)                 # convert to radians
    s = np.sin(angle)
    c = np.cos(angle)    # rotation terms
    aff_matrix = np.array([[c, s], [-s, c]])  # rotation matrix
    XY_r = np.dot(pnts, aff_matrix)           # numpy magic to rotate pnts
    return XY_r


def rectangle(dx=1, dy=1, cols=1, rows=1):
    """Create the array of pnts to pass on to arcpy using numpy magic
    :  dx - increment in x direction, +ve moves west to east, left/right
    :  dy - increment in y direction, -ve moves north to south, top/bottom
    """
    X = [0.0, 0.0, dx, dx, 0.0]       # X, Y values for a unit square
    Y = [0.0, dy, dy, 0.0, 0.0]
    seed = np.array(list(zip(X, Y)))  # [dx0, dy0] keep for insets
    a = [seed + [j * dx, i * dy]       # make the shapes
         for i in range(0, rows)   # cycle through the rows
         for j in range(0, cols)]  # cycle through the columns
    a = np.asarray(a)
    return a


def hex_flat(dx=1, dy=1, cols=1, rows=1):
    """generate the points for the flat-headed hexagon
    :dy_dx - the radius width, remember this when setting hex spacing
    :  dx - increment in x direction, +ve moves west to east, left/right
    :  dy - increment in y direction, -ve moves north to south, top/bottom
    """
    f_rad = np.deg2rad([180., 120., 60., 0., -60., -120., -180.])
    X = np.cos(f_rad) * dy
    Y = np.sin(f_rad) * dy            # scaled hexagon about 0, 0
    seed = np.array(list(zip(X, Y)))  # array of coordinates
    dx = dx * 1.5
    dy = dy * np.sqrt(3.)/2.0
    hexs = [seed + [dx * i, dy * (i % 2)] for i in range(0, cols)]
    m = len(hexs)
    for j in range(1, rows):  # create the other rows
        hexs += [hexs[h] + [0, dy * 2 * j] for h in range(m)]
    return hexs


def hex_pointy(dx=1, dy=1, cols=1, rows=1):
    """pointy hex angles, convert to sin, cos, zip and send
    :dy_dx - the radius width, remember this when setting hex spacing
    :  dx - increment in x direction, +ve moves west to east, left/right
    :  dy - increment in y direction, -ve moves north to south, top/bottom
    """
    p_rad = np.deg2rad([150., 90, 30., -30., -90., -150., 150.])
    X = np.cos(p_rad) * dx
    Y = np.sin(p_rad) * dy      # scaled hexagon about 0, 0
    seed = np.array(list(zip(X, Y)))
    dx = dx * np.sqrt(3.)/2.0
    dy = dy * 1.5
    hexs = [seed + [dx * i * 2, 0] for i in range(0, cols)]
    m = len(hexs)
    for j in range(1, rows):  # create the other rows
        hexs += [hexs[h] + [dx * (j % 2), dy * j] for h in range(m)]
    return hexs


def repeat(seed=None, corner=[0, 0], cols=1, rows=1, angle=0):
    """Create the array of pnts to pass on to arcpy using numpy magic to
    :  produce a fishnet of the desired in_shp.
    :seed - use grid_array, hex_flat or hex_pointy.  You specify the width
    :       and height or its ratio when making the shapes
    :corner - lower left corner of the shape pattern
    :dx, dy - offset of the shapes... this is different
    :rows, cols - the number of rows and columns to produce
    :angle - rotation angle in degrees
    """
    if seed is None:
        a = rectangle(dx=1, dy=1, cols=3, rows=3)
    else:
        a = np.asarray(seed)
    if angle != 0:
        a = [rotate(p, angle) for p in a]      # rotate the scaled points
    pnts = [p + corner for p in a]            # translate them
    return pnts


def output_polygons(output_shp, SR, pnts):
    """produce the output polygon shapefile"""
    msg = '\nRead the script header... A projected coordinate system required'
    assert (SR is not None), msg  # and (SR.type == 'Projected'), msg
    polygons = []
    for pnt in pnts:                 # create the polygon geometry
        pl = arcpy.Polygon(arcpy.Array([arcpy.Point(*xy) for xy in pnt]), SR)
        polygons.append(pl)
    if arcpy.Exists(output_shp):     # overwrite any existing versions
        arcpy.Delete_management(output_shp)
    arcpy.CopyFeatures_management(polygons, output_shp)


msg = """
: --------------------------------------------------------------------
: output {}
: SR  .. {}
: type . {}
: corner .. {}
: size..... {} (dx, dy)
: cols/rows {}
: sample seed
{}
: --------------------------------------------------------------------
"""

def _demo(seed=None, out_fc=False, SR=None, corner=[0, 0], angle=0):
    """Generate the grid using the specified or default parameters
    """
    corner = corner  # [300000.0, 5000000.0]
    dx, dy = [1, 1]
    cols, rows = [3, 3]
    if seed is None:
        seed = rectangle(dx=1, dy=1, cols=3, rows=3)
#        seed = hex_pointy(dx=10, dy=10, cols=3, rows=3)
#        seed = hex_flat(dx=10, dy=10, cols=3, rows=3)
        seed_t = 'rectangle'
    if SR is None:
        SR = 3857  # -- WGS84 Web Mercator (Auxiliary Sphere)
    pnts = repeat(seed=seed, corner=corner, cols=3, rows=3, angle=0)
    args = ["", SR, seed_t, corner, [dx, dy], [cols, rows], seed[0]]
    print(msg.format(*args))
    return pnts


def _tool():
    """run when script is from a tool"""
    out_fc = sys.argv[1]  #
    SR = sys.argv[2]
    seed_t = sys.argv[3]
    corn_x = float(sys.argv[4])
    corn_y = float(sys.argv[5])
    dx = float(sys.argv[6])
    dy = float(sys.argv[7])
    cols = int(sys.argv[8])
    rows = int(sys.argv[9])
    #
    angle = float(sys.argv[10])
    corner = [corn_x, corn_y]
    if seed_t == 'rectangle':
        seed = rectangle(dx, dy, cols, rows)
    elif seed_t == 'hex_pointy':
        seed = hex_pointy(dx, dy, cols, rows)
    elif seed_t == 'hex_flat':
        seed = hex_flat(dx, dy, cols, rows)
    else:
        seed = rectangle(dx, dy, cols, rows)
    # ----
    msg = """
    : --------------------------------------------------------------------
    : output {}
    : SR  .. {}
    : type . {}
    : corner .. {}
    : size..... {} (dx, dy)
    : cols/rows {}
    : sample seed
    {}
    """
    args = [out_fc, SR, seed_t, corner, [dx, dy], [cols, rows], seed[0]]
    arcpy.AddMessage(msg.format(*args))
    arcpy.GetMessages()
    pnts = repeat(seed=seed, corner=corner, cols=cols, rows=rows, angle=angle)
    return out_fc, SR, pnts


# ----------------------------------------------------------------------
# __main__ .... code section
if len(sys.argv) == 1:
    testing = True
    pnts = _demo()
else:
    testing = False
    out_fc, SR, pnts = _tool()
#
if not testing:
    output_polygons(out_fc, SR, pnts)
    print('\nSampling grid was created... {}'.format(out_fc))

#
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
