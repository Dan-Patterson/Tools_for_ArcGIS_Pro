# -*- coding: UTF-8 -*-
"""
:Script:   spiral.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2018-02-28
:Purpose:  tools for working with numpy arrays
:Useage:
:
:References:
:
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
import numpy as np
import arcpy


ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=5, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


root5 = 2.23606797749978967
golden = (1.0 + np.sqrt(5.0))/2  # 1.6180339887498948482..


# ---- spiral shape generation -----------------------------------------------
#
# ---- arcpy related ----
#
def extent_scale(ext_fc, scale_by=1.0):
    """Scale up/down the extent defined by an extent featureclass by a
    : factor (1 = 100%.  The scaling is done about the center point.
    """
    fc_ext = arcpy.Describe(ext_fc).extent
    ext_w = fc_ext.width
    ext_h = fc_ext.height
    buff_dist = min(ext_w, ext_h) * scale_by
    ext_poly = fc_ext.polygon
    ext_buff = ext_poly.buffer(buff_dist)
    new_ext = ext_buff.extent
    cent = ext_poly.centroid
    return new_ext, cent


def output_polylines(output_shp, SR, pnts):
    """Produce the output polygon shapefile"""
    msg = '\nRead the script header... A projected coordinate system required'
    assert (SR is not None), msg  # and (SR.type == 'Projected'), msg
    polylines = []
    for pair in pnts:                 # create the polygon geometry
        pl = arcpy.Polyline(arcpy.Array([arcpy.Point(*xy) for xy in pair]), SR)
        polylines.append(pl)
    if arcpy.Exists(output_shp):     # overwrite any existing versions
        arcpy.Delete_management(output_shp)
    arcpy.CopyFeatures_management(polylines, output_shp)
    return output_shp


def shapes(sides=5, radius=1.0):
    """coordinates of a pentagram with the y-axis as the bisector and the base
    :  on the x-axis
    :  clockwise - a, b, c, d, e
    The vertices will have coordinates (x+rsinθ,y+rcosθ)(x+rsin⁡θ,y+rcos⁡θ)
    , where θ is an integer multiple of 2π/n or 360/n if you prefer degrees
    to radians.)
    """
    rad_360 = np.radians(360.).astype(float)
    step = rad_360/sides
    st_end = np.arange(0.0, rad_360+step, step)
    x = np.sin(st_end) * radius
    y = np.cos(st_end) * radius
    pnts = np.c_[x, y]
    return pnts


# ---- spiral examples ------------------------------------------------------
#
def spiral_archim(N, n, clockwise=True, reverse=False):
    """Create an Archimedes spiral in the range 0 to N points with 'n' steps
    : between each incrementstep.  You could use np.linspace
    :Notes: When n is small relative to N, then you begin to form rectangular
    :  spirals, like rotated rectangles
    :Tried:  N = 1000, n = 30
    """
    rnge = np.arange(0.0, N+1.0)
    if clockwise:
        rnge = rnge[::-1]
    phi = rnge/n * np.pi
    xs = phi * np.cos(phi)
    ys = phi * np.sin(phi)
    if reverse:
        tmp = np.copy(xs)
        xs = ys
        ys = tmp
    xy = np.c_[xs, ys]
    wdth, hght = np.ptp(xy, axis=0)
    return xs, ys, xy


def spiral_sqr(ULx=-10, n_max=100):
    """Create a square spiral from the centre in a clockwise direction
    : ULx = upper left x coordinate, relative to center (0, 0)
    : n-max = maximum number of iterations should ULx not be reached
    :- see spirangle, Ulam spiral
    """
    def W(x, y, c):
        x -= c[0]
        return x, y, c

    def S(x, y, c):
        y -= c[1]
        return x, y, c

    def E(x, y, c):
        x += c[2]
        return x, y, c

    def N(x, y, c):
        y += c[3]
        return x, y, c

    c = np.array([1, 1, 2, 2])
    pos = [0, 0, c]
    n = 0
    v = [pos]
    cont = True
    while cont:
        p0 = W(*v[-1])
        p1 = S(*p0)
        p2 = E(*p1)
        p3 = N(*p2)
        c = c + 2
        p3 = [p3[0], p3[1], c]
        for i in [p0, p1, p2, p3]:
            v.append(i)
        # --- print(p0, p0[0])  # for testing
        if (p0[0] <= ULx):      # bail option 1
            cont = False
        if n > n_max:           # bail option 2
            cont = False
        n = n+1
    coords = np.asarray([np.array([i[0], i[1]]) for i in v])[:-3]
    return coords


# -------Excellent one-------------------------------------------------------
#  https://stackoverflow.com/questions/36834505/
#        creating-a-spiral-array-in-python
def spiral_cw(A):
    A = np.array(A)
    out = []
    while(A.size):
        out.append(A[0])        # take first row
        A = A[1:].T[::-1]       # cut off first row and rotate counterclockwise
    return np.concatenate(out)


def spiral_ccw(A):
    A = np.array(A)
    out = []
    while(A.size):
        out.append(A[0][::-1])    # first row reversed
        A = A[1:][::-1].T         # cut off first row and rotate clockwise
    return np.concatenate(out)


def base_spiral(nrow, ncol):
    return spiral_ccw(np.arange(nrow*ncol).reshape(nrow, ncol))[::-1]


def to_spiral(A):
    A = np.array(A)
    B = np.empty_like(A)
    B.flat[base_spiral(*A.shape)] = A.flat
    return B


def from_spiral(A):
    A = np.array(A)
    return A.flat[base_spiral(*A.shape)].reshape(A.shape)
# ---- end code section--------------------------------------


def _demo():
    """ demo to create an archimedes spiral ----
    """
    # ---- (1) basic parameters
    scale_by = 1.10  # scale output extent so it is slightly bigger than needed
    pnts_cnt = 1000  # points for spiral
    pnts_div = 40.0  # divisions between points
    pth = r"C:\GIS\Geometry_projects\Spiral_sort\Polygons\spiral_sort.gdb"
    ext_poly = r"\extent_line"
    out_sp = r"\spiral"
    #
    ext_fc = pth + ext_poly
    out_fc = pth + out_sp
    #
    desc = arcpy.da.Describe(ext_fc)  # get infor from extent poly
    SR = desc['spatialReference']
    #
    # ---- (2) create the spiral
    xs, ys, xy = spiral_archim(pnts_cnt, pnts_div)  # (1) make a spiral
    w, h = np.ptp(xy, axis=0)
    #
    ext, cent = extent_scale(ext_fc, scale_by=scale_by)  # (2) get extent info
    x_c = cent.X  # cent is an arcpy point object
    y_c = cent.Y
    x_fac = ext.width/float(w)
    y_fac = ext.height/float(h)
    #
    # ---- (3) put it all together and create the featureclass
    a = np.array(xy)
    a[:, 0] = a[:, 0] * x_fac + x_c
    a[:, 1] = a[:, 1] * y_fac + y_c
    pnts = a.tolist()
    output_polylines(out_fc, SR, [pnts])
    #
    return a


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
#    _demo()
#    pth = r"C:\Git_Dan\a_Data\arcpytools_demo.gdb"
#    in_fc = pth + r"\r_extent"