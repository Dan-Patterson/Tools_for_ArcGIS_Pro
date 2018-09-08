# -*- coding: UTF-8 -*-
"""
split_polys
===========

Script :   split_polys.py

Author :   Dan_Patterson@carleton.ca

Modified:  2018-09-07

Purpose :  tools for working with numpy arrays

Notes:
-----
The xs and ys form pairs with the first and last points being identical
The pairs are constructed using n-1 to ensure that you don't form a
line from identical points.

First split polygon is a sample of a multipart.  Added 0, 0 and 0, 80
back in

>>> xs = [0., 0., 80., 0, 0., 100., 100., 0.]
>>> ys = [0., 30., 30., 80., 100., 100., 0., 0.]
>>> a = np.array(list(zip(xs, ys))) * 1.0  # --- must be floats
>>> v = np.array([[50., 0], [50, 100.]])
>>> ext = np.array([[0., 0], [0, 100.],[100, 100.], [100., 0.], [0., 0.]])
return a, v

References:
----------

`<https://stackoverflow.com/questions/3252194/numpy-and-line-intersections>`_.
`<https://community.esri.com/message/627051?commentID=627051#comment-627051>`
`<https://community.esri.com/message/779043-re-how-to-divide-irregular-
polygon-into-equal-areas-using-arcgis-105?commentID=779043#comment-779043>`

This is a good one
`<https://tereshenkov.wordpress.com/2017/09/10/dividing-a-polygon-into-a-given
-number-of-equal-areas-with-arcpy/>`

---------------------------------------------------------------------
"""
# ---- imports, formats, constants ----
import sys
import math
from textwrap import dedent
import numpy as np
import warnings
from arcpytools_plt import tweet, fc_info, _poly_ext, trans_rot, cal_area
import arcpy

warnings.simplefilter('ignore', FutureWarning)

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=5, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


# ---- Do the work or run the demo ------------------------------------------
#

frmt = """
Input features.... {}
Output features... {}
Number of splits . {}
Split types ...... {}
"""


def _cut_poly(poly, p_id, step=1.0, split_fac=4, SR=None):
    """Perform the poly* cutting and return the result.

    step : number
        fractional step for division, 1.0 equates to 1%
    split_face : number
        number of areas to produce, 4, means split into 4 equal areas
    """
    L, B, R, T = _poly_ext(poly)
    dx = 100.0/step
    s_fac = math.ceil((R - L)/dx)
    lefts = np.linspace(L+dx, R, num=s_fac, endpoint=True)
    splitters = np.array([[[l, B-1.0], [l, T+1.0]] for l in lefts])
#    n = []
#    for i in splitters:
#        n.append(trans_rot(i, 45))
#    splitters = np.array(n)
    cutters = []
    for s in splitters:
        s = s.tolist()
        c = arcpy.Polyline(arcpy.Array([arcpy.Point(*xy) for xy in s]), SR)
        cutters.append(c)
    # ----
    cuts = []
    for i in cutters:
        rght = poly
        if i.crosses(poly):
            try:
                left, rght = poly.cut(i)
                if rght is None:
                    cuts.append(left)
                cuts.append(left)
                poly = rght
                rght = left
            except RuntimeError:
                tweet("Issues with poly...{}".format(p_id))
                continue
        else:
            cuts.append(rght)
    return cuts, cutters


def get_polys(in_fc):
    """Return polygons from a polygon featureclass
    """
    out_polys = []
    out_ids = []
    with arcpy.da.SearchCursor(in_fc,  ["SHAPE@", "OID@"]) as cursor:
        for row in cursor:
            out_polys.append(row[0])
            out_ids.append(row[1])
    return out_polys, out_ids


def do_work(in_fc, out_fc, s_type, s_fac, s_axis):
    """Do the actual work for either the demo or the tool

    Requires:
    --------
    in_fc : feautureclass
        polygon or polyline featureclass
    out_fc : featureclass
        same as input
    s_type : choice of `extent` or `areas`
        `extent` uses the bounds with `s_fac` to subdivide the range into
        sub-polygons

    **extent option**

    x_tent : extent parameter
        LBRT L(eft), B(ottom), R(ight), T(op) in the featureclass units
    s_fac : number
        used to represent the divisions, for example, a factor of 2 is the
        same as half (1/2) or 50%

    **width option**

    s_fac : number
        represents the distance for the cut spacing for the x or y directions

    **area option**

    s_perc : double
        Percentage of the total area representing sub-area size. 25% would
        result in 4 sub-areas with approximately 25% of the total area.
    """
#    def __poly_ext__(p):
#        """poly* extent
#        """
#        L, B = p.extent.lowerLeft.X, p.extent.lowerLeft.Y
#        R, T = p.extent.upperRight.X, p.extent.upperRight.Y
#        return L, B, R, T

    def __cutter__(p, s_type, s_fac, s_axis):
        """Produce the cutters for the shape
        fac = np.linspace(L, R, num=divisor+1, endpoint=True)
        steps, incr = np.linspace(ax_min, ax_max, num=pieces+1,
                              endpoint=True, retstep=True)
        """
        L, B, R, T = _poly_ext(p)
        if s_type == 'extent':
            dx = (R - L)/s_fac
            dy = (T - B)/s_fac
        elif s_type == 'distance':  # just use the s_fact
            dx = s_fac
            dy = s_fac
            if s_axis == 'X':
                s_fac = math.ceil((R - L)/s_fac)
            else:
                s_fac = math.ceil((T - B)/s_fac)
        elif s_type == 'area':
            dx = s_fac
            dy = s_fac
        else:
            dx = 2
            dy = 2
        if s_axis == 'X':
            lefts = np.linspace(L+dx, R, num=s_fac, endpoint=True)
            splitters = np.array([[[l, B-1.0], [l, T+1.0]] for l in lefts])
        elif s_axis == 'Y':
            tops = np.linspace(B+dy, T, num=s_fac, endpoint=True)
            splitters = np.array([[[R+1.0, t], [L-1.0, t]] for t in tops])
        cutters = []
        for s in splitters:
            s = s.tolist()
            c = arcpy.Polyline(arcpy.Array([arcpy.Point(*xy) for xy in s]), SR)
            cutters.append(c)
        return cutters
    #
    def __get_cutters__(poly, s_fac, s_axis):
        """Get the poly* cutters
        """
        cutters = __cutter__(poly, s_fac, s_axis)
        return cutters
    #
    def __cut__(poly, p_id, cutters):
        """Perform the poly* cutting and return the result.
        """
        cuts = []
        for i in cutters:
            rght = poly
            if i.crosses(poly):
                try:
                    left, rght = poly.cut(i)
                    if rght is None:
                        cuts.append(left)
                    cuts.append(left)
                    poly = rght
                    rght = left
                except RuntimeError:
                    tweet("Issues with poly...{}".format(p_id))
                    continue
            else:
                cuts.append(rght)
        return cuts
    # ---- Do the work -------------------------------------------------------
    #
    s_fac = int(s_fac)
    #
    # (1) get the polys and the cutters
    out_polys = []
    out_ids = []
    with arcpy.da.SearchCursor(in_fc,  ["SHAPE@", "OID@"]) as cursor:
        for row in cursor:
            poly = row[0]
            p_id = row[1]
            cutters = __cutter__(poly, s_type, s_fac, s_axis)
            cuts = __cut__(poly, p_id, cutters)
            out_polys.extend(cuts)
            out_ids.append(p_id)
    #
    for p in out_polys:
        if not (p.area > 0.):
            out_polys.remove(p)
    return out_polys, out_ids


# ---- demo and tool section -------------------------------------------------
#
"""

"""

if len(sys.argv) == 1:
    testing = True
    in_pth = script.split("/")[:-2] + ["Polygon_lineTools.gdb"]
    in_fc = "/".join(in_pth) + "/shapes_mtm9"
    out_fc = "/".join(in_pth) + "/c0"
    s_type = 'extent'
    s_fac = 4
    s_axis = 'X'
    shp_fld, oid_fld, shp_type, SR = fc_info(in_fc)
    out_polys, out_ids = get_polys(in_fc)

else:
    in_fc = sys.argv[1]
    out_fc = sys.argv[2]
    s_type = sys.argv[3]
    # --- extent parameters
    s_fac = int(sys.argv[4])
    s_axis = sys.argv[5]
    shp_fld, oid_fld, shp_type, SR = fc_info(in_fc)
    out_polys, out_ids = do_work(in_fc, out_fc, s_type, s_fac, s_axis)
    for p in out_polys:
        if not (p.area > 0.):
            out_polys.remove(p)
    if arcpy.Exists(out_fc):
        arcpy.Delete_management(out_fc)
    arcpy.CopyFeatures_management(out_polys, out_fc)
    out_ids = np.repeat(out_ids, s_fac)
    id_fld = np.zeros((len(out_polys),),
                      dtype=[("key", "<i4"), ("Old_ID", "<i4")])
    id_fld["key"] = np.arange(1, len(out_polys) + 1)
    id_fld["Old_ID"] = out_ids
    arcpy.da.ExtendTable(out_fc, oid_fld, id_fld, "key")


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
