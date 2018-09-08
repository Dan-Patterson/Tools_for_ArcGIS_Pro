# -*- coding: UTF-8 -*-
"""
split_by_area
===========

Script :   split_by_area.py

Author :   Dan_Patterson@carleton.ca

Modified:  2018-08-27

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
from arcpytools_plt import (tweet, fc_info, _poly_ext,
                            trans_rot, cal_area, get_polys)
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

def _cut_poly(poly, p_id, step=1.0, split_axis="X", split_fac=4, SR=None):
    """Perform the poly* cutting and return the result.

    step : number
        fractional step for division, 1.0 equates to 1%
    split_face : number
        number of areas to produce, 4, means split into 4 equal areas
    """
    L, B, R, T = _poly_ext(poly)
#    s_fac = math.ceil((R - L)/step)
#    lefts = np.linspace(L+dx, R, num=s_fac, endpoint=True)
    dx = step
    dy = step
    if split_axis == "X":
        lefts = np.arange(L+dx, R+dx, dx, dtype='float')
        splitters = np.array([[[l, B-1.0], [l, T+1.0]] for l in lefts])
    elif s_axis == 'Y':
        tops = np.arange(B+dy, T+dy, dy, dtype='float')
        splitters = np.array([[[R+1.0, t], [L-1.0, t]] for t in tops])
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


def final_cut(cutters, poly):
    """ final cut
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
    return cuts  # , cutters

# ---- demo and tool section -------------------------------------------------
#

if len(sys.argv) == 1:
    testing = False
    in_pth = script.split("/")[:-2] + ["Polygon_lineTools.gdb"]
    in_fc = "/".join(in_pth) + "/shapes_mtm9"
    out_fc = "/".join(in_pth) + "/c0"
    s_axis = "Y"
    s_fac = 4
else:
    testing = False
    in_fc = sys.argv[1]
    out_fc = sys.argv[2]
    s_fac = int(sys.argv[3])
    s_axis = sys.argv[4]

# ---- for both
#
shp_fld, oid_fld, shp_type, SR = fc_info(in_fc)
out_polys, out_ids = get_polys(in_fc)
#old_ids = np.repeat(out_ids, s_fac)  # produce data for the output id field

# ---- instant bail
if SR.type == 'Projected':
    result_ = []
    for i in range(len(out_polys)):
        poly = out_polys[i]
        p_id = out_ids[i]
        cuts, cutters = _cut_poly(poly, p_id, step=1,
                                  split_axis = s_axis,
                                  split_fac=4, SR=SR)
        idxs = cal_area(poly, cuts, cutters, s_fac)
        f_cutters = [cutters[i] for i in idxs]
        r = final_cut(f_cutters, poly)
        result_.extend(r)
    if not testing:
        if arcpy.Exists(out_fc):
            arcpy.Delete_management(out_fc)
        arcpy.CopyFeatures_management(result_, out_fc)
        out_ids = np.repeat(out_ids, s_fac)
        id_fld = np.zeros((len(result_),),
                          dtype=[("key", "<i4"), ("Old_ID", "<i4")])
        id_fld["key"] = np.arange(1, len(result_) + 1)
        id_fld["Old_ID"] = out_ids
        arcpy.da.ExtendTable(out_fc, oid_fld, id_fld, "key")
else:
    msg = """
    -----------------------------------------------------------------
    Input data is not in a projected coordinate system....
    bailing...
    -----------------------------------------------------------------
    """
    tweet(msg)

# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
