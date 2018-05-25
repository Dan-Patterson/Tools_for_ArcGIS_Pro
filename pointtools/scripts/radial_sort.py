# -*- coding: UTF-8 -*-
"""
:Script:   radial_sort.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2018-02-27
:Purpose:  tools for working with numpy arrays
:Useage:
:
:References:
:
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
import os
from textwrap import dedent
import numpy as np
import arcpy
from arcpytools import _describe, fc_info, tweet
import warnings

warnings.simplefilter('ignore', FutureWarning)

arcpy.env.overwriteOutput = True

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

# -------------------------------------------------------------------------

msg0 = """
: -----------------------------------------------------------
Script....
.... {}
No output produced because either...
 - The input and/or output path and/or filename has a space in it.
 - The output path was no good.
 - A projected coordinate system is required for the inputs
Copy and run with locally stored data or fix one or more conditions...
...
: -----------------------------------------------------------
"""

msg1 = """
: -----------------------------------------------------------
Script....
.... {}
Completed....
...
: -----------------------------------------------------------
"""


def check_files(file_path, ext=""):
    """Check expected file paths and extensions, to ensure compliance with
    :  tool specifications
    """
    is_good = True
    head, tail = os.path.split(file_path)
    if not os.path.exists(head):
        return False
    if " " in tail:
        return False
    if os.path.splitext(tail)[1] != ext:
        return False
    if " " in file_path:
        return False
    return is_good
    # ----


def extent_cent(in_fc):
    """Some basic featureclass properties
    """
    ext = arcpy.Describe(in_fc).extent
    ext_poly = ext.polygon
    cent = ext_poly.centroid
    return cent


def _xyID(in_fc, to_pnts=True):
    """Convert featureclass geometry (in_fc) to a simple 2D structured array
    :  with ID, X, Y values. Optionally convert to points, otherwise centroid.
    """
    flds = ['OID@', 'SHAPE@X', 'SHAPE@Y']
    args = [in_fc, flds, None, None, to_pnts, (None, None)]
    cur = arcpy.da.SearchCursor(*args)
    a = cur._as_narray()
    a.dtype = [('IDs', '<i4'), ('Xs', '<f8'), ('Ys', '<f8')]
    del cur
    return a


def _center(a, remove_dup=True):
    """Return the center of an array. If the array represents a polygon, then
    :  a check is made for the duplicate first and last point to remove one.
    """
    if remove_dup:
        if np.all(a[0] == a[-1]):
            a = a[:-1]
    return a.mean(axis=0)


def e_dist(a, b, metric='euclidean'):
    """Distance calculation for 1D, 2D and 3D points using einsum
    : a, b   - list, tuple, array in 1,2 or 3D form
    : metric - euclidean ('e','eu'...), sqeuclidean ('s','sq'...),
    :-----------------------------------------------------------------------
    """
    a = np.asarray(a)
    b = np.atleast_2d(b)
    a_dim = a.ndim
    b_dim = b.ndim
    if a_dim == 1:
        a = a.reshape(1, 1, a.shape[0])
    if a_dim >= 2:
        a = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
    if b_dim > 2:
        b = b.reshape(np.prod(b.shape[:-1]), b.shape[-1])
    diff = a - b
    dist_arr = np.einsum('ijk,ijk->ij', diff, diff)
    if metric[:1] == 'e':
        dist_arr = np.sqrt(dist_arr)
    dist_arr = np.squeeze(dist_arr)
    return dist_arr


def radial_sort(pnts, cent=None):
    """Sort about the point cloud center or from a given point
    : pnts - an array of points (x,y) as array or list
    : cent - list, tuple, array of the center's x,y coordinates
    :      - cent = [0, 0] or np.array([0, 0])
    :Returns: the angles in the range -180, 180 x-axis oriented
    """
    pnts = np.asarray(pnts, dtype=np.float64)
    if cent is None:
        cent = _center(pnts, remove_dup=False)
    ba = pnts - cent
    ang_ab = np.arctan2(ba[:, 1], ba[:, 0])
    ang_ab = np.degrees(ang_ab)
    sort_order = np.argsort(ang_ab)
    return ang_ab, sort_order


def output_points(out_fc, pnts):
    """Produce the output point featureclass"""
    msg = '\nRead the script header... A projected coordinate system required'
    assert (SR is not None), msg  # and (SR.type == 'Projected'), msg
    pnts_lst = []
    for pnt in pnts:                 # create the point geometry
        pnts_lst.append(arcpy.PointGeometry(arcpy.Point(*pnt), SR))
    if arcpy.Exists(out_fc):     # overwrite any existing versions
        arcpy.Delete_management(out_fc)
    arcpy.CopyFeatures_management(pnts_lst, out_fc)
    return out_fc


def output_polylines(out_fc, SR, pnts):
    """Produce the output polyline featureclass"""
    msg = '\nRead the script header... A projected coordinate system required'
    assert (SR is not None), msg  # and (SR.type == 'Projected'), msg
    polylines = []
    for pair in pnts:                 # create the polyline geometry
        pl = arcpy.Polyline(arcpy.Array([arcpy.Point(*xy) for xy in pair]), SR)
        polylines.append(pl)
    if arcpy.Exists(out_fc):     # overwrite any existing versions
        arcpy.Delete_management(out_fc)
    arcpy.CopyFeatures_management(polylines, out_fc)
    return out_fc


def test_envs(in_fc, cent, out_fc0, out_fc1):
    """ test the required parameters
    """
    # (1) ---- check input feature and for projected data
    if not arcpy.Exists(in_fc):
        tweet("\nThis file doesn't exist.\n")
        return False, []
    shp_fld, oid_fld, shp_type, SR = fc_info(in_fc)
    #
    if SR.type != 'Projected':
        tweet("\nRadial sorts only make sense for projected data.\n")
        return False, []
    if shp_type != 'Point':
        tweet("\nYou need a point file.\n")
        return False, []
    #
    # (2) ---- check the output files
    if out_fc0 not in (None, 'None', " ", "", "#"):
        is_good = check_files(out_fc0)
        if not is_good:
            tweet("\nWrong path or filename?....{}\n".format(out_fc0))
            return False, []
    if out_fc1 not in (None, 'None', " ", "", "#"):
        is_good = check_files(out_fc1)
        if not is_good:
            tweet("\nWrong path or filename?....{}\n".format(out_fc1))
            return False, []
    #
    # (3) check the center ....
    if cent in (None, 'None', " ", "", "#"):
        cent = None
    elif isinstance(cent, str):
        for i in [", ", ",", ";"]:
            cent = cent.replace(i, " ")
        try:
            cent = [float(i.strip()) for i in cent.split(" ")]
            if len(cent) != 2:
                cent = [cent[0], cent[0]]
                tweet("\nBad center so I used... {} instead \n".format(cent))
        except ValueError:
            cent = None
            tweet("\nCenter used... {}\n".format(cent))
    # (4) all should be good
    return True, [out_fc0, out_fc1, cent, SR]


def _tool():
    """run when script is from a tool
    """
    in_fc = sys.argv[1]
    cent = str(sys.argv[2])
    from_north = str(sys.argv[3])
    out_fc0 = sys.argv[4]
    out_fc1 = sys.argv[5]
    return in_fc, from_north, cent, out_fc0, out_fc1


# ----------------------------------------------------------------------
# .... running script or testing code section

gdb_pth = "/".join(script.split("/")[:-2]) + "/Data/Point_tools.gdb"

if len(sys.argv) == 1:
    testing = True
    in_fc = gdb_pth + r"/radial_pnts"
    cent = None
    out_fc0 = gdb_pth + r"/radial"
    out_fc1 = gdb_pth + r"/OD_01"
else:
    testing = False
    in_fc, from_north, cent, out_fc0, out_fc1 = _tool()

#
# (1) Run the test to see whether to continue
#
results = test_envs(in_fc, cent, out_fc0, out_fc1)
cleared, vals = results
#
if not cleared:
    tweet(dedent(msg0).format(script))
else:
    tweet("\nPassed all checks-------------\n")
    #
    # ---- Process section ------------------------------
    #
    pnts_out, plys_out, cent, SR = vals
    desc = _describe(in_fc)
    arcpy.env.workspace = desc['path']  # set the workspace to the gdb
    arr = _xyID(in_fc, to_pnts=True)
    indx = arr['IDs']
    pnts = arr[['Xs', 'Ys']]
    pnts = pnts.view(np.float64).reshape(pnts.shape[0], 2)
    if cent is None:
        cent = np.mean(pnts, axis=0).tolist()
    #
    # (2) perform the radial sort ....
    #
    ang_ab, sort_order = radial_sort(pnts, cent=cent)  # angles and sort_order

    # indx_sorted = indx[sort_order]
    pnts_sorted = pnts[sort_order]
    ang_sorted = ang_ab[sort_order]

    dist_arr = e_dist(cent, pnts_sorted)
    dist_indx = np.argsort(dist_arr)
    dist_id = sort_order[dist_indx]  # indx[dist_indx]

    pairs = [np.asarray([cent, pnt]) for pnt in pnts_sorted]

    # ---- form the output results for use with extend table
    #
    dt = [('IDcent', '<i4'), ('Xp', '<f8'), ('Yp', '<f8'), ('Angle_', '<f8'),
          ('Dist_', '<f8'), ('Orig_ID', '<i4')]
    Xsort = pnts_sorted[:, 0]
    Ysort = pnts_sorted[:, 1]
    ang_id = sort_order + 1  # the centroid ID for the sorted IDs
    ext_tbl = np.empty(arr.shape, dtype=dt)
    nms = ext_tbl.dtype.names
    new_oid = np.arange(1, arr.shape[0]+1)  # to match OBJECTID values
    vals = [new_oid, Xsort, Ysort, ang_sorted, dist_arr, ang_id]
    for i in range(len(nms)):
        ext_tbl[nms[i]] = vals[i]
        ext_tbl2 = np.copy(ext_tbl)
    # ---- create the output point file
    tweet("plys etc out {}, {}".format(pnts_out, plys_out))
    if pnts_out != "#":
        output_points(out_fc0, pnts_sorted.tolist())
        arcpy.da.ExtendTable(out_fc0, 'OBJECTID', ext_tbl, 'IDcent')
    if plys_out != "#":
        output_polylines(out_fc1, SR, pairs)
        arcpy.da.ExtendTable(out_fc1, 'OBJECTID', ext_tbl2, 'IDcent')
if not testing:
    tweet('\nDone....')


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
#    _demo()
