# -*- coding: UTF-8 -*-
"""
:Script:   closest_od.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2018-05-11
:
:Purpose:  Determine the nearest points based on euclidean distance between
: point files and then connect them
:
:References:
:----------
: - see near.py documentation for documentation
:
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----

import sys
import warnings
import numpy as np
import arcpy
from arcpytools_pnt import fc_info, tweet

warnings.simplefilter('ignore', FutureWarning)

# from textwrap import dedent

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.1f}'.format}
np.set_printoptions(edgeitems=10, linewidth=120, precision=2,
                    suppress=True, threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')

script = sys.argv[0]

arcpy.env.overwriteOutput = True
# ---- functions ----

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


def line_dir(orig, dest, fromNorth=False):
    """Direction of a line given 2 points
    : orig, dest - two points representing the start and end of a line.
    : fromNorth - True or False gives angle relative to x-axis)
    :Notes:
    :
    """
    orig = np.atleast_2d(orig)
    dest = np.atleast_2d(dest)
    dxy = dest - orig
    ang = np.degrees(np.arctan2(dxy[:, 1], dxy[:, 0]))
    if fromNorth:
        ang = np.mod((450.0 - ang), 360.)
    return ang


def to_array(in_fc):
    """Extract the shapes and produce a coordinate array.
    """
    shp_fld, oid_fld, shp_type, SR = fc_info(in_fc)
    in_flds = [oid_fld] + ['SHAPE@X', 'SHAPE@Y']
    a = arcpy.da.FeatureClassToNumPyArray(in_fc, in_flds)
    a = a[['SHAPE@X', 'SHAPE@Y']]
    a = a.view(np.float64).reshape(a.shape[0], 2).copy()
    return a, SR


def n_near_od(orig, dest, N=3, ordered=True):
    """Return the coordinates and distance to the nearest N points between
    :  two 2D numpy arrays, 'a' and 'b', with optional ordering of the inputs.
    :Requires:
    :--------
    : a, b - ndarrays of uniform int or float dtype.  Extract the fields
    :     representing the x,y coordinates before proceeding.
    : N - number of closest points to return
    :Returns:
    :-------
    :  A structured array is returned containing an ID number.  The ID number
    :  is the ID of the points as they were read.  The array will contain
    :  (C)losest fields and distance fields
    :  (Dest0_X, Dest0_Y, Dest1_X, Dest1_Y, Dist0, Dist1 etc)
    :    representing coordinates
    :  and distance to the required 'closest' points.
    """
    rows, cols = orig.shape
    dt_near = [('Orig_X', '<f8'), ('Orig_Y', '<f8')]
    dt_new = [('Dest{}'.format(i) + '{}'.format(j), '<f8')
              for i in range(N)
              for j in ['_X', '_Y']]
    dt_near.extend(dt_new)
    dt_dist = [('Dist{}'.format(i), '<f8') for i in range(N)]
    dt_ang = [('Angle0', '<f8')]
    dt = [('ID', '<i4'), *dt_near, *dt_dist]
    dt.extend(dt_ang)
    n_array = np.zeros((rows,), dtype=dt)
    n_array['ID'] = np.arange(1, rows+1)  # 1 to N+1 numbering like OBJECTID
    n_array['Orig_X'] = orig[:, 0]
    n_array['Orig_Y'] = orig[:, 1]
    #
    # ---- distance matrix calculation using einsum ----
    d = e_dist(orig, dest, metric='euclidean')
    #
    # ---- format for use in structured array output ----
    # steps are outlined as follows....
    #
    kv = np.argsort(d, axis=1)   # sort 'd' on last axis to get keys
    toos = dest[kv]              # pull coordinates from destination
    s0, s1, s2 = toos.shape
    toos = toos.reshape((s0, s1*s2))
    coords = np.c_[orig, toos]
    dist = np.sort(d)  #[:, 1:]         # slice sorted distances, skip 1st
    # ---- construct the structured array ----
    dt_names = n_array.dtype.names
    s0, s1, s2 = (1, (N+1)*2 + 1, len(dt_names))
    for i in range(1, s1+1):           # coordinate field names
        nm = dt_names[i]
        n_array[nm] = coords[:, i-1]
    dist_names = dt_names[s1:]
    for i in range(N):               # fill n_array with the results
        nm = dist_names[i]
        n_array[nm] = dist[:, i]
    return coords, dist, n_array


def _uniq_by_row_col(a, axis=0):
    """unique to emulate numpy 1.13 ...
    :Requires:
    :--------
    : a - an array of uniform dtype with ndim > 1
    : axis - if 0, then unique rows are returned, if 1, then unique columns
    :
    :References:
    :----------
    : - https://github.com/numpy/numpy/blob/master/numpy/lib/arraysetops.py
    : - http://stackoverflow.com/questions/16970982/
    :          find-unique-rows-in-numpy-array?noredirect=1&lq=1
    :Notes:
    :-----  Must reshape to a contiguous 2D array for this to work...
    : a.dtype.char - ['AllInteger'] + ['Datetime'] + 'S') = 'bBhHiIlLqQpPMmS'
    """
    a = np.asanyarray(a)
    a = np.swapaxes(a, axis, 0)
    orig_shape, _ = a.shape, a.dtype  # orig_shape, orig_dtype
    a = a.reshape(orig_shape[0], -1)
    a = np.ascontiguousarray(a)
    if a.dtype.char in ('bBhHiIlLqQpPMmS'):
        dt = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
    else:
        dt = [('f{i}'.format(i=i), a.dtype) for i in range(a.shape[1])]
    b = a.view(dt)
    _, idx = np.unique(b, return_index=True)
    unique_a = a[idx]
    return unique_a


def make_polyline(pnts, out_=None, sr=None):
    """make the polyline from the points
    """
    s = []
    for pt in pnts:
        arr = arcpy.Array([arcpy.Point(*p) for p in pt])
        s.append(arcpy.Polyline(arr, sr))
    if arcpy.Exists(out_):
        arcpy.Delete_management(out_)
    arcpy.CopyFeatures_management(s, out_)


def connect_od(orig_fc, dest_fc, out_fc, N=1, testing=False):
    """Run the analysis to form the closest point pairs.
    :  Calls n_near_od to produce the nearest features.
    """
    orig, SR0 = to_array(orig_fc)                 # call to_array
    dest, SR1 = to_array(dest_fc)
    # ---- run n_near ----
    coords, dist, n_array = n_near_od(orig, dest, N, ordered=True)
    # ----
#    fr_to = coords[:, :(N+1)*2]
#    frum = fr_to[:, :2]
#    twos = fr_to[:, 2:].reshape(-1, N, 2)
#    r = []
#    for i in range(len(frum)):
#        f = frum[i]
#        t = twos[i]
#        for j in range(len(t)):
#            r.append(np.array([f, t[j]]))
#    rr = np.array(r)
#    r0 = np.array([i[np.lexsort((i[:, 1], i[:, 0]))] for i in rr])  # slicesort
#    r1 = r0.reshape(-1, 4)
#    r2 = _uniq_by_row_col(r1, axis=0)  # use if np.version < 1.13
#    # r2 = unique_2d(r1)
#    r3 = r2[np.argsort(r2[..., 0])]
#    r3 = r3.reshape(-1, 2, 2)
#    #
#    # add angles
    n = n_array.shape[0]
    f = n_array[['Orig_X', 'Orig_Y']].view(np.float64).reshape(n, 2).copy()
    t = n_array[['Dest0_X', 'Dest0_Y']].view(np.float64).reshape(n, 2).copy()
    #
    # calculate the angle
    ang = line_dir(f, t, fromNorth=False)
    n_array['Angle0'] = ang
#   #
    # form the points
    pnts = np.array(list(zip(f, t)))

    if not testing:
        make_polyline(pnts, out_=out_fc, sr=SR0)
        arcpy.da.ExtendTable(out_fc, 'OBJECTID', n_array, 'ID')
    return orig, dest,pnts, n_array


# ---- Run the analysis ----
frmt = """\n
:Running ... {}
:Testing ... {}
:Using .....
:  origins {}
:  destins {}
:Finding ... {} closest points and forming connections
:Producing.. {}\n
"""

# ---- Run the analysis ----
#
def _tool():
    """Run the analysis from the tool
    """
    testing = False
    orig_fc = sys.argv[1]
    dest_fc = sys.argv[2]
    N = int(sys.argv[3])
    out_fc = sys.argv[4]
#    out_fc = r"C:\GIS\A_Tools_scripts\PointTools\Data\Near_testing.gdb\a2b"
    args = ['closest_od.py', testing, orig_fc, dest_fc, N, out_fc]
    return args


if len(sys.argv) == 1:
    testing = True
    pth = "/".join(script.split("/")[:-2]) + "/Data/Near_testing.gdb"
    orig_fc = pth + "/orig_0"
    dest_fc = pth + "/dest_0"
    N = 1
#    out_fc = r"C:\GIS\A_Tools_scripts\PointTools\Data\Near_testing.gdb\a2b"
    out_fc = None
    args = [script, testing, orig_fc, dest_fc, N, out_fc]
else:
    args = _tool()


tweet(frmt.format(*args))                    # call tweet
__, testing, orig_fc, dest_fc, N, out_fc = args
returned = connect_od(orig_fc, dest_fc, out_fc, N=N, testing=testing)   # call connect
orig, dest, pnts, n_array = returned

# ---------------------------------------------------------------------
if __name__ == "__main__":
    """Main section...   """
#    print("Script... {}".format(script))
"""
    in_fc = r"C:\GIS\array_projects\data\Pro_base.gdb\small"
    out_fc = r"C:\GIS\array_projects\data\Pro_base.gdb\ft3"
    N = 1
    testing = True
    a, b, r0, r1, r2, r3 = connect(in_fc, out_fc, N=N, testing=True)
"""
