# -*- coding: UTF-8 -*-
"""
:Script:   closest.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-04-11
:
:Purpose:  Determine the nearest points based on euclidean distance within
:  a point file and then connect them
:References:
:----------
: - see near.py documentation for documentation
:
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----

import sys
import numpy as np
import arcpy
from arcpytools_pnt import fc_info, tweet

# from textwrap import dedent

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.1f}'.format}
np.set_printoptions(edgeitems=10, linewidth=120, precision=2,
                    suppress=True, threshold=140, formatter=ft)
np.ma.masked_print_option.set_display('-')

script = sys.argv[0]

arcpy.env.overwriteOutput = True
# ---- functions ----


def n_near(a, N=3, ordered=True):
    """Return the coordinates and distance to the nearest N points within
    :  an 2D numpy array, 'a', with optional ordering of the inputs.
    :Requires:
    :--------
    : a - an ndarray of uniform int or float dtype.  Extract the fields
    :     representing the x,y coordinates before proceeding.
    : N - number of closest points to return
    :Returns:
    :-------
    :  A structured array is returned containing an ID number.  The ID number
    :  is the ID of the points as they were read.  The array will contain
    :  (C)losest fields and distance fields
    :  (C0_X, C0_Y, C1_X, C1_Y, Dist0, Dist1 etc) representing coordinates
    :  and distance to the required 'closest' points.
    """
    if not (isinstance(a, (np.ndarray)) and (N >= 1)):
        print("\nInput error...read the docs\n\n{}".format(n_near.__doc__))
        return a
    rows, cols = a.shape
    dt_near = [('Xo', '<f8'), ('Yo', '<f8')]
    dt_new = [('C{}'.format(i) + '{}'.format(j), '<f8')
              for i in range(N)
              for j in ['_X', '_Y']]
    dt_near.extend(dt_new)
    dt_dist = [('Dist{}'.format(i), '<f8') for i in range(N)]
    dt = [('ID', '<i4'), *dt_near, *dt_dist]
    n_array = np.zeros((rows,), dtype=dt)
    n_array['ID'] = np.arange(rows)
    # ---- distance matrix calculation using einsum ----
    if ordered:
        a = a[np.argsort(a[:, 0])]
    b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
    diff = b - a
    dist = np.einsum('ijk,ijk->ij', diff, diff)
    d = np.sqrt(dist).squeeze()
    # ---- format for use in structured array output ----
    # steps are outlined as follows....
    #
    kv = np.argsort(d, axis=1)       # sort 'd' on last axis to get keys
    coords = a[kv]                   # pull out coordinates using the keys
    s0, s1, s2 = coords.shape
    coords = coords.reshape((s0, s1*s2))
    dist = np.sort(d)[:, 1:]         # slice sorted distances, skip 1st
    # ---- construct the structured array ----
    dt_names = n_array.dtype.names
    s0, s1, s2 = (1, (N+1)*2 + 1, len(dt_names))
    for i in range(0, s1):           # coordinate field names
        nm = dt_names[i+1]
        n_array[nm] = coords[:, i]
    dist_names = dt_names[s1:s2]
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


def connect(in_fc, out_fc, N=1, testing=False):
    """Run the analysis to form the closest point pairs.
    :  Calls n_near to produce the nearest features.
    """
    shp_fld, oid_fld, shp_type, SR = fc_info(in_fc)
    a = arcpy.da.FeatureClassToNumPyArray(in_fc, shp_fld, "", SR)
    dt = '<f8'
    b = np.array([tuple(i) for i in a[shp_fld]], dtype=dt)
    coords, dist, n_array = n_near(b, N, ordered=True)  # ---- run n_near ----
    fr_to = coords[:, :(N+1)*2]
    frum = fr_to[:, :2]
    twos = fr_to[:, 2:].reshape(-1, N, 2)
    r = []
    for i in range(len(frum)):
        f = frum[i]
        t = twos[i]
        for j in range(len(t)):
            r.append(np.array([f, t[j]]))
    rr = np.array(r)
    r0 = np.array([i[np.lexsort((i[:, 1], i[:, 0]))] for i in rr])  # slicesort
    r1 = r0.reshape(-1, 4)
    r2 = _uniq_by_row_col(r1, axis=0)  # use if np.version < 1.13
    # r2 = unique_2d(r1)
    r3 = r2[np.argsort(r2[..., 0])]
    r3 = r3.reshape(-1, 2, 2)
    if not testing:
        s = []
        for pt in r3:
            arr = arcpy.Array([arcpy.Point(*p) for p in pt])
            s.append(arcpy.Polyline(arr, SR))
        if arcpy.Exists(out_fc):
            arcpy.Delete_management(out_fc)
        arcpy.CopyFeatures_management(s, out_fc)
        return None
    else:
        return a, b, r0, r1, r2, r3


# ---- Run the analysis ----
frmt = """\n
:Running ... {}
:Using ..... {}
:Finding ... {} closest points and forming connections
:Producing.. {}\n
"""

in_fc = sys.argv[1]
N = int(sys.argv[2])
out_fc = sys.argv[3]
args = [script, in_fc, N, out_fc]
tweet(frmt.format(*args))                    # call tweet
ret = connect(in_fc, out_fc, N=N, testing=False)   # call connect

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
