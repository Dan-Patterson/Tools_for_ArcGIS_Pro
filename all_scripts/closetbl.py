# -*- coding: UTF-8 -*-
"""
closetbl
========

Script : closetbl.py

Author : Dan_Patterson@carleton.ca

Modified : 2018-03-31

Purpose:
    Determine the nearest points based on euclidean distance within a point
    file.  Emulates Generate Near Table in ArcMap

References:
----------
[1] http://desktop.arcgis.com/en/arcmap/latest/tools/analysis-toolbox/
generate-near-table.htm

:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----

import sys
import numpy as np
import arcpy
from arcpytools import fc_info, tweet

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.1f}'.format}
np.set_printoptions(edgeitems=10, linewidth=120, precision=2,
                    suppress=True, threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')

script = sys.argv[0]


# ---- functions ----
def e_dist(a, b, metric='euclidean'):
    """Distance calculation for 1D, 2D and 3D points using einsum

    - ``a, b   :`` list, tuple, array in 1,2 or 3D form
    - ``metric :`` euclidean ('e','eu'...), sqeuclidean ('s','sq'...),
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


def to_array(in_fc):
    """Extract the shapes and produce a coordinate array.
    """
    shp_fld, oid_fld, shp_type, SR = fc_info(in_fc)
    in_flds = [oid_fld, shp_fld]
    a = arcpy.da.FeatureClassToNumPyArray(in_fc, in_flds)
    a = a[shp_fld]
    return a


def line_dir(orig, dest, fromNorth=False):
    """Direction of a line given 2 points

    - ``orig, dest :`` two points representing the start and end of a line.
    - ``fromNorth  :`` True or False gives angle relative to x-axis)
    """
    orig = np.atleast_2d(orig)
    dest = np.atleast_2d(dest)
    dxy = dest - orig
    ang = np.degrees(np.arctan2(dxy[:, 1], dxy[:, 0]))
    if fromNorth:
        ang = np.mod((450.0 - ang), 360.)
    return ang


def near_tbl(a, b=None, N=1):
    """Return the coordinates and distance to the nearest N points within
    an 2D numpy array, `a`, with optional ordering of the inputs.

    Parameters:
    -----------

    `e_dist`, `fc_info`, and `tweet` from arcpytools

    -`a` : array
        shape coordinates extracted from a point array
    -`b` : array or None
        if b is None, then within file differences are used, otherwise provide
        another set of coordinates to do between file distances
    -`N` : number
        the closest N distances and angles to calculate

    Returns:
    -------

    A structured array containing Origin, Dest and Dist_FT (from - to points)
    """
    # ---- Calculate the distance array ----
    offset = False
    if b is None:
        b = np.copy(a)
        offset = True
    dist = e_dist(a, b, metric='euclidean')
    if dist.ndim == 1:
        print("do stuff")
        return dist
    if offset:
        np.fill_diagonal(dist, np.inf)
    n, m = dist.shape
    rows, cols = np.triu_indices(n, offset, m)  # shape and diag. offset
    idx = dist[rows, cols].argsort()   # slicing with [:2] gives overall 2
    r, c = rows[idx], cols[idx]
    d = dist[r, c]
    az0 = line_dir(a[r], b[c], fromNorth=True)
    az1 = line_dir(b[c], a[r], fromNorth=True)
    z0 = list(zip(r, c, a[r, 0], a[r, 1], b[c, 0], b[c, 1], d, az0))
    z1 = list(zip(c, r, b[c, 0], b[c, 1], a[r, 0], a[r, 1], d, az1))
    dt = [('Orig', '<i4'), ('Dest', '<i4'),
          ('X_orig', '<f8'), ('Y_orig', '<f8'),
          ('X_dest', '<f8'), ('Y_dest', '<f8'),
          ('OD_dist', '<f8'), ('Azim_N', '<f8')]
    ft = np.array(z0 + z1, dtype=dt)
    ft_idx = np.argsort(ft, order=('Orig', 'OD_dist'))  # sort by Orig first
    ft2 = ft[ft_idx]
    num_pnts = len(a)
    nt = np.asanyarray([ft2[ft2['Orig'] == i][:N] for i in range(num_pnts)])
    nt = np.asanyarray([i for i in nt if len(i) > 0])
    nt = nt.reshape((np.product(nt.shape),))
    return nt


# ---- Run the analysis ----
#
def tool():
    """ run the tool"""
    in_fc = sys.argv[1]
    N = int(sys.argv[2])
    out_tbl = sys.argv[3]
    args = [script, in_fc, N, out_tbl]
    tweet(frmt.format(*args))           # call tweet
    a = to_array(in_fc)                 # call to_array
    nt = near_tbl(a, b=None, N=N)       # call near_tbl
    tweet("\nnear table\n{}".format(nt.reshape(nt.shape[0], 1)))
    arcpy.da.NumPyArrayToTable(nt, out_tbl)


frmt = """\n
:Running ... {}
:Using ..... {}
:Finding ... {} closest points
:Producing.. {}\n
"""
if len(sys.argv) == 1:
    fn = r'C:\GIS\points\points.gdb\Fishnet_label'
    a = arcpy.da.FeatureClassToNumPyArray(fn, 'Shape')
    a = a['Shape']
else:
    tool()


# ---------------------------------------------------------------------
if __name__ == "__main__":
    """Main section...   """

#    print("Script... {}".format(script))
