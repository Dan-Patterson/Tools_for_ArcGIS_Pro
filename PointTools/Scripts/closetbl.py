# -*- coding: UTF-8 -*-
"""
:Script:   closetbl.py
:Author:   Dan_Patterson@carleton.ca
:Modified: 2018-03-20
:
:Purpose:  Determine the nearest points based on euclidean distance within
:  a point file.  Emulates Generate Near Table in ArcMap
:
:References:
:----------
: - http://desktop.arcgis.com/en/arcmap/latest/tools/analysis-toolbox/
:   generate-near-table.htm
:
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----

import sys
import numpy as np
import warnings
import arcpy
from arcpytools_pnt import fc_info, tweet

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.1f}'.format}
np.set_printoptions(edgeitems=10, linewidth=120, precision=2,
                    suppress=True, threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')

warnings.simplefilter('ignore', FutureWarning)

script = sys.argv[0]


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


def to_array(in_fc):
    """Extract the shapes and produce a coordinate array.
    """
    shp_fld, oid_fld, shp_type, SR = fc_info(in_fc)
    key_flds = ['SHAPE@X', 'SHAPE@Y']
    in_flds = [oid_fld] + key_flds
    a = arcpy.da.FeatureClassToNumPyArray(in_fc, in_flds)
    uni, idx = np.unique(a[key_flds], True)
    uni_pnts = a[idx]
    #a = a[['SHAPE@X', 'SHAPE@Y']]
    a = uni.view(np.float64).reshape(uni.shape[0], 2)
    return a, uni_pnts, idx


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


def near_tbl(a, b=None, N=1):
    """Return the coordinates and distance to the nearest N points within
    :  an 2D numpy array, 'a', with optional ordering of the inputs.
    :Requires:
    :--------
    : e_dist, fc_info, tweet from arcpytools
    : a - shape coordinates extracted from a point array
    : b - is b is None, then within file differences are used, otherwise
    :     provide another set of coordinates to do between file distances
    : N - the closest N distances and angles to calculate
    :
    :Returns:
    :-------
    :  A structured array containing Origin, Dest and Dist_FT
    """
    # ---- Calculate the distance array ----
    offset = False
    if b is None:
        b = np.copy(a)
        offset = True
    dist = e_dist(a, b, metric='sqeuclidean')  # use sqeuclidean for now
    if dist.ndim == 1:
        print("do stuff")
        return dist
    if offset:
        np.fill_diagonal(dist, np.inf)
    n, m = dist.shape
    rows, cols = np.triu_indices(n, offset, m)  # shape and diag. offset
    idx = dist[rows, cols].argsort()   # slicing with [:2] gives overall 2
    r, c = rows[idx], cols[idx]
    d = np.sqrt(dist[r, c])  # now take the sqrt to get the actual distance
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


frmt = """\n
:Running ... {}
:Using ..... {}
:optional .. {}
:Finding ... {} closest points
:Producing.. {}\n
"""

def nn_kdtree(a, N=1, sorted_=True, to_tbl=True, as_cKD=True):
    """Produce the N closest neighbours array with their distances using
    scipy.spatial.KDTree as an alternative to einsum.

    Parameters:
    -----------
    a : array
        Assumed to be an array of point objects for which `nearest` is needed.
    N : integer
        Number of neighbors to return.  Note: the point counts as 1, so N=3
        returns the closest 2 points, plus itself.
        For table output, max N is limited to 5 so that the tabular output
        isn't ridiculous.
    sorted_ : boolean
        A nice option to facilitate things.  See `xy_sort`.  Its mini-version
        is included in this function.
    to_tbl : boolean
        Produce a structured array output of coordinate pairs and distances.
    as_cKD : boolean
        Whether to use the `c` compiled or pure python version

    References:
    -----------
    `<https://stackoverflow.com/questions/52366421/how-to-do-n-d-distance-
    and-nearest-neighbor-calculations-on-numpy-arrays/52366706#52366706>`_.

    `<https://stackoverflow.com/questions/6931209/difference-between-scipy-
    spatial-kdtree-and-scipy-spatial-ckdtree/6931317#6931317>`_.
    """
    def _xy_sort_(a):
        """mini xy_sort"""
        a_view = a.view(a.dtype.descr * a.shape[1])
        idx = np.argsort(a_view, axis=0, order=(a_view.dtype.names)).ravel()
        a = np.ascontiguousarray(a[idx])
        return a, idx
    #
    def xy_dist_headers(N):
        """Construct headers for the optional table output"""
        vals = np.repeat(np.arange(N), 2)
        names = ['X_{}', 'Y_{}']*N + ['d_{}']*(N-1)
        vals = (np.repeat(np.arange(N), 2)).tolist() + [i for i in range(1, N)]
        n = [names[i].format(vals[i]) for i in range(len(vals))]
        f = ['<f8']*N*2 + ['<f8']*(N-1)
        return list(zip(n, f))
    #
    from scipy.spatial import cKDTree, KDTree
    #
    if sorted_:
        a, idx_srt = _xy_sort_(a)
    # ---- query the tree for the N nearest neighbors and their distance
    if as_cKD:
        t = cKDTree(a)
    else:
        t = KDTree(a)
    dists, indices = t.query(a, N+1)  # so that point isn't duplicated
    dists = dists[:,1:]               # and the array is 2D
    frumXY = a[indices[:,0]]
    indices = indices[:,1:]
    if to_tbl and (N <= 5):
        dt = xy_dist_headers(N+1)  # --- Format a structured array header
        xys =  a[indices]
        new_shp = (xys.shape[0], np.prod(xys.shape[1:]))
        xys = xys.reshape(new_shp)
        #ds = dists[:, 1]  # [d[1:] for d in dists]
        arr = np.concatenate((frumXY, xys, dists), axis=1)
        z = np.zeros((xys.shape[0],), dtype=dt)
        names = z.dtype.names
        for i, j in enumerate(names):
            z[j] = arr[:, i]
        return z
    dists = dists.view(np.float64).reshape(dists.shape[0], -1)
    return dists


def _tool():
    """ run the tool"""
    in_fc = sys.argv[1]
    N = int(sys.argv[2])
    out_tbl = sys.argv[3]
    args = [script, in_fc, N, out_tbl]
    tweet(frmt.format(*args))           # call tweet
    a = to_array(in_fc)                 # call to_array
#    nt = near_tbl(a, b=None, N=N)       # call near_tbl
    nt = nn_kdtree(a, N=3, sorted_=True, to_tbl=True, as_cKD=True)
    tweet("\nnear table\n{}".format(nt)) #.reshape(nt.shape[0], 1)))
    arcpy.da.NumPyArrayToTable(nt, out_tbl)

if len(sys.argv) == 1:
    in_fc = r'C:\GIS\A_Tools_scripts\PointTools\Point_tools.gdb\pnts_01'
    in_fc = r'C:\GIS\A_Tools_scripts\Ice\icebergs.gdb\x_0'
    out_tbl = r'C:\GIS\A_Tools_scripts\Ice\icebergs.gdb\x_0_kd'
    a0 = arcpy.da.FeatureClassToNumPyArray(in_fc,
                                          ['OID@', 'SHAPE@X', 'SHAPE@Y'])
    a, uni_pnts, idx = to_array(in_fc)
    ret = nn_kdtree(a, N=3, sorted_=True, to_tbl=True, as_cKD=True)
else:
    _tool()

# ---------------------------------------------------------------------
if __name__ == "__main__":
    """Main section...   """

#    print("Script... {}".format(script))

"""
fn = r'C:\GIS\A_Tools_scripts\PointTools\Point_tools.gdb\pnts_25'
a = arcpy.da.FeatureClassToNumPyArray(fn, 'Shape')
a = a['Shape']
"""
