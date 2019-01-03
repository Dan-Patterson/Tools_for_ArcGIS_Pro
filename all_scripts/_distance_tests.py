# -*- coding: utf-8 -*-
"""

=======

Script :   .py

Author :   Dan_Patterson@carleton.ca

Modified : 2018-

Purpose :  tools for working with numpy arrays and geometry

Results:
--------
>>> a = np.random.rand(2, n, 3)  # 10, 100, 1000 
::
    1 linalg_norm
    2 sqrt_sum
    3 scipy_distance
    4 sqrt_einsum(
                          time
    method   1      2      3         4
    n = 10   6.25   5.64   94.5     3.18  µs
       100   8.57   7.65  938.0     4.25  µs
      1000  30.5   29.1    9.27 ms 14.0
     10000 204    195     93 ms    71.6 

References:
-----------
`<https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance
-be-calculated-with-numpy>`_.
"""
import sys
import numpy as np
from scipy.spatial import distance


ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=140, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


def linalg_norm(data):
    a, b = data
    return np.linalg.norm(a-b, axis=1)


def sqrt_sum(data):
    a, b = data
    return np.sqrt(np.sum((a-b)**2, axis=1))


def scipy_distance(data):
    a, b = data
    return list(map(distance.euclidean, a, b))


def sqrt_einsum(data):
    a, b = data
    a_min_b = a - b
    return np.sqrt(np.einsum('ij,ij->i', a_min_b, a_min_b))

# =============================================================================
# np.random.RandomState(123)
# 
# n = 10
# a = np.random.rand(2, n, 3)
# =============================================================================
# ----------------------------------
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



def kd_tree(a, as_cKD=True):
    """Construct a KDTree from a point set to facilitate queries
    """
    from scipy.spatial import KDTree, cKDTree
    if as_cKD:
        t = cKDTree(a)
    else:
        t = KDTree(a)
    return t

def _kd_data(as_cKD=True):
    """ testing data for KDTree
    """
#    a = np.random.randint(0, 100, size=(20,2))
    xs = np.array([19.0, 82, 43, 96, 37, 10, 66, 23, 80, 13, 94, 91,
                     61, 43, 67, 78, 76, 34, 56,  3]) * 1.0
    ys = np.array([22.0, 91, 89, 40, 64, 94,  6, 94, 67, 18, 95, 66,
                     87,  1, 41, 50, 20, 16, 84, 98])
    a = np.array(list(zip(xs, ys)))
    idx = a[:, 0].argsort()
    a_s = a[idx]  # sort columns ascending
    t = kd_tree(a_s, as_cKD=as_cKD)
    return a, idx, a_s, t

def sparse_(t, dist=np.inf):
    """form a sparse matrix from a tree
    'dok_matrix', 'coo_matrix', 'dict', or 'ndarray'. Default: 'dok_matrix'.
    a0 =t.sparse_distance_matrix(t, max_distance=np.inf,
                                 output_type='dok_matrix')
    (0, 0)        0.0
    (1, 0)        8.06225774829855
    (2, 0)        80.62257748298549
    (3, 0)        77.6659513557904
     :  :
    (17, 19)      26.476404589747453
    (18, 19)      55.036351623268054
    (19, 19)      0.0
    
    a0.toarray()
    array([[ 0.  ,  8.06, 80.62 ...],  top left
           [ 8.06,  0.  , 76.06 ...],
           [80.62, 76.06,  0.   ...],
           ...
           [...  0.  , 29.15, 26.48],
           [... 29.15,  0.  , 55.04],
           [... 26.48, 55.04,  0.  ]]) bottom right

    np.tril(a0.toarray()) as alternative
    """
    d = t.sparse_distance_matrix(t, dist, p=2)
    return d


def xy_sort(a):
    """Sort 2D array assumed to be coordinates, b x, then y, using argsort.
    Returns the sorted array and the indices of the original positions in the
    input array.

    see: view_sort in arraytools.tools
    """
    a_view = a.view(a.dtype.descr * a.shape[1])
    idx =np.argsort(a_view, axis=0, order=(a_view.dtype.names)).ravel()
    a = np.ascontiguousarray(a[idx])
    return a, idx


def nn_kdtree(a, N=3, sorted=True, to_tbl=True, as_cKD=True):
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
    sorted : boolean
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
        idx =np.argsort(a_view, axis=0, order=(a_view.dtype.names)).ravel()
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
        return list(zip(n,f))
    #    
    from scipy.spatial import cKDTree, KDTree
    #
    idx_orig = []
    if sorted:
        a, idx_orig = _xy_sort_(a)
    # ---- query the tree for the N nearest neighbors and their distance
    if as_cKD:
        t = cKDTree(a)
    else:
        t = KDTree(a)
    dists, indices = t.query(a, N)
    if to_tbl and (N<=5):
        dt = xy_dist_headers(N)  # --- Format a structured array header
        xys = a[indices]
        new_shp = (xys.shape[0], np.prod(xys.shape[1:]))
        xys = xys.reshape(new_shp)
        ds = dists[:, 1:]  #[d[1:] for d in dists]
        arr = np.concatenate((xys, ds), axis=1)
        arr = arr.view(dtype=dt).squeeze()
        return arr
    else:
        dists = dists.view(np.float64).reshape(dists.shape[0], -1)
        return dists #np.array(indices) #, idx_orig]


def sequential_dist(a):
    """sequential distances, an array of coordinates
    """
    diff = a[:-1] - a[1:]
    dis = np.einsum('ij,ij->i', diff, diff)
    return dis


def _data_():
    """data for memmap tests
    """
    f0 = r"C:\GIS\A_Tools_scripts\Polygon_lineTools\Data\samplepoints3_2d.npy"
    f1 = r"C:\GIS\A_Tools_scripts\Polygon_lineTools\Data\samplepoints3_3d.npy"
    a0 = np.load(f0)
    shp0 = a0.shape
    del a0
    a1 = np.load(f1)
    shp1 = a1.shape
    del a1
    a = np.memmap(f0, dtype='float64', mode='r', shape=shp0)
    b = np.memmap(f1, dtype='float64', mode='r', shape=shp1)
    return a, b

def _data_2():
    """
    """
    f = r"\points_2000_from_to.npy"
    p = r"C:\GIS\A_Tools_scripts\Polygon_lineTools\Data"
    fp = p + f
    a = np.load(fp)
    names = list(a.dtype.names[1:])
    N = a.shape[0]
    m = len(names)
    a_s = a[names]
    a_s = np.copy(a_s.view(np.float64).reshape(N, m))
    xy0 = a_s[:, :2]
    xy1 = a_s[:, 2:4]
    uni0, idx0, inv0, cnts0 = np.unique(xy0, True, True, True, axis=0)
    uni1, idx1, inv1, cnts1 = np.unique(xy1, True, True, True, axis=0)
    uni_from = xy0[idx0]
    uni_to = xy1[idx1]
    return xy0, uni_from, uni_to


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """

