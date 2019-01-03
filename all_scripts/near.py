# -*- coding: UTF-8 -*-
"""
near
====

Script :   near.py

Author :   Dan_Patterson@carleton.ca

Modified: 2018-03-28

Purpose :
    Determine the nearest points based on euclidean distance within
    a point file.

    Also, a function to ensure points have a minimum spacing.

References:
----------

**creating meshgrids from x,y data and plotting**

[1]
`2D array of values based on coordinates`__:

__ http://stackoverflow.com/questions/30764955/python-numpy-create-2darray-of-values-based-on-coordinates

[2]
`2D histogram issues`__:

__ https://github.com/numpy/numpy/issues/7317


**distance calculations and related** .... (scipy, skikit-learn)

[3]
`scipy spatial distance`__:

__ https://github.com/scipy/scipy/blob/v0.18.1/scipy/spatial/distance.py#L1744-L2211

[4]
`einsum and distance calculations`__:

    __ http://stackoverflow.com/questions/32154475/einsum-and-distance-calculations

[5]
`optimizations for calculating squared euclidean distances`__:

__ http://stackoverflow.com/questions/23983748/possible-optimizations-for-calculating-squared-euclidean-distance

[6]
`pairwise euclidean distances`__:

__ http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html

[7]
`euclidean distance between points`__:

__ http://stackoverflow.com/questions/1871536/euclidean-distance-between-points-in-two-different-numpy-arrays-not-within

---------------------------------------------------------------------
"""
# ---- imports, formats, constants ----

import sys
import numpy as np
from textwrap import dedent

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.1f}'.format}
np.set_printoptions(edgeitems=10, linewidth=120, precision=2,
                    suppress=True, threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')

script = sys.argv[0]

__all__ = ['distances',
           'not_closer',
           'n_check',
           'n_near',
           '_pnts',
           '_n_near_demo',
           '_not_closer_demo'
           ]
# ---- functions ----


def distances(a, b):
    """A fast implementation for distance calculations

    Requires:
    --------
    `a`, `b` - arrays
        2D arrays of equal size!! ... can be the same array

    Notes:
    -----
        Similar to my e_dist and scipy cdist
    """
    if (len(a) != len(b)):
        print("\nInput array error...\n{}".format(distances.__doc__))
        return None
    d0 = np.subtract.outer(a[:, 0], b[:, 0])
    d1 = np.subtract.outer(a[:, 1], b[:, 1])
    return np.hypot(d0, d1)


def not_closer(a, min_d=1, ordered=False):
    """Find the points that are separated by a distance greater than
     min_d.  This ensures a degree of point spacing

    Requires:
    --------
     `a` : coordinates
         2D array of coordinates.
     `min_d` : number
         Minimum separation distance
     `ordered` : boolean
         Order the input points
    """
    if ordered:
        a = a[np.argsort(a[:, 0])]
    b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
    diff = b - a
    d = np.einsum('ijk,ijk->ij', diff, diff)
    d = np.sqrt(d).squeeze()
    c = ~(np.triu(d <= min_d, 1)).any(0)
    b = a[c]
    return b, c, d


def n_check(a, N=3, order=True):
    """n_check prior to running n_near analysis

    Requires:
    --------
       Two 2D array of X,Y coordinates required.  Parse your data to comply.
    """
    has_err = False
    if isinstance(a, (list, tuple, np.ndarray)):
        if (hasattr(a[0], '__len__')) and (len(a[0]) == 2):
            return True
        else:
            has_err = True
    else:
        has_err = True
    if has_err:
        print(n_check.__doc__)
        return False


def n_near(a, N=3, ordered=True):
    """Return the coordinates and distance to the nearest N points within
      an 2D numpy array, 'a', with optional ordering of the inputs.

    Requires:
    --------

    `a` : array
        An ndarray of uniform int or float dtype.  Extract the fields
        representing the x,y coordinates before proceeding.

    `N` : number
         Number of closest points to return

    Returns:
    -------
      A structured array is returned containing an ID number.  The ID number
      is the ID of the points as they were read.  The array will contain
      (C)losest fields and distance fields
      (C0_X, C0_Y, C1_X, C1_Y, Dist0, Dist1 etc) representing coordinates
      and distance to the required 'closest' points.
    """
    if not (isinstance(a, (np.ndarray)) and (N > 1)):
        print("\nInput error...read the docs\n\n{}".format(n_near.__doc__))
        return a
    rows, cols = a.shape
    dt_near = [('Xo', '<f8'), ('Yo', '<f8')]
    dt_new = [('C{}'.format(i) + '{}'.format(j), '<f8')
              for i in range(N)
              for j in ['_X', '_Y']]
    dt_near.extend(dt_new)
    dt_dist = [('Dist{}'.format(i), '<f8') for i in range(N)]
    # dt = [('ID', '<i4')]  + dt_near + dt_dist # python 2.7
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


def _pnts(L, B, R, T, num, as_int=True, as_recarry=False):
    """Create the points"""
    xs = (R-L) * np.random.random_sample(size=num) + L
    ys = (T-B) * np.random.random_sample(size=num) + B
    a = np.array(list(zip(xs, ys)))
    if as_int:
        a = a.astype('int32')
    a = a[np.argsort(a[:, 0])]
    return a


def _n_near_demo():
    """Demonstrate n_near function"""
    frmt = """
    -----------------------------------------------------------------
    Closest {} points for points in an array.  Results returned as
      a structured array with coordinates and distance values.
    {} ....\n
    Input points... array 'a'
    {}\n
    output array
    """
    vals = [[0.0, 0.0], [0, 0.5], [0, 1], [0, 1.5], [0, 2], [1, 2],
            [2, 2], [2, 1], [2, 0], [1, 0], [1, 0]]
    vals = [tuple(i) for i in vals]      # has to be tuples
#    dt = np.dtype([('X', '<f8'), ('Y', '<f8')])
    a = np.array(vals, dtype='float64')
#    b = np.array(vals, dtype=dt)
    N = 2
    coords, dist, n_r = n_near(a, N=N, ordered=True)  # a, coords, dist,
    args = [N, _n_near_demo.__doc__, a]
    print(dedent(frmt).format(*args))
    n = len(n_r[0])
    names = n_r.dtype.names
    frmt = "{!s:>7}"*n
    print(frmt.format(*names))
    for i in n_r:  # .reshape(n_r.shape[0],-1)
        frmt = "{:> 7.2f}"*n
        print(frmt.format(*i))
    print(":"+"-"*66)
    return a, coords, dist, n_r


def _not_closer_demo():
    """Perform 'closest' analysis and produce a histogram classed using
      distance bands.  The histogram can be used to produce a 2D raster
      representation of the point pattern.
      np.histogram2d(x, y, bins=10, range=None, normed=False, weights=None)
    """
    a = np.array([[6, 79], [7, 24], [17, 11], [33, 47], [37, 46], [38, 42],
                  [46, 98], [48, 66], [49, 21], [57, 40], [71, 74], [74, 86],
                  [85, 20], [87, 98], [88,  5], [88, 56], [89, 95], [89, 55],
                  [92, 97], [96, 93]], dtype=np.int32)
    b, c, d = not_closer(a, ordered=False, min_d=20)
    idx = np.arange(len(a))
    e = np.c_[a, c, idx]
    x_bin = np.arange(0, 101, 5)
    y_bin = np.arange(0, 101, 5)
    #
    h, hx, hy = np.histogram2d(e[:, 1], e[:, 0], bins=(x_bin, y_bin),
                               weights=e[:, -2])
    h = h.astype('int64')  # return the counts in the 10x10 cells
    return a, b, c, d, e, h


# ---------------------------------------------------------------------
if __name__ == "__main__":
    """Main section...   """
#    print("Script... {}".format(script))
#    a, coords, d, n_r = _n_near_demo()
#    a, b, c, d, e, h = _not_closer_demo()
