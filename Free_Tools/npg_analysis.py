# -*- coding: utf-8 -*-
"""
============
npg_analysis
============

Script :
    npg_analysis.py

Author :
    Dan_Patterson@carleton.ca

Modified :
    2019-09-07

Purpose :
    Analysis tools for the Geom class.

Notes:

References
----------
Derived from arraytools ``convex_hull, mst, near, n_spaced``

"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

# import sys
from textwrap import dedent
import numpy as np

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=5, linewidth=120, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

__all__ = ['closest_n', 'distances', 'not_closer', 'n_check', 'n_near',
           'n_spaced', 'intersects', 'knn', 'knn0', '_dist_arr_', '_e_dist_',
           'mst', 'connect', 'concave']


# ===========================================================================
# ---- def section: def code blocks go here ---------------------------------
def closest_n(a, N=3, ordered=True):
    """A shell to `n_near`, see its doc string"""
    coords, dist, n_array = n_near(a, N=N, ordered=ordered)
    return coords, dist, n_array


def distances(a, b):
    """Distances for 2D arrays using einsum.  Based on a simplified version
    of e_dist in arraytools.
    """
    diff = a[:, None] - b
    return np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))


def not_closer(a, min_d=1, ordered=False):
    """Find the points that are separated by a distance greater than
     min_d.  This ensures a degree of point spacing

    Parameters
    ----------
     `a` : coordinates
         2D array of coordinates.
     `min_d` : number
         Minimum separation distance.
     `ordered` : boolean
         Order the input points.

    Returns
    -------
    - b : points where the spacing condition is met
    - c : the boolean array indicating which of the input points were valid.
    - d : the distance matrix

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


def n_check(a):  # N=3, order=True):
    """n_check prior to running n_near analysis

    Parameters
    ----------
    Two 2D array of X,Y coordinates required.  Parse your data to comply.
    """
    has_err = False
    if isinstance(a, (list, tuple, np.ndarray)):
        if (hasattr(a[0], '__len__')) and (len(a[0]) == 2):
            return True
        has_err = True
    else:
        has_err = True
    if has_err:
        print(n_check.__doc__)
        return False


def n_near(a, N=3, ordered=True):
    """Return the coordinates and distance to the nearest N points within
      an 2D numpy array, 'a', with optional ordering of the inputs.

    Parameters
    ----------
    `a` : array
        An ndarray of uniform int or float dtype.  Extract the fields
        representing the x,y coordinates before proceeding.
    `N` : number
         Number of closest points to return.

    Returns
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
    rows, _ = a.shape
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


def n_spaced(L=0, B=0, R=10, T=10, min_space=1, num=10, verbose=True):
    """Produce num points within the bounds specified by the extent (L,B,R,T)

    Parameters
    ----------
    L(eft), B, R, T(op) : numbers
        Extent coordinates.
    min_space : number
        Minimum spacing between points.
    num : number
        Number of points... this value may not be reached if the extent
        is too small and the spacing is large relative to it.
    """
    #
    def _pnts(L, B, R, T, num):
        """Create the points"""
        xs = (R-L) * np.random.random_sample(size=num) + L
        ys = (T-B) * np.random.random_sample(size=num) + B
        return np.array(list(zip(xs, ys)))

    def _not_closer(a, min_space=1):
        """Find the points that are greater than min_space in the extent."""
        b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
        diff = b - a
        dist = np.einsum('ijk,ijk->ij', diff, diff)
        dist_arr = np.sqrt(dist).squeeze()
        case = ~(np.triu(dist_arr <= min_space, 1)).any(0)
        return a[case]
    #
    cnt = 1
    n = num * 2  # check double the number required as a check
    result = 0
    frmt = "Examined: {}  Found: {}  Need: {}"
    a0 = []
    while (result < num) and (cnt < 6):  # keep using random points
        a = _pnts(L, B, R, T, num)
        if cnt > 1:
            a = np.vstack((a0, a))
        a0 = _not_closer(a, min_space)
        result = len(a0)
        if verbose:
            print(dedent(frmt).format(n, result, num))
        cnt += 1
        n += n
    # perform the final sample and calculation
    use = min(num, result)
    a0 = a0[:use]  # could use a0 = np.random.shuffle(a0)[:num]
    a0 = a0[np.argsort(a0[:, 0])]
    return a0


# ==== intersection
#
def intersects(*args):
    """Line intersection check.  Two lines or 4 points that form the lines.

    Parameters
    ----------
      intersects(line0, line1) or intersects(p0, p1, p2, p3)
        p0, p1 -> line 1
        p2, p3 -> line 2

    Returns:
    --------
    boolean, if the segments do intersect

    References:
    -----------
    `<https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-
    line-segments-intersect#565282>`_.
    """
    if len(args) == 2:
        p0, p1, p2, p3 = *args[0], *args[1]
    elif len(args) == 4:
        p0, p1, p2, p3 = args
    else:
        raise AttributeError("Pass 2, 2-pnt lines or 4 points to the function")
    #
    # ---- First check ----   np.cross(p1-p0, p3-p2 )
    x0, y0, x1, y1, x2, y2, x3, y3 = *p0, *p1, *p2, *p3  # points to xs and ys
    #
    # ---- First check ----   np.cross(p1-p0, p3-p2 )
    denom = (x1 - x0) * (y3 - y2) - (x3 - x2) * (y1 - y0)
    if denom == 0.0:
        return False
    #
    # ---- Second check ----  np.cross(p1-p0, p0-p2 )
    denom_gt0 = denom > 0  # denominator greater than zero
    #
    s_numer = (x1 - x0) * (y0 - y2) - (y1 - y0) * (x0 - x2)
    if (s_numer < 0) == denom_gt0:
        return False
    #
    # ---- Third check ----  np.cross(p3-p2, p0-p2)
    t_numer = (x3 - x2) * (y0 - y2) - (y3 - y2) * (x0 - x2)
    if (t_numer < 0) == denom_gt0:
        return False
    #
    if ((s_numer > denom) == denom_gt0) or ((t_numer > denom) == denom_gt0):
        return False
    #
    # ---- check to see if the intersection point is one of the input points
    # substitute p0 in the equation  These are the intersection points
    t = t_numer / denom
    x = x0 + t * (x1 - x0)
    y = y0 + t * (y1 - y0)
    # be careful that you are comparing tuples to tuples, lists to lists
    if sum([(x, y) == tuple(i) for i in [p0, p1, p2, p3]]) > 0:
        return False
    return True


def knn(p, pnts, k=1, return_dist=True):
    """
    Calculates k nearest neighbours for a given point.

    Parameters:
    -----------
    p :array
        x,y reference point
    pnts : array
        Points array to examine
    k : integer
        The `k` in k-nearest neighbours

    Returns
    -------
    Array of k-nearest points and optionally their distance from the source.
    """

    def _remove_self_(p, pnts):
        """Remove a point which is duplicated or itself from the array
        """
        keep = ~np.all(pnts == p, axis=1)
        return pnts[keep]

    def _e_2d_(p, a):
        """ array points to point distance... mini e_dist
        """
        diff = a - p[np.newaxis, :]
        return np.einsum('ij,ij->i', diff, diff)

    p = np.asarray(p)
    k = max(1, min(abs(int(k)), len(pnts)))
    pnts = _remove_self_(p, pnts)
    d = _e_2d_(p, pnts)
    idx = np.argsort(d)
    if return_dist:
        return pnts[idx][:k], d[idx][:k]
    return pnts[idx][:k]


def knn0(pnts, p, k):
    """Calculates `k` nearest neighbours for a given point, `p`, relative to
     otherpoints.

    Parameters
    ----------
    points : array
        list of points
    p : array-like
        reference point, two numbers representing x, y
    k : integer
        number of neighbours

    Returns
    -------
    list of the k nearest neighbours, based on squared distance
    """
    p = np.asarray(p)
    pnts = np.asarray(pnts)
    diff = pnts - p[np.newaxis, :]
    d = np.einsum('ij,ij->i', diff, diff)
    idx = np.argsort(d)[:k]
    return pnts[idx].tolist()


# ---- minimum spanning tree
#
def _dist_arr_(a, verbose=False):
    """Minimum spanning tree prep... """
    a = a[~np.isnan(a[:, 0])]
    idx = np.lexsort((a[:, 1], a[:, 0]))  # sort X, then Y
    # idx= np.lexsort((a[:, 0], a[:, 1]))  # sort Y, then X
    a_srt = a[idx, :]
    d = _e_dist_(a_srt)
    if verbose:
        frmt = """\n    {}\n    :Input array...\n    {}\n\n    :Sorted array...
        {}\n\n    :Distance...\n    {}
        """
        args = [_dist_arr_.__doc__, a, a_srt, d]  # d.astype('int')]
        print(dedent(frmt).format(*args))
    return d, idx, a_srt


def _e_dist_(a):
    """Return a 2D square-form euclidean distance matrix.  For other
    dimensions, use e_dist in ein_geom.py"""
    b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
    diff = a - b
    d = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff)).squeeze()
    # d = np.triu(d)
    return d


def mst(W, calc_dist=True):
    """Determine the minimum spanning tree for a set of points represented
    by their inter-point distances. ie their `W`eights

    Parameters
    ----------
    W : array, normally an interpoint distance array
        Edge weights for example, distance, time, for a set of points.
        W needs to be a square array or a np.triu perhaps

    calc_dist : boolean
        True, if W is a points array, calculate W as the interpoint distance.
        False means that W is not a points array, but some other `weight`
        representing the interpoint relationship

    Returns
    -------
    pairs - the pair of nodes that form the edges
    """
    W = W[~np.isnan(W[:, 0])]
    if calc_dist:
        W = _e_dist_(W)
    if W.shape[0] != W.shape[1]:
        raise ValueError("W needs to be square matrix of edge weights")
    Np = W.shape[0]
    pairs = []
    pnts_seen = [0]  # Add the first point
    n_seen = 1
    # exclude self connections by assigning inf to the diagonal
    diag = np.arange(Np)
    W[diag, diag] = np.inf
    #
    while n_seen != Np:
        new_edge = np.argmin(W[pnts_seen], axis=None)
        new_edge = divmod(new_edge, Np)
        new_edge = [pnts_seen[new_edge[0]], new_edge[1]]
        pairs.append(new_edge)
        pnts_seen.append(new_edge[1])
        W[pnts_seen, new_edge[1]] = np.inf
        W[new_edge[1], pnts_seen] = np.inf
        n_seen += 1
    return np.vstack(pairs)


def connect(a, dist_arr, edges):
    """Return the full spanning tree, with points, connections and distance

    Parameters
    ----------
    a : array
        A point array
    dist : array
        The distance array, from _e_dist
    edge : array
        The edges derived from mst
    """
    a = a[~np.isnan(a[:, 0])]
    p_f = edges[:, 0]
    p_t = edges[:, 1]
    d = dist_arr[p_f, p_t]
    n = p_f.shape[0]
    dt = [('Orig', '<i4'), ('Dest', 'i4'), ('Dist', '<f8')]
    out = np.zeros((n,), dtype=dt)
    out['Orig'] = p_f
    out['Dest'] = p_t
    out['Dist'] = d
    return out


# ---- find
#
# ---- concave hull
def concave(points, k, pip_check=False):
    """Calculates the concave hull for given points

    Parameters
    ----------
    points : array-like
        initially the input set of points with duplicates removes and
        sorted on the Y value first, lowest Y at the top (?)
    k : integer
        initially the number of points to start forming the concave hull,
        k will be the initial set of neighbors
    pip_check : boolean
        Whether to do the final point in polygon check.  Not needed for closely
        spaced dense point patterns.
    knn0, intersects, angle, point_in_polygon : functions
        Functions used by `concave`

    Notes:
    ------
    This recursively calls itself to check concave hull.

    p_set : The working copy of the input points

    70,000 points with final pop check removed, 1011 pnts on ch
        23.1 s ± 1.13 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
        2min 15s ± 2.69 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
    """
    PI = np.pi

    def _angle_(p0, p1, prv_ang=0):
        """Angle between two points and the previous angle, or zero.
        """
        ang = np.arctan2(p0[1] - p1[1], p0[0] - p1[0])
        a0 = (ang - prv_ang)
        a0 = a0 % (PI * 2) - PI
        return a0

    def _point_in_polygon_(pnt, poly):  # pnt_in_poly(pnt, poly):  #
        """Point is in polygon. ## fix this and use pip from arraytools
        """
        x, y = pnt
        N = len(poly)
        for i in range(N):
            x0, y0, xy = [poly[i][0], poly[i][1], poly[(i + 1) % N]]
            c_min = min([x0, xy[0]])
            c_max = max([x0, xy[0]])
            if c_min < x <= c_max:
                p = y0 - xy[1]
                q = x0 - xy[0]
                y_cal = (x - x0) * p / q + y0
                if y_cal < y:
                    return True
        return False
    # ----
    k = max(k, 3)  # Make sure k >= 3
    if isinstance(points, np.ndarray):  # Remove duplicates if not done already
        p_set = np.unique(points, axis=0).tolist()
    else:
        pts = []
        p_set = [pts.append(i) for i in points if i not in pts]  # Remove dupls
        p_set = np.array(p_set)
        del pts
    if len(p_set) < 3:
        raise Exception("p_set length cannot be smaller than 3")
    elif len(p_set) == 3:
        return p_set  # Points are a polygon already
    k = min(k, len(p_set) - 1)  # Make sure k neighbours can be found
    frst_p = cur_p = min(p_set, key=lambda x: x[1])
    hull = [frst_p]       # Initialize hull with first point
    p_set.remove(frst_p)  # Remove first point from p_set
    prev_ang = 0
    # ----
    while (cur_p != frst_p or len(hull) == 1) and len(p_set) > 0:
        if len(hull) == 3:
            p_set.append(frst_p)          # Add first point again
        knn_pnts = knn0(p_set, cur_p, k)  # Find nearest neighbours
        cur_pnts = sorted(knn_pnts, key=lambda x: -_angle_(x, cur_p, prev_ang))
        its = True
        i = -1
        while its and i < len(cur_pnts) - 1:
            i += 1
            last_point = 1 if cur_pnts[i] == frst_p else 0
            j = 1
            its = False
            while not its and j < len(hull) - last_point:
                its = intersects(hull[-1], cur_pnts[i], hull[-j - 1], hull[-j])
                j += 1
        if its:  # All points intersect, try a higher number of neighbours
            return concave(points, k + 1)
        prev_ang = _angle_(cur_pnts[i], cur_p)
        cur_p = cur_pnts[i]
        hull.append(cur_p)  # Valid candidate was found
        p_set.remove(cur_p)
    if pip_check:
        for point in p_set:
            if not _point_in_polygon_(point, hull):
                return concave(points, k + 1)
    #
    hull = np.array(hull)
    return hull


def _demo():
    """ """
    # L, R, B, T = [300000, 300100, 5025000, 5025100]
    L, B, R, T = [1, 1, 10, 10]
    tol = 1
    N = 10
    a = n_spaced(L, B, R, T, tol, num=N, verbose=True)
    return a


# ==== Processing finished ====
# ===========================================================================
#
if __name__ == "__main__":
    """optional location for parameters"""
    # print("")
