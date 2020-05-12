# -*- coding: UTF-8 -*-
"""
spanning_tree
=============

Script: spanning_tree.py

Author: Dan_Patterson@carleton.ca

Modified: 2018-06-13

Original: ... mst.py in my github
    extensive documentation is there.

Purpose:
--------

Produce a spanning tree from a point set.  I have yet to confirm
whether it constitutes a minimum spanning tree, since the implementation
doesn't specify whether Prim's algorithm is being used (see ref. 2)

References:
-----------
`<http://stackoverflow.com/questions/41903502/sort-two-dimensional-
list-python>`_.

`<http://peekaboo-vision.blogspot.ca/2012/02/simplistic-minimum-
spanning-tree-in.html>`_.

Also referenced here...

`<http://stackoverflow.com/questions/34374839/minimum-spanning-tree-
distance-and-graph>`_.

Notes:
------

>>> array 'a' array([[ 0,  0],  constructed for minimum spanning tree example
                     [ 0,  8],
                     [10,  8],
                     [10,  0],
                     [ 3,  4],
                     [ 7,  4]])

(1) sorting

>>> np.lexsort((a[:,1], a[:,0])) sort by x, then y
>>> np.lexsort(a.T) >= np.lexsort((a[:,0], a[:,1])) sort y, x

(2) Distances

unsorted....

>>> np.linalg.norm(a[1:] - a[:-1], axis=1)
array([ 8.0,  10.0,  8.0,  8.1,  4.0])
>>> np.sum(np.linalg.norm(a[1:] - a[:-1], axis=1)) => 38.0622...

sorted....

>>> a_srt = a[np.lexsort(a.T),:]
>>> np.linalg.norm(a_srt[1:] - a_srt[:-1], axis=1)
array([ 8.0,  5.0,  4.0,  5.0,  8.0])
>>> np.sum(np.linalg.norm(a_srt[1:] - a_srt[:-1], axis=1)) => 30.0...

(3) Near results...

>>> coords, dist, n_array = n_near(s, N=2)
  ID     Xo    Yo  C0_x C0_y   C1_x C1_y   Dist0 Dist1
([(0,  0.0, 0.0,  3.0, 4.0,   0.0, 8.0,  5.0,  8.0),
  (1,  0.0, 8.0,  3.0, 4.0,   0.0, 0.0,  5.0,  8.0),
  (2,  3.0, 4.0,  7.0, 4.0,   0.0, 0.0,  4.0,  5.0),
  (3,  7.0, 4.0,  3.0, 4.0,  10.0, 8.0,  4.0,  5.0),
  (4, 10.0, 8.0,  7.0, 4.0,  10.0, 0.0,  5.0,  8.0),
  (5, 10.0, 0.0,  7.0, 4.0,  10.0, 8.0,  5.0,  8.0)],
dtype=[('ID', '<i4'),
       ('Xo', '<f8'), ('Yo', '<f8'),
       ('C0_X', '<f8'), ('C0_Y', '<f8'),
       ('C1_X', '<f8'), ('C1_Y', '<f8'),
       ('Dist0', '<f8'), ('Dist1', '<f8')])

Connections:

>>> o_d
array([(0, 2, 5.0),
       (2, 3, 4.0),
       (2, 1, 5.0),
       (3, 4, 5.0),
       (3, 5, 5.0)],
      dtype=[('Orig', '<i4'), ('Dest', '<i4'), ('Dist', '<f8')])

>>> a[o_d['Orig']]     a[o_d['Dest']]
array([[ 0,  0],   array([[10,  8],
       [10,  8],          [10,  0],
       [10,  8],          [ 0,  8],
       [10,  0],          [ 3,  4],
       [10,  0]])         [ 7,  4]])

distance array:

>>> array([[ 5.0,  8.0,  8.1,  10.0,  12.8],
           [ 5.0,  8.0,  8.1,  10.0,  12.8],
           [ 4.0,  5.0,  5.0,   8.1,   8.1],
           [ 4.0,  5.0,  5.0,   8.1,   8.1],
           [ 5.0,  8.0,  8.1,  10.0,  12.8],
           [ 5.0,  8.0,  8.1,  10.0,  12.8]])

Back to the original distance and sorted array, a_srt.
  The distances are determined using the sorted points, the diagonal
  distances are set to np.inf so that they have the maximal distance.
  The distance values can be sorted to get their indices in the array
  Then the array can be sliced to retrieve the points coordinates and the
  distance array can be sliced to get the distances.

>>> dix = np.arange(d.shape[0])
d[dix, dix] = np.inf

distance array, `d`

>>> d
array([[ inf,  8.0,  5.0,  8.1,  10.0,  12.8],
       [ 8.0,  inf,  5.0,  8.1,  12.8,  10.0],
       [ 5.0,  5.0,  inf,  4.0,  8.1,  8.1],
       [ 8.1,  8.1,  4.0,  inf,  5.0,  5.0],
       [ 10.0,  12.8,  8.1,  5.0,  inf,  8.0],
       [ 12.8,  10.0,  8.1,  5.0,  8.0,  inf]])

>>> np.argsort(d[0])  # => array([2, 1, 3, 4, 5, 0])

>>> a_srt[np.argsort(d[0])]
array([[3, 4], [ 0, 8], [7, 4], [10, 0], [10, 8], [0, 0]])

>>> d[0][np.argsort(d[0])]  # => array([ 5.0, 8.0, 8.1, 10.0, 12.8, inf])

: ---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
#

import sys
import numpy as np
import arcpy
from arcpytools_pnt import fc_info, tweet
from textwrap import dedent

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.1f}'.format}
np.set_printoptions(edgeitems=10, linewidth=100, precision=2,
                    suppress=True, threshold=120,
                    formatter=ft)
np.ma.masked_print_option.set_display('-')

script = sys.argv[0]

# ---- process and functions ----
# (1) run dist_arr which calls _e_dist
# (2) perform the mst (minimum spanning tree, using Prims algorithm)
# (3) connect the points and return the structured array and then the fc


def dist_arr(a, prn=False):
    """Minimum spanning tree prep... see main header
    : paths from given data set...
    """
    # idx = np.lexsort(a.T)  # sort y, then x
    idx = np.lexsort((a[:, 1], a[:, 0]))  # sort X, then Y
    # idx= np.lexsort((a[:,0], a[:,1]))  # sort Y, then X
    a_srt = a[idx, :]
    d = _e_dist(a_srt)
    if prn:
        frmt = """\n    {}\n    :Input array...\n    {}\n\n    :Sorted array...
        {}\n\n    :Distance...\n    {}
        """
        args = [dist_arr.__doc__, a, a_srt, d]  # d.astype('int')]
        print(dedent(frmt).format(*args))
    return idx, a_srt, d


def _e_dist(a):
    """Return a 2D square-form euclidean distance matrix.  For other
    dimensions, use e_dist in ein_geom.py
    """
    b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
    diff = a - b
    d = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff)).squeeze()
    # d = np.triu(d)
    return d


def mst(W, copy_W=True):
    """Determine the minimum spanning tree for a set of points represented
    by their inter-point distances... ie their 'W'eights

    Requires:
    ---------
    W: array
      edge weights (distance, time) for a set of points. W needs to be
      a square array or a np.triu perhaps

    Returns:
    --------

    pairs: array
        the pair of nodes that form the edges
    """
    if copy_W:
        W = W.copy()
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


def connect(a, dists, edges):
    """Return the full spanning tree, with points, connections and distance

    a: point array
        points to connect

    dist: array
        distance array, from _e_dist

    edge: array
        edges, from mst
    """
    p_f = edges[:, 0]
    p_t = edges[:, 1]
    d = dists[p_f, p_t]
    n = p_f.shape[0]
    dt = [('Orig', '<i4'), ('Dest', 'i4'), ('Dist', '<f8')]
    out = np.zeros((n,), dtype=dt)
    out['Orig'] = p_f
    out['Dest'] = p_t
    out['Dist'] = d
    return out


# ---- main section ----
def _demo():
    """A sample run demonstrating the principles and workflow"""
#    a = np.array([[0, 0], [0, 8], [10, 8], [10, 0], [3, 4], [7, 4]])
    pth = script.split("/")[:-2] + ['Point_tools.gdb', 'unsorted_pnts']
    in_fc = "/".join(pth)
    shp_fld, oid_fld, shp_type, SR = fc_info(in_fc)
    a = arcpy.da.FeatureClassToNumPyArray(in_fc,
                                          ['SHAPE@X', 'SHAPE@Y'], "", SR)
    a = a.view(np.dtype('float64')).reshape(a.shape[0], 2)
    idx, a_srt, d = dist_arr(a, prn=False)  # distance array and sorted pnts
    pairs = mst(d)                  # the orig-dest pairs for the mst
    o_d = connect(a_srt, d, pairs)  # produce an o-d structured array
    os = a_srt[pairs[:, 0]]
    ds = a_srt[pairs[:, 1]]
    fr_to = np.array(list(zip(os, ds)))
    return a, o_d, fr_to


def _tool():
    in_fc = sys.argv[1]
    out_fc = sys.argv[2]
    shp_fld, oid_fld, shp_type, SR = fc_info(in_fc)
    out_flds = [oid_fld, shp_fld]
    frmt = """\nScript.... {}\nUsing..... {}\nSR...{}\n"""
    args = [script, in_fc, SR.name]
    msg = frmt.format(*args)
    tweet(msg)
    a = arcpy.da.FeatureClassToNumPyArray(in_fc, shp_fld, "", SR)
    if len(a) >= 2:
        z = np.zeros((a.shape[0], 2))
        z[:, 0] = a['Shape'][:, 0]
        z[:, 1] = a['Shape'][:, 1]
        idx, a_srt, d = dist_arr(z)
        pairs = mst(d)
        o_d = connect(a_srt, d, pairs)

        os = a_srt[pairs[:, 0]]
        ds = a_srt[pairs[:, 1]]

        fr_to = np.array(list(zip(os, ds)))
        s = []
        for pt in fr_to:
            s.append(arcpy.Polyline(arcpy.Array([arcpy.Point(*p) for p in pt]), SR))

        if arcpy.Exists(out_fc):
            arcpy.Delete_management(out_fc)
        arcpy.CopyFeatures_management(s, out_fc)
    else:
        msg2 = """
        |
        ---- Potential User error......
        Technically the script didn't fail.... but...
        You need at least 2 different points... make sure you don't have an
        incorrect selection
        ---- Try again
        |
        """
        tweet(dedent(msg2))


# ---- demo section ----


if len(sys.argv) == 1:
    testing = True
    a, o_d, fr_to = _demo()
else:
    testing = False
    _tool()
"""
Dissolve_management (in_features, out_feature_class, {dissolve_field},
    {statistics_fields}, {multi_part}, {unsplit_lines})
Dissolve_management (in_features, out_feature_class, '', '',
   'SINGLE_PART', 'UNSPLIT_LINES')

collapse from-to data by reshaping it
fr_to.shape
Out[57]: (199, 2, 2)

fr_to.reshape(199*2, 2)
"""
# ---------------------------------------------------------------------
if __name__ == "__main__":
    """Main section...   """
#    print("Script... {}".format(script))
#    a = np.random.randint(1, 10, size=(10,2))
#    a, d, pairs, o_d = _demo()
