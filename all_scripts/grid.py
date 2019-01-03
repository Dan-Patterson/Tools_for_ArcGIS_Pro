# -*- coding: UTF-8 -*-
"""
grid
====

Script :   grid.py

Author :   Dan_Patterson@carleton.ca

Modified : 2018-11-23

Purpose :  tools for working with numpy arrays

Requires:
---------
arraytools.tools - nd2struct, stride

Functions:
----------
>>> art.grid.__all__
['check_shapes', 'combine_', 'expand_zone', 'euc_dist', 'euc_alloc', 'expand_',
 'shrink_', 'regions_', 'expand_zone', 'fill_arr', 'reclass_vals',
 'reclass_ranges', 'scale_up']

References:
-----------

`<https://community.esri.com/blogs/dan_patterson/2018/01/19/
combine-data-classification-from-raster-combinations>`_

`<https://stackoverflow.com/questions/48035246/
intersect-multiple-2d-np-arrays-for-determining-zones>`_


---------------------------------------------------------------------
"""
# pylint: disable=C0103
# pylint: disable=R1710
# pylint: disable=R0914

# ---- imports, formats, constants ----
import sys
#from textwrap import dedent, indent
import numpy as np
from arraytools.tools import nd_rec, stride

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.2f}'.format}
np.set_printoptions(edgeitems=5, linewidth=80, precision=2, suppress=True,
                    threshold=500, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

__all__ = ['check_shapes',
           'combine_',
           'euc_dist',
           'euc_alloc',
           'expand_',
           'shrink_',
           'regions_',
           'expand_zone',  # other functions
           'fill_arr',
           'reclass_vals',
           'reclass_ranges',
           'scale_up'
           ]


# ---- array checks and creation --------------------------------------------
# ---- 3D arrays for stacked operations
#
def check_shapes(arrs):
    """Check the shapes of the arrays to ensure they are all equal
    """
    shps = [i.shape for i in arrs]
    eq = np.all(np.array([shps[0] == i for i in shps[1:]]))
    err = "Arrays `arrs` need to have the same shape..."
    if not eq:
        raise ValueError("{}\n{}".format(err, shps))


# ---- array functions ----------------------------------------------------
# (1) combine ----
def combine_(arrs, ret_classes=False):
    """Combine arrays to produce a unique classification scheme

    `arrs` : iterable
        list, tuple of arrays of the same shape
    `ret_classes` : array
        a structured array with the class values for each array and the
        last column is the new_class

    Notes:
    ------
    You should mask any values prior to running this if you want to account
    for nodata values.

    """
    err = "\n...A list of 2D arrays, or a 3D array is required, not...{}\n"
    check_shapes(arrs)
    seq = [isinstance(i, (list, tuple)) for i in arrs]
    is_seq = np.array(seq).all()
    is_nd = np.array([isinstance(i, np.ndarray) for i in arrs]).all()
    if is_seq:
        indices = [np.unique(arr, return_inverse=True)[1] for arr in arrs]
    elif is_nd:
        if isinstance(arrs, np.ma.MaskedArray):
            indices = [np.ma.unique(arrs[i], return_inverse=True)[1]
                       for i in range(arrs.shape[0])]
        else:
            indices = [np.unique(arrs[i], return_inverse=True)[1]
                       for i in range(len(arrs))]
    else:
        print(err.format(arrs))
        return arrs
    #
    M = np.array([item.max()+1 for item in indices])
    M = np.r_[1, M[:-1]]
    strides = M.cumprod()
    indices = np.stack(indices, axis=-1)
    vals = (indices * strides).sum(axis=-1)
    uniqs, cls_new = np.unique(vals, return_inverse=True)
    combo = cls_new.reshape(arrs[0].shape)
    if ret_classes:
        classes = np.array([np.ravel(i) for i in arrs]).T
        classes = np.c_[classes, cls_new]
        classes = nd_rec(classes)    # call nd_rec
        classes = np.unique(classes)
        classes = classes[np.argsort(classes, order=classes.dtype.names)]
        return combo, classes
    return combo


def euc_dist(a, origins=0, cell_size=1):
    """Calculate the euclidean distance and/or allocation

    Parameters:
    -----------
    a : array
        numpy float or integer array
    origins : number, list or tuple
        The locations to calculate distance for.  Anything that is not a mask
        is an origin. If a single number is provided, a `mask` will be created
        using it.  A list/tuple of values can be used for multiple value
        masking.
    cell_size : float, int
        The cell size of the raster.  What does each cell represent on the
        ground.  1.0 is assumed
    """
    from scipy import ndimage as nd
    #
    cell_size = abs(cell_size)
    if cell_size == 0:
        cell_size = 1
    msk = (~np.isin(a, origins)).astype('int')
    dist = nd.distance_transform_edt(msk,
                                     sampling=cell_size,
                                     return_distances=True)
    return dist


def euc_alloc(a, fill_zones=0):
    """Calculate the euclidean distance and/or allocation

    Parameters:
    -----------
    a : array
        numpy float or integer array
    fill_zones : number, list or tuple
        These are the cells/zones to fill with the values of the closest cell.
        If a single number is provided, a `mask` will be created using it.  A
        list or tuple of values can be used to provide multiple value masking.
    dist : boolean
        True, the distance of the closest non-masked value to the masked cell
    alloc : boolean
        True, the value of the closest non-masked value to the masked cell
    """
    from scipy import ndimage as nd
    #
    msk = (np.isin(a, fill_zones)).astype('int')
    idx = nd.distance_transform_edt(msk,
                                    return_distances=False,
                                    return_indices=True)
    alloc = a[tuple(idx)]
    return alloc


def expand_(a, val=1, mask_vals=0, buff_dist=1):
    """Expand/buffer a raster by cells (a distance)
    """
    from scipy import ndimage as nd
    if isinstance(val, (list, tuple)):
        m = np.isin(a, val, invert=True).astype('int')
    else:
        m = np.where(a == val, 0, 1)
    dist, idx = nd.distance_transform_edt(m, return_distances=True,
                                          return_indices=True)
    alloc = a[tuple(idx)]
    a0 = np.where(dist <= buff_dist, alloc, a)  #0)
    return a0


def shrink_(a, val=1, mask_vals=0, buff_dist=1):
    """Expand/buffer a raster by a distance
    """
    from scipy import ndimage as nd
    if isinstance(val, (list, tuple)):
        m = np.isin(a, val, invert=False).astype('int')
    else:
        m = np.where(a == val, 1, 0)
    dist, idx = nd.distance_transform_edt(m, return_distances=True,
                                          return_indices=True)
    alloc = a[tuple(idx)]
    m = np.logical_and(dist > 0, dist <= buff_dist)
    a0 = np.where(m, alloc, a)  #0)
    return a0


def regions_(a, cross=True):
    """Delineate `regions` or `zones` in a raster.  This is analogous to
    `regiongroup` in gis software.  In scipy.ndimage, a `label` is ascribed
    to these groupings.  Any nonzero value will be considered a zone.
    A `structure` is used to filter the raster to describe cell connectivity.

    Parameters:
    -----------
    a : ndarray
        pre-processing may be needed to assign values to `0` which will be
        considered background/offsite
    cross : boolean
       - True, [[0,1,0], [1,1,1], [0,1,0]], diagonal cells not included
       - False, [[1,1,1], [1,1,1], [1,1,1]], diagonals included

    Notes:
    ------
    The use of `np.unique` will ensure that array values are queried and
    returned in ascending order.

    big sample 2000x2000  about 1 sec with 16 classes
        aa = np.repeat(np.repeat(a, 500, axis=1), 500, axis=0)
    """
    from scipy import ndimage as nd
    #
    if (a.ndim != 2) or (a.dtype.kind != 'i'):
        msg = "\nA 2D array of integers is required, you provided\n{}"
        print(msg.format(a))
        return a
    if cross:
        struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    else:
        struct = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    #
    u = np.unique(a)
    out = np.zeros_like(a, dtype=a.dtype)
    details = []
    is_first = True
    for i in u:
        z = np.where(a == i, 1, 0)
        s, n = nd.label(z, structure=struct)
        details.append([i, n])
        m = np.logical_and(out == 0, s != 0)
        if is_first:
            out = np.where(m, s, out)
            is_first = False
            n_ = n
        else:
            out = np.where(m, s+n_, out)
            n_ += n
    details = np.array(details)
    details = np.c_[(details, np.cumsum(details[:, 1]))]
    return out, details


# ---- Raster functions ------------------------------------
#
def expand_zone(a, zone=None, win=2):
    """Expand a value (zone) in a 2D array, normally assumed to represent a
    raster surface.

    zone : number
        The value/class to expand into the surrounding cells
    win : list/tuple
        select a (2, 2) or (3, 3) moving window
    """
    msg = "\nYou need a zone that is within the range of values."
    if zone is None:
        print(msg)
        return a, None
    if (zone < a.min()) or (zone > a.max()):
        print(msg)
        return a, None
    if win not in (2, 3):
        win = 2
    p = [1, 0][win == 2]  # check for 2 or 3 in win
    ap = np.pad(a, pad_width=(1, p), mode="constant", constant_values=(0, 0))
    # n, m = ap.shape
    if win == 2:
        a_c = ap[1:, 1:]  # for 2x2 even
    elif win == 3:
        a_c = ap[1:-1, 1:-1]  # for 3x3 odd
    a_s = stride(ap, win=(win, win), stepby=(win, win))  # stride the array
    r, c = a_s.shape[:2]
    out = []
    x = a_s.shape[0]
    y = a_s.shape[1]
    for i in range(x):
        for j in range(y):
            if zone in a_s[i, j]:
                out.append(1)
            else:
                out.append(0)
    out1 = np.asarray(out).reshape(r, c)
    out = np.repeat(np.repeat(out1, 2, axis=1), 2, axis=0)
    dx, dy = np.array(out.shape) - np.array(a.shape)
    if dx != 0:
        out = out[:dx, :dy]
    final = np.where(out == 1, zone, a_c)
    return final


def fill_arr(a, win=(3, 3)):
    """try filling an array
    as in fill, sinks
    """
    #fd = np.array([[32, 64, 128], [16, 0, 1], [8, 4, 2]])  # flow direction
#    if (zone < a.min()) or (zone > a.max()) or (zone is None):
#        print("\nYou need a zone that is within the range of values.")
#        return a, None
    if win[0] == 3:
        pr = 1
    else:
        pr = 0
    ap = np.pad(a, pad_width=(1, pr), mode="constant", constant_values=(0, 0))
    if win == (2, 2):
        a_c = ap[1:, 1:]  # for 2x2 even
    elif win == (3, 3):
        a_c = ap[1:-1, 1:-1]   # for 3x3 odd
    a_s = stride(a_c, win=win)  # stride the array
    r, c = a_s.shape[:2]
    out = []
    x = a_s.shape[0]
    y = a_s.shape[1]
    for i in range(x):
        for j in range(y):
            # do stuff
            sub = a_s[i, j].ravel()
            edges = np.asarray([sub[:4], sub[5:]]).ravel()
            e_min = edges[np.argmax(edges)]  # argmax or argmin???
            if sub[4] < e_min:
                out.append(e_min)
            else:
                out.append(sub[4])
    out = np.asarray(out).reshape(r, c)
    return out  # , a_s, ap, a_c


# (xx) reclass_vals .... code section
def reclass_vals(a, old_vals=[], new_vals=[], mask=False, mask_val=None):
    """Reclass an array of integer or floating point values.

    Requires:
    --------
    old_vals : number(s)
        list/array of values to reclassify
    new_bins : number(s)
        new class values for old value
    mask : boolean
        Does the raster contains nodata values or values to be masked
    mask_val : number(s)
        Values to use as the mask

    Array dimensions will be squeezed.

     >>> a = np.arange(10).reshape(2,5)
     >>> a0 = np.arange(5)
     >>> art.reclass_vals(a, a0, np.ones_like(a0))
     # array([[0, 1, 2, 3, 4]   ==> array([[1, 1, 1, 1, 1],
     #        [5, 6, 7, 8, 9]])           [5, 6, 7, 8, 9]])
    """
    a_rc = np.copy(a)
    args = [old_vals, new_vals]
    msg = "\nError....\nLengths of old and new classes not equal \n{}\n{}\n"
    if len(old_vals) != len(new_vals):
        print(msg.format(*args))
        return a
    old_new = np.array(list(zip(old_vals, new_vals)), dtype='int32')
    for pair in old_new:
        q = (a == pair[0])
        a_rc[q] = pair[1]
    return a_rc


# ----------------------------------------------------------------------
# (15) reclass .... code section
def reclass_ranges(a, bins=[], new_bins=[], mask=False, mask_val=None):
    """Reclass an array of integer or floating point values based on old and
    new range values.

    Requires:
    --------

    bins : list/array
        Sequential list/array of the lower limits of each class include one
        value higher to cover the upper range.
    new_bins : number(s)
        New class values for each bin
    mask : boolean
        Does the raster contains nodata values or values to be masked
    mask_val : number(s)
        Values to use as the mask

    Array dimensions will be squeezed.

    >>> z = np.arange(3*5).reshape(3,5)
    >>> bins = [0, 5, 10, 15]
    >>> new_bins = [1, 2, 3, 4]
    >>> z_recl = reclass(z, bins, new_bins, mask=False, mask_val=None)
       # ==> .... z                     ==> .... z_recl
       array([[ 0,  1,  2,  3,  4],   array([[1, 1, 1, 1, 1],
              [ 5,  6,  7,  8,  9],          [2, 2, 2, 2, 2],
              [10, 11, 12, 13, 14]])         [3, 3, 3, 3, 3]])
    """
    a_rc = np.zeros_like(a)
    if len(bins) < 2:  # or (len(new_bins <2)):
        print("Bins = {} new = {} won't work".format(bins, new_bins))
        return a
    if len(new_bins) < 2:
        new_bins = np.arange(1, len(bins)+2)
    new_classes = list(zip(bins[:-1], bins[1:], new_bins))
    for rc in new_classes:
        q1 = (a >= rc[0])
        q2 = (a < rc[1])
        a_rc = a_rc + np.where(q1 & q2, rc[2], 0)
    return a_rc


# (16) scale .... code section
def scale_up(a, x=2, y=2, num_z=None):
    """Scale the input array repeating the array values up by the
    x and y factors.

    Requires:
    --------
    a : array
        An ndarray, 1D arrays will be upcast to 2D
    x, y : numbers
        Factors to scale the array in x (col) and y (row).  Scale factors
        must be greater than 2
    num_z : number
        For 3D, produces the 3rd dimension, ie. if num_z = 3 with the
        defaults, you will get an array with shape=(3, 6, 6).  If
        num_z != None or 0, then the options are 'repeat', 'random'.
        With 'repeat' the extras are kept the same and you can add random
        values to particular slices of the 3rd dimension, or multiply them.

    Returns:
    -------
    >>> a = np.array([[0, 1, 2], [3, 4, 5]]
    >>> b = scale(a, x=2, y=2)
    array([[0, 0, 1, 1, 2, 2],
           [0, 0, 1, 1, 2, 2],
           [3, 3, 4, 4, 5, 5],
           [3, 3, 4, 4, 5, 5]])

    Notes:
    -----
    >>> a = np.arange(2*2).reshape(2,2)
    array([[0, 1],
           [2, 3]])
    >>> f_(scale(a, x=2, y=2, num_z=2))
    Array... shape (3, 4, 4), ndim 3, not masked
    0, 0, 1, 1    0, 0, 1, 1    0, 0, 1, 1
    0, 0, 1, 1    0, 0, 1, 1    0, 0, 1, 1
    2, 2, 3, 3    2, 2, 3, 3    2, 2, 3, 3
    2, 2, 3, 3    2, 2, 3, 3    2, 2, 3, 3
    sub (0)       sub (1)       sub (2)

    """
    if (x < 1) or (y < 1):
        print("x or y scale < 1... \n{}".format(scale_up.__doc__))
        return None
    a = np.atleast_2d(a)
    z0 = np.tile(a.repeat(x), y)  # repeat for x, then tile
    z1 = np.hsplit(z0, y)         # split into y parts horizontally
    z2 = np.vstack(z1)            # stack them vertically
    if a.shape[0] > 1:            # if there are more, repeat
        z3 = np.hsplit(z2, a.shape[0])
        z3 = np.vstack(z3)
    else:
        z3 = np.vstack(z2)
    if num_z not in (0, None):
        d = [z3]
        for i in range(num_z):
            d.append(z3)
        z3 = np.dstack(d)
        z3 = np.rollaxis(z3, 2, 0)
    return z3


# ---- demo functions -------------------------------------------------------
#
def _demo_combine():
    """demo combine
    dt = [('a', '<i8'), ('b', '<i8'), ('c', '<i8'), ('vals', '<i8')]
    """
    a = np.array([[0, 0, 0, 4, 4, 4, 1, 1, 1],
                  [0, 0, 0, 4, 4, 4, 1, 1, 1],
                  [0, 0, 0, 4, 4, 4, 1, 1, 1],
                  [2, 2, 2, 1, 1, 1, 2, 2, 2],
                  [2, 2, 2, 1, 1, 1, 2, 2, 2],
                  [2, 2, 2, 1, 1, 1, 2, 2, 2],
                  [1, 1, 1, 4, 4, 4, 0, 0, 0],
                  [1, 1, 1, 4, 4, 4, 0, 0, 0],
                  [1, 1, 1, 4, 4, 4, 0, 0, 0]])

    b = np.array([[0, 0, 0, 1, 1, 1, 2, 2, 2],
                  [0, 0, 0, 1, 1, 1, 2, 2, 2],
                  [0, 0, 0, 1, 1, 1, 2, 2, 2],
                  [3, 3, 3, 4, 4, 4, 5, 5, 5],
                  [3, 3, 3, 4, 4, 4, 5, 5, 5],
                  [3, 3, 3, 4, 4, 4, 5, 5, 5],
                  [0, 0, 0, 1, 1, 1, 2, 2, 2],
                  [0, 0, 0, 1, 1, 1, 2, 2, 2],
                  [0, 0, 0, 1, 1, 1, 2, 2, 2]])

    c = np.array([[0, 0, 0, 3, 3, 3, 0, 0, 0],
                  [0, 0, 0, 3, 3, 3, 0, 0, 0],
                  [0, 0, 0, 3, 3, 3, 0, 0, 0],
                  [1, 1, 1, 4, 4, 4, 1, 1, 1],
                  [1, 1, 1, 4, 4, 4, 1, 1, 1],
                  [1, 1, 1, 4, 4, 4, 1, 1, 1],
                  [2, 2, 2, 5, 5, 5, 2, 2, 2],
                  [2, 2, 2, 5, 5, 5, 2, 2, 2],
                  [2, 2, 2, 5, 5, 5, 2, 2, 2]])
#    ret = combine_(*[a, b, c])
    return a, b, c  #, ret


def _demo_reclass():
    """
    : -
    """
    a = np.array([[9, 8, 2, 3, 4, 3, 5, 5, 2, 2],
                  [4, 1, 4, 2, 4, 2, 4, 2, 3, 2],
                  [5, 3, 5, 4, 5, 4, 5, 3, 1, 2],
                  [5, 2, 3, 1, 4, 4, 3, 5, 4, 3],
                  [2, 3, 2, 5, 5, 2, 5, 5, 4, 4],
                  [5, 3, 4, 4, 2, 1, 3, 2, 4, 3],
                  [3, 2, 3, 3, 3, 4, 3, 2, 4, 3],
                  [4, 5, 2, 3, 2, 2, 3, 1, 4, 4],
                  [3, 5, 5, 5, 2, 2, 4, 3, 4, 4],
                  [4, 5, 4, 5, 3, 2, 4, 3, 1, 3]])
#    f = np.array([[32, 64, 128], [16, 0, 1], [8, 4, 2]])
#    out, out2 = expand_zone(a, zone=1, win=(3,3))
    a_rc = reclass_vals(a,
                        old_vals=[1, 3, 5],
                        new_vals=[9, 5, 1],
                        mask=False,
                        mask_val=None)
    return a_rc

def _demo_euclid():
    """ euclid functions"""
    a = np.array([[0, 1, 0, 0, 2, 0, 0, 0],   # note the block of 0's in the
                  [1, 0, 0, 1, 1, 0, 0, 0],   # top right corner
                  [0, 1, 0, 1, 1, 0, 0, 0],
                  [0, 2, 0, 3, 0, 0, 0, 3],
                  [0, 1, 2, 0, 0, 4, 2, 0],
                  [4, 0, 0, 3, 2, 5, 1, 0],
                  [1, 1, 0, 0, 0, 5, 0, 0],   # and the bottom right
                  [0, 5, 0, 4, 0, 3, 0, 0]])
    b = np.array(([0, 1, 1, 1, 1],  # from scipy help
                  [0, 0, 1, 1, 1],
                  [0, 1, 1, 1, 1],
                  [0, 1, 1, 1, 0],
                  [0, 1, 1, 0, 0]))
    return a, b

# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
# https://stackoverflow.com/questions/47861214/
