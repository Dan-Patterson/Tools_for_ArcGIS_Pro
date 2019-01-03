# -*- coding: UTF-8 -*-ct
"""
:Script:   rasters.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2018-02-13
:Purpose:  tools for working with numpy arrays
:Useage:
:
:Requires:
:  arraytools.tools - nd2struct, stride
:References:
: https://community.esri.com/blogs/dan_patterson/2018/01/19/
:       combine-data-classification-from-raster-combinations
: - combine
: https://stackoverflow.com/questions/48035246/
:       intersect-multiple-2d-np-arrays-for-determining-zones
: original def find_labels(*arrs):
:---------------------------------------------------------------------:
"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

# ---- imports, formats, constants ----
import sys
import numpy as np
from tools import nd2struct, stride

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

__all__ = ['combine_',      # 3D array functions
           'check_stack',
           'mask_stack',    # statistical functions
           'stack_sum', 'stack_cumsum',
           'stack_prod', 'stack_cumprod', 'stack_min', 'stack_mean',
           'stack_median', 'stack_max', 'stack_std', 'stack_var',
           'stack_percentile',
           'stack_stats']


# ---- array checks and creation --------------------------------------------
# ---- 3D arrays for stacked operations
#
def check_shapes(arrs):
    """Check the shapes of the arrays to ensure they are all equal
    """
    shps = [i.shape for i in arrs]
    eq = np.all(np.array([shps[0] == i for i in shps[1:]]))
    err = "Arrays arr not of the same shape..."
    if not eq:
        raise ValueError("{}\n{}".format(err, shps))


def check_stack(arrs):
    """Do the basic checking of the stack to ensure that a 3D array is
    :  generated
    """
    err1 = "Object, structured arrays not supported, current type..."
    err2 = "3D arrays supported current ndim..."
    if isinstance(arrs, (list, tuple)):
        arrs = np.array(arrs)
    if arrs.dtype.kind in ('O', 'V'):
        raise ValueError("{} {}".format(err1, arrs.dtype.kind))
    if arrs.ndim != 3:
        raise ValueError("{} {}".format(err2, arrs.ndim))
    return arrs


def mask_stack(arr, nodata=None):
    """Produce masks for a 3d array"""
    if (nodata is None) or (arr.ndim < 2) or (arr.ndim > 3):
        print("\n...mask_stack requires a 3d array and a nodata value\n")
        return arr
    m = (arr[:, ...] == nodata).any(0)
    msk = [m for i in range(arr.shape[0])]
    msk = np.array(msk)
    a_m = np.ma.MaskedArray(arr, mask=msk)
    return a_m


# ---- 3D array functions ----------------------------------------------------
# (1) combine ----
def combine_(*arrs, ret_classes=False):
    """Combine arrays to produce a unique classification scheme
    : arrs - list, tuple of arrays of the same shape
    : ret_classes - a structured array with the class values for each array
    :               and the last column is the new_class
    :Notes:
    :------
    : You should mask any values prior to running this if you want to account
    : for nodata values
    :
    :References:
    :-----------
    : https://stackoverflow.com/questions/48035246/
    :       intersect-multiple-2d-np-arrays-for-determining-zones
    : original: def find_labels(*arrs):
    """
    err = "\n...A list of 2D arrays, or a 3D array is required, not...{}\n"
    check_shapes(arrs)
    seq = [isinstance(i, (list, tuple)) for i in arrs]
    is_seq = np.array(seq).all()
    is_nd = [isinstance(i, np.ndarray) for i in arrs]
    is_nd.all()
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
        classes = nd2struct(classes)
        classes = np.unique(classes)
        classes = classes[np.argsort(classes, order=classes.dtype.names[-1])]
        return combo, classes
    return combo


# ---- Statistics for stacked arrays (3D) ------------------------------------
#
def stack_percentile(arrs, q=50, nodata=None):
    """nanpercentile for an array stack with optional nodata masked
    :  arrs - either a list, tuple of arrays or an array with ndim=3
    :  q - the percentile
    :  nodata - nodata value, numeric or np.nan (will upscale integers)
    """
    a = check_stack(arrs)
    if nodata is not None:
        a = mask_stack(a, nodata=nodata)
    nan_per = np.nanpercentile(a, q=q, axis=0)
    return nan_per


def stack_sum(arrs, nodata=None):
    """see stack_stats"""
    a = check_stack(arrs)
    if nodata is not None:
        a = mask_stack(a, nodata=nodata)
    return np.nansum(a, axis=0)


def stack_cumsum(arrs, nodata=None):
    """see stack_stats"""
    a = check_stack(arrs)
    if nodata is not None:
        a = mask_stack(a, nodata=nodata)
    return np.nancumsum(a, axis=0)


def stack_prod(arrs, nodata=None):
    """see stack_stats"""
    a = check_stack(arrs)
    if nodata is not None:
        a = mask_stack(a, nodata=nodata)
    return np.nanprod(a, axis=0)


def stack_cumprod(arrs, nodata=None):
    """see stack_stats"""
    a = check_stack(arrs)
    if nodata is not None:
        a = mask_stack(a, nodata=nodata)
    return np.nancumprod(a, axis=0)


def stack_min(arrs, nodata=None):
    """see stack_stats"""
    a = check_stack(arrs)
    if nodata is not None:
        a = mask_stack(a, nodata=nodata)
    return np.nanmin(a, axis=0)


def stack_mean(arrs, nodata=None):
    """see stack_stats"""
    a = check_stack(arrs)
    if nodata is not None:
        a = mask_stack(a, nodata=nodata)
    return np.nanmean(a, axis=0)


def stack_median(arrs, nodata=None):
    """see stack_stats"""
    a = check_stack(arrs)
    if nodata is not None:
        a = mask_stack(a, nodata=nodata)
    return np.nanmedian(a, axis=0)


def stack_max(arrs, nodata=None):
    """see stack_stats"""
    a = check_stack(arrs)
    if nodata is not None:
        a = mask_stack(a, nodata=nodata)
    return np.nanmax(a, axis=0)


def stack_std(arrs, nodata=None):
    """see stack_stats"""
    a = check_stack(arrs)
    if nodata is not None:
        a = mask_stack(a, nodata=nodata)
    return np.nanstd(a, axis=0)


def stack_var(arrs, nodata=None):
    """see stack_stats"""
    a = check_stack(arrs)
    if nodata is not None:
        a = mask_stack(a, nodata=nodata)
    return np.nanvar(a, axis=0)


def stack_stats(arrs, ax=0, nodata=None):
    """All statistics for arrs
    :
    :  arrs - either a list, tuple of arrays or an array with ndim=3
    :  ax - axis, either, 0 (by band) or (1,2) to get a single value for
    :       each band
    :  nodata - nodata value, numeric or np.nan (will upscale integers)
    """
    arrs = check_stack(arrs)
    a_m = mask_stack(arrs, nodata=nodata)
    nan_sum = np.nansum(a_m, axis=ax)
    nan_min = np.nanmin(a_m, axis=ax)
    nan_mean = np.nanmean(a_m, axis=ax)
    nan_median = np.nanmean(a_m, axis=ax)
    nan_max = np.nanmax(a_m, axis=ax)
    nan_std = np.nanstd(a_m, axis=ax)
    nan_var = np.nanvar(a_m, axis=ax)
    stats = [nan_sum, nan_min, nan_mean, nan_median, nan_max, nan_std, nan_var]
    if len(ax) == 1:
        nan_cumsum = np.nancumsum(a_m, axis=ax)
        stats.append(nan_cumsum)
    return stats


def expand_zone(a, zone=None, win=2):
    """Expand a value (zone) in a 2D array, normally assumed to represent a
    :  raster surface.
    :zone - the value/class to expand into the surrounding cells
    :win - select a (2, 2) or (3, 3) moving window
    :
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
    #n, m = ap.shape
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
    """try filling an array"""
#    fd = np.array([[32, 64, 128], [16, 0, 1], [8, 4, 2]])  # flow direction
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
        #w, h = win
    elif win == (3, 3):
        #w, h = win
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
            e_min = edges[np.argmin(edges)]
            if sub[4] < e_min:
                out.append(e_min)
            else:
                out.append(sub[4])
    out = np.asarray(out).reshape(r, c)
    return out  # , a_s, ap, a_c


# (xx) reclass_vals .... code section
def reclass_vals(a, old_vals=None, new_vals=None, mask=False, mask_val=None):
    """Reclass an array of integer or floating point values.
    :Requires:
    :--------
    : old_vals - list/array of values to reclassify
    : new_bins - new class values for old value
    : mask - whether the raster contains nodata values or values to
    :        be masked with mask_val
    : Array dimensions will be squeezed.
    :Example:
    :-------
    :  array([[ 0,  1,  2,  3,  4],   array([[1, 1, 1, 1, 1],
    :         [ 5,  6,  7,  8,  9],          [2, 2, 2, 2, 2],
    :         [10, 11, 12, 13, 14]])         [3, 3, 3, 3, 3]])
    """
    err = "Inputs are incorrect...old_vals: {}, new_vals: {}"
    if old_vals is None or new_vals is None:
        print(err.format(old_vals, new_vals))
        return a
    a_rc = np.copy(a)
    args = [old_vals, new_vals]
    if len(old_vals) != len(new_vals):
        print(err.format(*args))
        return a
    old_new = np.array(list(zip(old_vals, new_vals)), dtype='int32')
    for pair in old_new:
        q = (a == pair[0])
        a_rc[q] = pair[1]
    return a_rc


# ----------------------------------------------------------------------
# (15) reclass .... code section
def reclass_ranges(a, bins=None, new_bins=None, mask=False, mask_val=None):
    """Reclass an array of integer or floating point values based on old and
    :  new range values
    :Requires:
    :--------
    : bins - sequential list/array of the lower limits of each class
    :        include one value higher to cover the upper range.
    : new_bins - new class values for each bin
    : mask - whether the raster contains nodata values or values to
    :        be masked with mask_val
    : Array dimensions will be squeezed.
    :Example:
    :-------
    :  z = np.arange(3*5).reshape(3,5)
    :  bins = [0, 5, 10, 15]
    :  new_bins = [1, 2, 3, 4]
    :  z_recl = reclass(z, bins, new_bins, mask=False, mask_val=None)
    :  ==> .... z                     ==> .... z_recl
    :  array([[ 0,  1,  2,  3,  4],   array([[1, 1, 1, 1, 1],
    :         [ 5,  6,  7,  8,  9],          [2, 2, 2, 2, 2],
    :         [10, 11, 12, 13, 14]])         [3, 3, 3, 3, 3]])
    """
    err = "Bins = {} new = {} won't work".format(bins, new_bins)
    if bins is None or new_bins is None:
        print(err)
    a_rc = np.zeros_like(a)
    if len(bins) < 2:  # or (len(new_bins <2)):
        print(err)
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
    :  x and y factors.
    :Requires:
    :--------
    : a - an ndarray, 1D arrays will be upcast to 2D
    : x, y - factors to scale the array in x (col) and y (row)
    :      - scale factors must be greater than 2
    : num_z - for 3D, produces the 3rd dimension, ie. if num_z = 3 with the
    :    defaults, you will get an array with shape=(3, 6, 6)
    : how - if num_z != None or 0, then the options are
    :    'repeat', 'random'.  With 'repeat' the extras are kept the same
    :     and you can add random values to particular slices of the 3rd
    :     dimension, or multiply them etc etc.
    :Returns:
    :-------
    : a = np.array([[0, 1, 2], [3, 4, 5]]
    : b = scale(a, x=2, y=2)
    :   =  array([[0, 0, 1, 1, 2, 2],
    :             [0, 0, 1, 1, 2, 2],
    :             [3, 3, 4, 4, 5, 5],
    :             [3, 3, 4, 4, 5, 5]])
    :Notes:
    :-----
    :  a=np.arange(2*2).reshape(2,2)
    :  a = array([[0, 1],
    :             [2, 3]])
    :  f_(scale(a, x=2, y=2, num_z=2))
    :  Array... shape (3, 4, 4), ndim 3, not masked
    :   0, 0, 1, 1    0, 0, 1, 1    0, 0, 1, 1
    :   0, 0, 1, 1    0, 0, 1, 1    0, 0, 1, 1
    :   2, 2, 3, 3    2, 2, 3, 3    2, 2, 3, 3
    :   2, 2, 3, 3    2, 2, 3, 3    2, 2, 3, 3
    :   sub (0)       sub (1)       sub (2)
    :--------
    """
    if (x < 1) or (y < 1):
        print("x or y scale < 1...\n{}".format(scale_up.__doc__))
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


def _demo():
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


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
# https://stackoverflow.com/questions/47861214/
# using-numpy-as-strided-to-retrieve-subarrays-centered-on-main-diagonal
"""
theta = inclination of sun from 90 in radians
theta2 = slope angle
phi = ((450 - sun orientation from north in degrees) mod 360) * 180/pi
"""
