# -*- coding: UTF-8 -*-ct
"""
stackstats
===========

Script:   stackstats.py

Author:   Dan.Patterson@carleton.ca

Modified: 2018-11-23

Purpose:  tools for working with numpy arrays

Requires:
---------
    arraytools.tools - nd2struct, stride

References:
-----------

`<https://community.esri.com/blogs/dan_patterson/2018/02/06/cell-\
statistics-made-easy-raster-data-over-time>`_.

"""
# pylint: disable=C0103
# pylint: disable=R1710
# pylint: disable=R0914

# ---- imports, formats, constants ----
import sys
import numpy as np

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=5, linewidth=100, precision=2, suppress=True,
                    threshold=150, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

__all__ = ['check_shapes', 'check_stack', 'mask_stack',
           'stack_sum', 'stack_cumsum',  # statistical functions
           'stack_prod', 'stack_cumprod',
           'stack_min', 'stack_mean',
           'stack_median', 'stack_max',
           'stack_std', 'stack_var',
           'stack_percentile',
           'stack_stats',
           'stack_stats_tbl']


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
    generated
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


def mask_stack(arrs, nodata=None):
    """Produce masks for a 3d array"""
    if (nodata is None) or (arrs.ndim < 2) or (arrs.ndim > 3):
        print("\n...mask_stack requires a 3d array and a nodata value\n")
        return arrs
    m = (arrs[:, ...] == nodata).any(0)
    msk = [m for i in range(arrs.shape[0])]
    msk = np.array(msk)
    a_m = np.ma.MaskedArray(arrs, mask=msk)
    return a_m


# ---- Statistics for stacked arrays (3D) ------------------------------------
#
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


def stack_percentile(arrs, q=50, nodata=None):
    """nanpercentile for an array stack with optional nodata masked

    -arrs :
        either a list, tuple of arrays or an array with ndim=3
    - q :
        the percentile
    - nodata :
        nodata value, numeric or np.nan (will upscale integers)
    """
    a = check_stack(arrs)
    if nodata is not None:
        a = mask_stack(a, nodata=nodata)
    nan_per = np.nanpercentile(a, q=q, axis=0)
    return nan_per


def stack_stats(arrs, ax=0, nodata=None):
    """All statistics for arrs.

    - arrs :
        either a list, tuple of arrays or an array with ndim=3
    - ax :
        axis, either, 0 (by band) or (1,2) to get a single value for each band
    - nodata :
        nodata value, numeric or np.nan (will upscale integers)
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
    if np.isscalar(ax):
        nan_cumsum = np.nancumsum(a_m, axis=ax)
        stats.append(nan_cumsum)
    return stats


def stack_stats_tbl(arrs, nodata=None):  # col_names, args):
    """Produce the output table

    Returns:
    --------
    Table of statistical results by band.  The dtype is shown below
    dtype=[('Band', '<i4'), ('N', '<i4'), ('N_nan', '<i4'), ('Sum', '<f8'),
           ('Min', '<f8'), ('Mean', '<f8'), ('Med', '<f8'), ('Max', '<f8'),
           ('Std', '<f8'), ('Var', '<f8')])
    """
    stats = stack_stats(arrs, ax=(1, 2), nodata=nodata)
    d = [(i, '<f8')
         for i in ['Sum', 'Min', 'Mean', 'Med', 'Max', 'Std', 'Var']]
    dts = [('Band', '<i4'), ('N', '<i4'), ('N_nan', '<i4')] + d
    N, r, c = arrs.shape
    cols = len(dts)
    z = np.empty(shape=(N,), dtype=dts)
    z[z.dtype.names[0]] = np.arange(0, N)
    z[z.dtype.names[1]] = np.array([r*c]*N)
    z[z.dtype.names[2]] = np.count_nonzero(arrs == nodata, axis=(1, 2))
    for i in range(cols-3):
        z[z.dtype.names[i+3]] = stats[i]
    return z


def _demo_stack():
    """
    demo stack :
        Simply 31 layers shaped (31, 100, 150) with uniform values one for
        each day, numbers from 1 to 31.
    >>> stack_stats_tbl(stack)
    array([( 0, 15000, 0,   15000.,   1.,   1.,   1.,   1.,  0.,  0.),
           ( 1, 15000, 0,   30000.,   2.,   2.,   2.,   2.,  0.,  0.),
           ( 2, 15000, 0,   45000.,   3.,   3.,   3.,   3.,  0.,  0.),
           ( 3, 15000, 0,   60000.,   4.,   4.,   4.,   4.,  0.,  0.),
           ( 4, 15000, 0,   75000.,   5.,   5.,   5.,   5.,  0.,  0.),
           ( 5, 15000, 0,   90000.,   6.,   6.,   6.,   6.,  0.,  0.),
           .... snip ....
           (27, 15000, 0,  420000.,  28.,  28.,  28.,  28.,  0.,  0.),
           (28, 15000, 0,  435000.,  29.,  29.,  29.,  29.,  0.,  0.),
           (29, 15000, 0,  450000.,  30.,  30.,  30.,  30.,  0.,  0.),
           (30, 15000, 0,  465000.,  31.,  31.,  31.,  31.,  0.,  0.)],
          dtype=[('Band', '<i4'), ('N', '<i4'), ('N_nan', '<i4'),
                 ('Sum', '<f8'), ('Min', '<f8'), ('Mean', '<f8'),
                 ('Med', '<f8'), ('Max', '<f8'), ('Std', '<f8'),
                 ('Var', '<f8')])

    """
    fname = "/".join(script.split("/")[:-1]) + "/Data/Arr_31_100_150.npy"
    stack = np.load(fname)
    return stack


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
#    stack = _demo_stack()
