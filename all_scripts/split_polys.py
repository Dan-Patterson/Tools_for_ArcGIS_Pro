# -*- coding: UTF-8 -*-
"""
script
======

Script :   split_polys.py

Author :   Dan_Patterson@carleton.ca

Modified : 2018-06-14

Purpose :  tools for working with numpy arrays

Useage :

References
----------

`<https://stackoverflow.com/questions/3252194/numpy-and-line-intersections>`_.

"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

# ---- imports, formats, constants ----
import sys
from textwrap import dedent
import numpy as np
from arraytools.fc_tools._common import * # fc_info
from arraytools.geom import e_area, _extent
from arraytools.fc_tools.apt import arc_np, _id_geom_array, output_polygons
import arcpy


ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}

np.set_printoptions(edgeitems=5, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

# ---- Split poly* geometry using several approaches ----
#
# -- helper functions

def _clip_ax(a, x_axis=True, ax_min=0, ax_max=1):
    """Clip geometry x or y coordinates and return the result
    a : array
        Input array.  If polygon geometry, then coordinates are ordered
        clockwise for outer rings with the first and last points identical.
    x_axis : boolean
        True for x-axis, False for y-axis
    ax_min, ax_max : number
        Number for the x or y coordinate minimum and maximum
    """
    b = np.zeros_like(a)
    if x_axis:
        b[:, 1] = a[:, 1]
        b[:, 0] = np.clip(a[:, 0], ax_min, ax_max)
    else:
        b[:, 0] = a[:, 0]
        b[:, 1] = np.clip(a[:, 1], ax_min, ax_max)
    return b


def _uni_pnts(a):
    """Return unique points forming a *poly feature
    """
    _, idx = np.unique(a, return_index=True, axis=0)
    a = np.concatenate((a[np.sort(idx)], [a[0]]))
    return a

# ---- Split extent by a factor ----
#
def split_ext(a, divisor=2, x_axis=True):
    """Split the extent of a feature by a factor/percentage
    a : array
        Input array.
    divisor : factor
        The split factor representing the number of divisions to make over the
        range of the geometry.
    x_axis : boolean
        Split the X-axis if True, otherwise the y_axis
    """
    #
    L, B, R, T = _extent(a)
    keep = []
    if x_axis:
        fac = np.linspace(L, R, num=divisor+1, endpoint=True)
        f = fac[:-1]
        t = fac[1:]
        for i in range(divisor):
            k = _clip_ax(a, x_axis=True, ax_min=f[i], ax_max=t[i])
            k = _uni_pnts(k)
            keep.append(k)
    else:
        fac = np.linspace(B, T, num=divisor+1, endpoint=True)
        f = fac[:-1]
        t = fac[1:]
        for i in range(divisor):
            k = _clip_ax(a, x_axis=False, ax_min=f[i], ax_max=t[i])
            k = _uni_pnts(k)
            keep.append(k)
    return keep


# ---- Split by area ----
#
def split_area(a, pieces=4, step_size=10, tol=1.0, x_axis=True):
    """Split the extent of a feature by a factor/percentage
    a : array
        Input array.
    pieces : number
        Number of pieces to split the poly* into
    step : factor
        The step is the distance to move in planar units.
    tol : number
        The percentage tolerance in the area
    x_axis : boolean
        Split the X-axis if True, otherwise the y_axis
    """
    xs = a[:, 0]
    uni = np.unique(xs)
    ax_min, ax_max = uni[0], uni[-1]
    steps, incr = np.linspace(ax_min, ax_max, num=pieces+1,
                              endpoint=True, retstep=True)
    # ---- Do the work, clip the axes, get the unique points, calculate area
    arrs = []
    areas = []
    for i in range(1, len(steps)):
        sub = _clip_ax(a, True, steps[i-1], steps[i])  # ---- clip_ax
        sub = _uni_pnts(sub)             # ---- uni_pnts
        areas.append(e_area(sub))        # ---- e_area
        arrs.append(sub)
    tot_area = sum(areas)
    cum_area = np.cumsum(areas)
    area = tot_area/float(pieces)  # required area
    bins = np.arange(1, pieces+1) * area
    inds = np.digitize(cum_area, bins)
    f = np.where(inds[:-1]-inds[1:] != 0)[0]
    t = f + 2
#    L, B, R, T = _extent(a)
#    keep = []
#    tot_area = e_area(a)
#    area = tot_area/float(pieces)  # required area
#    cal = 0.0
#    tol = area*tol/100.
#    check = np.abs(area-cal)
#    if x_axis:
#        n = 0
#        step=step_size
#        right = L + step
#        while (check > tol) and (n < 20):
#            k = clip_ax(a, x_axis=True, ax_min=L, ax_max=right)
#            cal = e_area(k)
#            #print(n, cal)
#            check = area-cal
#            print("area {} check {}  right {}".format(cal, check < tol, step))
#            if check > 0.:
#                right += step_size
#            else:
#                step_size /= 2.
#                right -= step_size
#                k = clip_ax(a, x_axis=True, ax_min=L, ax_max=right)
#                cal = e_area(k)
#                check = np.abs(area-cal)
#                print("....area-cal {} ".format(cal))
##            step += step_size
#            n += 1

#    else:
#        fac = np.linspace(B, T, num=divisor+1, endpoint=True)
#        f = fac[:-1]
#        t = fac[1:]
#        for i in range(divisor):
#            k = clip_ax(a, x_axis=False, ax_min=f[i], ax_max=t[i])
#            _, idx = np.unique(k, return_index=True, axis=0)
#            k = np.concatenate((k[np.sort(idx)], [k[0]]))
#            keep.append(k)
    return arrs, areas, f, t


def perp(a):
    """Perpendicular to array"""
    b = np.empty_like(a)
    b_dim = b.ndim
    if b_dim == 1:
        b[0] = -a[1]
        b[1] = a[0]
    elif b_dim == 2:
        b[:, 0] = -a[:, 1]
        b[:, 1] = a[:, 0]
    return b


def seg_int(a, v):
    """Returns the point of intersection of the line segments passing through
    a1, a0 and v1, v0.

    a0, a1 : points
        [x, y] pairs for each line segment representing the start and end
    v0, v1 : points
        [x, y] pairs on the intersecting line
    **** vertical line *****
    v = np.array([[L + delta, B], [L + delta, T]]) # just need the extent
    """
    a0 = a[:-1]  # start points
    a1 = a[1:]   # end points
    b0, b1 = v   # start and end of the intersecting line
    b_ = b0[0]
    ox = a0[:, 0]
    dx = a1[:, 0]
#    f_t = np.array(list(zip(xs[:-1], xs[1:])))
#    f_t = np.sort(f_t, axis=1)
    idx0 = np.where((ox <= b_) & (b_ <= dx))[0]  # incrementing x's
    idx1 = np.where((ox >= b_) & (b_ >= dx))[0]  # decreasing x's
    idx_s = np.concatenate((idx0, idx1))
    # ---- alternate
    da = a1 - a0
    db = b1 - b0
    dp = a0 - b0
    dap = perp(da)
    denom = np.dot(dap, db)     # or dap @ db
    # num = np.dot(dap, dp )
    db2 = db.reshape(1, 2)
    denom = np.einsum('ij,ij->i', dap, db2)
    num = np.einsum('ij,ij->i', dap, dp)
    int_pnts = (num/denom).reshape(num.shape[0], 1) * db + b0
    ft_int = np.hstack((a0, a1, int_pnts))
    return int_pnts, ft_int


def _demo():
    """
    Notes:
    -----

    The xs and ys form pairs with the first and last points being identical
    The pairs are constructed using n-1 to ensure that you don't form a
    line from identical points.
    #
    first split polygon as a sample of a multipart.  I had to add 0,0 and 0, 80
    back in
    l = [[[ 0., 0.], [ 0., 30.], [ 10., 30.], [ 10., 0.], [ 0.,  0.]],
         [[ 0., 80.], [ 0., 100.], [ 10., 100.], [ 10., 73.75], [ 0., 80.]]]
    """
#    xs = [10., 20., 20., 0., 40., 70., 80., 80., 60., 40., 10.]
#    ys = [0., 30., 50., 70., 70., 60., 30., 0., 10., 10., 0.]
#    xs = [0., 0., 100., 100., 0.]  # simple square
#    ys = [0., 100., 100., 0., 0.]

    xs = [0., 0., 80., 0, 0., 100., 100., 0.]
    ys = [0., 30., 30., 80., 100., 100., 0., 0.]
    a = np.array(list(zip(xs, ys))) * 1.0  # --- must be floats
    v = np.array([[50., 0], [50, 100.]])
    ext = np.array([[0., 0], [0, 100.], [100, 100.], [100., 0.], [0., 0.]])

#    return a, v

#    fc = r"C:\Temp\junk.gdb\smp"
#
#    a0 = arc_np(fc)
##    b = _id_geom_array(fc)
#    a = (a0[['Xs', 'Ys']].view(dtype='float64')).reshape(a0.shape[0], 2)
    arrs, areas, f, t = split_area(a, 5)
    out = np.array(arrs)
    out_fc = r"C:\Temp\junk.gdb\s"
    SR = 'NAD_1983_2011_StatePlane_Mississippi_East_FIPS_2301_Ft_US'
    polygons = []
    frmt = """
    area {}
    extent ll {}
    extent ur {}
    width {}
    height {}
    centroid {}
    """
    for pair in out:
        print("pair ...\n{}".format(pair))
        pnts = [arcpy.Point(*xy) for xy in pair]
        print("points ...\n{}".format(pnts))
        pl = arcpy.Polygon(arcpy.Array(pnts), SR)
        args = [pl.area, pl.extent.lowerLeft, pl.extent.upperRight,
                pl.extent.width, pl.extent.height, pl.centroid]
        print(dedent(frmt).format(*args))
        polygons.append(pl)
    return polygons, out
#   arcpy.CopyFeatures_management(polygons, out_fc)
#    output_polygons(out_fc, SR, out)
#    # this works
#    # z = [array_struct(i, fld_names=['X', 'Y'], dt=['<f8', '<f8']) for i in out]
#
#    return a, out, out_fc, SR, polygons

#    L, B, R, T = _extent(a)
#    delta = 50
#    v = np.array([[L + delta, B], [L + delta, T]]) # just need the extent
#
#    area = e_area(a)
#    #
##    arrs = split_ext(a, divisor=2, x_axis=True)
#    args  = split_area(a, pieces=2, step_size=50, tol=0.05, x_axis=True)
#    arrs, areas, f, t = args
#    print("full area {}".format(area))
##    for i in arrs:
##        print(e_area(i))
#    return a, arrs, areas, f, t, v


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
#    polygons, out = _demo()
#    a, arrs, areas, f, t, v = _demo()
#    a, out, out_fc, SR, polygons = _demo()
"""
for i in range(2, len(a)+1):
    print("{}, {}, {}".format(a[i-2], seg_int(a[i-2:i], v), a[i-1]))

[0. 0.], [[-n- inf]], [ 0. 30.]
[ 0. 30.], [[10. 30.]], [80. 30.]
[80. 30.], [[10.   73.75]], [ 0. 80.]
[ 0. 80.], [[-n- inf]], [  0. 100.]
[  0. 100.], [[-n- -n-]], [  0. 100.]
[  0. 100.], [[ 10. 100.]], [100. 100.]
[100. 100.], [[-n- inf]], [100.   0.]
[100.   0.], [[10.  0.]], [0. 0.]
"""

