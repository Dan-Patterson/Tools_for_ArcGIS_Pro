# -*- coding: UTF-8 -*-
"""
:Script:   spaced.py
:Author:   Dan_Patterson@carleton.ca
:Modified: 2017-04-11
:Purpose:  tools for working with numpy arrays
:
:Original sources:
:----------------
:n_spaced :  ...\arraytools\geom\n_spaced.py
: - n_spaced(L=0, B=0, R=10, T=10, min_space=1, num=10, verbose=True)
:   Produce num points within the bounds specified by the extent (L,B,R,T)
:   L(eft), B, R, T(op) - extent coordinates
:   min_space - minimum spacing between points.
:   num - number of points... this value may not be reached if the extent
:   is too small and the spacing is large relative to it.
:
:arr_struct :  ...\arcpytools.py
: - array_struct(a, fld_names=['X', 'Y'], dt=['<f8', '<f8']):
:   Convert an array to a structured array
:   a - an ndarray with shape at least (N,2)
:   dt = dtype class
:   names - names for the fields
:
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
from textwrap import dedent
import numpy as np
# import arcpy
from arcpytools import array_fc, array_struct, tweet

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


# ---------------------------------------------------------------------------
# ---- from arraytools.geom ----
def n_spaced(L=0, B=0, R=10, T=10, min_space=1, num=10, verbose=True):
    """Produce num points within the bounds specified by the extent (L,B,R,T)
    :Requires:
    :--------
    :  L(eft), B, R, T(op) - extent coordinates
    :  min_space - minimum spacing between points.
    :  num - number of points... this value may not be reached if the extent
    :        is too small and the spacing is large relative to it.
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


# ---- main section ---------------------------------------------------------
#
aoi = sys.argv[1]  # '340000 5020000 344999.999999999 5025000 NaN NaN NaN NaN'
min_space = int(sys.argv[2])
num = int(sys.argv[3])
SR = sys.argv[4]
out_fc = sys.argv[5]

frmt = """\n
AOI extent for points...
{}
Minimum spacing.... {}
Number of points... {}
Spatial reference.. {}
Output featureclass.. {}\n
"""
args = [aoi, min_space, num, SR, out_fc]
msg = frmt.format(*args)
tweet(msg)
# ---- perform the point creation ----
aoi = aoi.split(" ")[:4]             # extent is returned as a string
ext = [round(float(i)) for i in aoi]
L, B, R, T = ext
a = n_spaced(L, B, R, T, min_space, num, verbose=False)
all_flds = ['X', 'Y', 'x_coord', 'y_coord']
xy_flds = all_flds[:2]
xy_dt = ['<f8', '<f8', 'float', 'float']
a = np.c_[(a, a)]
z = array_struct(a, fld_names=all_flds, dt=xy_dt)
# z = np.zeros((len(a)), dtype=[('X', '<f8'), ('Y', '<f8')])
# fld_names = ('X', 'Y')
# z['X'] = a[:, 0]
# z['Y'] = a[:, 1]
out_fc = array_fc(z, out_fc, xy_flds, SR)

# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
    pass
