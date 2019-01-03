# -*- coding: UTF-8 -*-
"""
image
=====

Script :   image.py

Author :   Dan.Patterson@carleton.ca

Modified : 2019-01-02

Purpose :  tools for working with numpy arrays as images

Useage:

Functions:
---------_
a_filter(a, mode=1, ignore_ndata=True)  # mode is a 3x3 filter

References:
:
:---------------------------------------------------------------------:
"""
# pylint: disable=C0103
# pylint: disable=R1710
# pylint: disable=R0914

# ---- imports, formats, constants ----
import sys
import warnings
import numpy as np
from arraytools.tools import stride

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=5, linewidth=80, precision=2,
                    suppress=True, threshold=500,
                    formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

warnings.filterwarnings('ignore')

__all__ = ['_even_odd',
           '_pad_even_odd', '_pad_nan', '_pad_zero',
           'a_filter',
           'plot_img',
           'rgb_gray', 'normalize', 'equalize']


# ----------------------------------------------------------------------
# (1) padding arrays
def _even_odd(a):
    """Even/odd from modulus.  Returns 0 for even, 1 for odd"""
    prod = np.cumprod(a.shape)[0]
    return np.mod(prod, 2)


def _pad_even_odd(a):
    """To use when padding a strided array for window construction
    """
    p = _even_odd(a)
    ap = np.pad(a, pad_width=(1, p), mode="constant", constant_values=(0, 0))
    return ap


def _pad_nan(a, nan_edge=True):
    """Pad a sliding array to allow for stats, padding uses np.nan
    : see also: num_to_nan(a, num=None, copy=True)
    """
    a = a.astype('float64')
    if nan_edge:
        cv = (np.NaN, np.NaN)
        a = np.pad(a, pad_width=(1, 1), mode="constant", constant_values=cv)
    return a


def _pad_zero(a, n=1):
    """To use when padding a strided array for window construction. n = number
    : of zeros to pad arround the array
    : see also: nun_to_nan (1.13)
    """
    ap = np.pad(a, pad_width=(n, n), mode="constant", constant_values=(0, 0))
    return ap


# ----------------------------------------------------------------------
# (1) filter array ---- convolution filters
#
def a_filter(a, mode=1, pad_output=True, ignore_nodata=True, nodata=None):
    """Various filters applied to an array.

    Requires:
    --------
    stride : function
        the `stride` function is used internally in this function
    a : array
        an array that will be strided using a 3x3 window
    pad_output : boolean
        True, produces a masked array padded so that the shape
        is the same as the input
    ignore_nodata : boolean
        True, all values used, False, array contains nodata
    nodata : None or number
        None :
            max int or float used
        value :
            use this value in integer or float form otherwise


    mode :
        a dictionary containing a choice from
    ::

        1.  `all_f`    : all 1's
        2.  `no_cnt`   : no center
        3.  `cross_f`  : cross filter, corners
        4.  `plus_f`   : up down, left right
        5.  `gradient` : `6 7 8 9 10` directional gradients
        11. `lap_33`   : laplacian
        12. `line_h`   : line detection, horizonal
        13. `line_ld`  : line detection, left diagonal
        14. `line_rd`  : line detection, right diagonal
        15. `line_v`   : line detection, vertical
        16. `high`     :
        17. `sob_hor`  : Sobel horizontal
        18. `sob_vert` : Sobel vertical
        19. `emboss`   :
        20. `sharp1`   : sharpen 1
        22. `sharp2`   : sharpen 2
        23. `sharp3`   : sharpen 3 highpass 3x3
        24. `lowpass`  : lowpass filter

    Notes:
    -----
        Only 3x3 filters covered here.  The output array is padded with np.nan
        and the array is returned as a masked array.

    >>> a0 = pyramid(core=4, steps=5, incr=(1, 1))
    >>> a0 = a0 * 2  # multiply by a number to increase slope
    >>> a0 = (pyramid(core=4, steps=5, incr=(1, 1)) + 1) * 2  # is also good!

    References:
    ----------
    ..
    [1]
    http://desktop.arcgis.com/en/arcmap/latest/tools/spatial-analyst-toolbox/how-filter-works.htm

    [2]
    http://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/convolution-function.htm

    [3]
    https://github.com/scikit-image/scikit-image/tree/master/skimage/filters
    """
    n = np.nan
    all_f = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    no_cnt = [1, 1, 1, 1, n, 1, 1, 1, 1]
    cross_f = [1, 0, 1, 0, 0, 0, 1, 0, 1]
    plus_f = [0, 1, 0, 1, 0, 1, 0, 1, 0]
    grad_e = [1, 0, -1, 2, 0, -2, 1, 0, -1]
    grad_n = [-1, -2, -1, 0, 0, 0, 1, 2, 1]
    grad_ne = [0, -1, -2, 1, 0, -1, 2, 1, 0]
    grad_nw = [2, -1, 0, -1, 0, 1, 0, 1, 2]
    grad_s = [1, 2, 1, 0, 0, 0, -1, -2, -1]
    grad_w = [-1, 0, 1, -2, 0, 2, -1, 0, 1]
    lap_33 = [0, -1, 0, -1, 4, -1, 0, -1, 0]
    line_h = [-1, -1, -1, 2, 2, 2, -1, -1, -1]
    line_ld = [2, -1, -1, -1, 2, -1, -1, -1, 2]
    line_rd = [-1, -1, 2, -1, 2, -1, 2, -1, -1]
    line_v = [-1, 0, -1, -1, 2, -1, -1, 2, -1]
    high = [-0.7, -1.0, -0.7, -1.0, 6.8, -1.0, -0.7, -1.0, -0.7]  # arc
    sob_hor = [1, 2, 1, 0, 0, 0, -1, -2, -1]   # sobel y  /4.0 weights
    sob_vert = [1, 0, -1, 2, 0, -2, 1, 0, -1]  # sobel x  /4
    emboss = [-1, -1, 0, -1, 0, 1, 0, 1, 1]
    sharp1 = [0., -0.25, 0., -0.25, 2.0, -0.25, 0., -0.25, 0.]  # arc
    sharp2 = [-0.25, -0.25, -0.25, -0.25, 3.0, -0.25, -0.25, -0.25, -0.25]
    sharp3 = [-1, -1, -1, -1, 9, -1, -1, -1, -1]  # arc
    lowpass = [1, 2, 1, 2, 4, 2, 1, 2, 1]  # arc
    # ---- assemble the dictionary ----
    d = {1: all_f, 2: no_cnt, 3: cross_f, 4: plus_f, 5: grad_e,
         6: grad_n, 7: grad_ne, 8: grad_nw, 9: grad_s, 10: grad_w,
         11: lap_33, 12: line_h, 13: line_ld, 14: line_rd, 15: line_v,
         16: high, 17: sob_hor, 18: sob_vert, 19: emboss, 20: sharp1,
         21: sharp2, 23: sharp3, 24: lowpass}
    filter_ = np.array(d[mode]).reshape(3, 3)
    # ---- stride the input array ----
    a_strided = stride(a)
    if ignore_nodata:
        c = np.sum(a_strided * filter_, axis=(2, 3))
    else:
        c = np.nansum(a_strided * filter_, axis=(2, 3))
    if pad_output:
        pad_ = nodata
        if nodata is None:
            if c.dtype.name in ('int', 'int32', 'int64'):
                pad_ = min([0, -1, a.min()-1])
            else:
                pad_ = min([0.0, -1.0, a.min()-1])
        c = np.lib.pad(c, (1, 1), "constant", constant_values=(pad_, pad_))
        m = np.where(c == pad_, 1, 0)
        c = np.ma.array(c, mask=m, fill_value=None)
    return c


def plot_img(img):
    """plot image as gray scale"""
    import matplotlib.pyplot as plt
    plt.imshow(img, cmap=plt.get_cmap('gray'))
#    plt.show()


def rgb_gray(a):
    """Convert 3d array to 2d gray scale"""
    shp = a.shape
    if shp[2] == 3:
        r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    else:
        r, g, b = a
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def normalize(a):
    """Normalize array to unsigned integer in range 0-255"""
    lmin = a.min()
    lmax = a.max()
    nor = np.floor((a - lmin) / (lmax - lmin)*255.)
    nor = nor.astype('uint8')
    return nor


def equalize(a):
    """Produce a raster equalization for a 2D array.
    :Notes:
    :------
    : - derive the histogram, produce the ogive, then normalize it.
    https://en.wikipedia.org/wiki/Histogram_equalization
    """
    hist, _ = np.histogram(a.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min())*255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    img2 = cdf[a]
    return img2


def _demo():
    """
    :------------------------------------------------------------------
    """
    rows, cols = (4, 4)
    a = np.arange(rows*cols, dtype='int32').reshape(rows, cols)
    x, y = (4, 4)
    a = np.tile(a.repeat(x), y)
    a = np.hsplit(a, y)         # split into y parts horizontally
    a = np.vstack(a)
    a = np.hsplit(a, a.shape[0])
    a = np.vstack(a).copy(order='C')
    print("array 'a'\n{}".format(a))
    print("equalized 'a'\n{}".format(equalize(a)))
    return a


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
#    a = _demo()
