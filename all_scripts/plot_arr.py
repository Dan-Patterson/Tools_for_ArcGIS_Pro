# -*- coding: UTF-8 -*-
"""
:Script:   plot_arr.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2018-08-18
:Purpose:  To plot a 3D array as a graph with each dimension appearing
:  row-wise, instead of down a column.  This produces a side-by-side
:  comparison of the data.
:
:Functions:
: - plot_grid
:Notes:
:  plt is matplotlib.pyplot from import
:  fig = plt.gcf()        # get the current figure
:  fig.get_size_inches()  # find out its size in inches
:     array([ 9.225,  6.400])
:  fig.bbox_inches
:     Bbox('array([[ 0.000,  0.000],\n       [ 9.225,  6.400]])')
:References:
:----------
:  https://matplotlib.org/2.0.0/api/figure_api.html
:
"""
# ---- imports, formats, constants ----
import sys
import numpy as np
from textwrap import dedent

import matplotlib.pyplot as plt
import matplotlib.cm as cm
# import matplotlib.colors as mc

# local import

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float_kind': '{: 0.2f}'.format}
np.set_printoptions(edgeitems=3, linewidth=80, precision=2,
                    suppress=True, threshold=20,
                    formatter=ft)

script = sys.argv[0]


# ---- functions -------------------------------------------------------------
#
def plot_grid(a):
    """Emulate a 3D array plotting each dimension sequentially.
    :Requires:
    :---------
    : a - an ndarray.  if a.ndim < 3, an extra dimension is prepended to it.
    :     The min and max are determined from the input data values.
    :     This is used later on for interpolation.
    :Returns:
    :  A plot of the array dimensions.
    """
    def plot_3d(rows=1, cols=1):
        """Set the parameters of the 3D array for plotting"""
        x = np.arange(cols)
        y = np.arange(rows)
        xg = np.arange(-0.5, cols + 0.5, 1)
        yg = np.arange(-0.5, rows + 0.5, 1)
        return [x, y, xg, yg]
    #
    a = a.squeeze()
    if a.ndim < 3:
        frmt = "1D and 2D arrays not supported, read the docs\n{}"
        print(dedent(frmt).format(plot_grid.__doc__))
        return None
    # proceed with 3D array
    n, rows, cols = a.shape
    x, y, xg, yg = plot_3d(rows, cols)
    w = (cols*n)//2
    h = (rows + 1)//2
    w = max(w, h)
    h = min(w, h)
    fig, axes = plt.subplots(1, n, sharex=True, sharey=True,
                             dpi=150, figsize=(w, h))
    fig.set_tight_layout(True)
    fig.set_edgecolor('w')
    fig.set_facecolor('w')
    idx = 0
    for ax in axes:
        m_min = a.min()
        m_max = a.max()
        a_s = a[idx]
        col_lbl = "Cols: for " + str(idx)
        ax.set_aspect('equal')
        ax.set_adjustable('box')  # box-forced')  # deprecated prevents spaces
        ax.set_xticks(xg, minor=True)
        ax.set_yticks(yg, minor=True)
        ax.set_xlabel(col_lbl, labelpad=12)
        ax.xaxis.label_position = 'top'
        ax.xaxis.label.set_fontsize(12)
        if idx == 0:
            ax.set_ylabel("Rows", labelpad=2)  # was 12
            ax.yaxis.label.set_fontsize(12)
        ax.grid(which='minor', axis='x', linewidth=1, linestyle='-', color='k')
        ax.grid(which='minor', axis='y', linewidth=1, linestyle='-', color='k')
        t = [[x, y, a_s[y, x]]
             for y in range(rows)
             for x in range(cols)]
        for i, (x_val, y_val, c) in enumerate(t):
            ax.text(x_val, y_val, c, va='center', ha='center', fontsize=12)
        ax.matshow(a[idx], cmap=cm.gray_r, interpolation='nearest',
                   vmin=m_min, vmax=m_max, alpha=0.2)
        idx += 1
    # ---- end of script ----------------------------------------------------


def _plt_(a):
    """
    :one array shows numbers, the alternate text
    """
    plot_grid(a)
    print("array... shape {} ndim {}\n{}".format(a.shape, a.ndim, a))


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    """   """
#    print("Script... {}".format(script))
    d, r, c = [3, 8, 4]
    a = np.arange(d*r*c).reshape(d, r, c)
    b = a * 1.0
    c = np.random.randint(96, size=96).reshape(d, r, c)
    c1 = c * 1.0
    d = np.arange(2*3*3*4).reshape(2, 3, 3, 4)
    e = np.arange(4*5).reshape(4, 5)
#    _plt_(a)
