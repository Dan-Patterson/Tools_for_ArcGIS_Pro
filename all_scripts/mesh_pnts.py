# -*- coding: UTF-8 -*-
"""
mesh_pnts
=========

:Script :   mesh_pnts.py

Author :   Dan_Patterson@carleton.ca

Modified : 2017-03-11

Purpose :  Just makes points on a grid as well as the meshgrid

:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
import numpy as np
from textwrap import dedent


ft = {'bool': lambda x: repr(x.astype('int32')),
      'float_kind': '{: 0.3f}'.format}

np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


def mesh_xy(L=0, B=0, R=5, T=5, dx=1, dy=1, as_rec=True):
    """Create a mesh of coordinates within the specified X, Y ranges

    Requires:
    --------
    L(eft), R(ight), dx : number
        coordinate min, max and delta x for X axis
    B(ott), T(op), dy  : number
        same as above for Y axis
    as_rec : boolean
        Produce a structured array (or convert to a record array)

    Returns:
    -------
    -  A list of coordinates of X,Y pairs and an ID if as_rec is True.
    -  A mesh grid X and Y coordinates is also produced.
    :-------------
    """
    dt = [('Pnt_num', '<i4'), ('X', '<f8'), ('Y', '<f8')]
    x = np.arange(L, R + dx, dx, dtype='float64')
    y = np.arange(B, T + dy, dy, dtype='float64')
    mesh = np.meshgrid(x, y, sparse=False)
    if as_rec:
        xs = mesh[0].ravel()
        ys = mesh[1].ravel()
        p = list(zip(np.arange(len(xs)), xs, ys))
        pnts = np.array(p, dtype=dt)
    else:
        p = list(zip(mesh[0].ravel(), mesh[1].ravel()))
        pnts = np.array(p)
    return pnts, mesh


def _demo():
    """A set of points and mesh using real world projected coordinates"""
    args = [300000, 5025000, 301000, 5026000, 250., 250., True]
    L, B, R, T, dx, dy, as_rec = args
    pnts, mesh = mesh_xy(L, B, R, T, dx, dy, as_rec)
    frmt = """\n
    :Points...
    {!r:}\n
    :Mesh...(Xs)
    {!r:}\n
    :.......(Ys)
    {!r:}
    """
    print(dedent(frmt).format(pnts, mesh[0], mesh[1]))
#    return pnts, mesh


# ----------------------------------------------------------------------
# __main__ .... code section

if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
#    pnts, mesh = _demo()
