# -*- coding: UTF-8 -*-
"""
:Script:   mesh_pnts.py
:Author:   Dan_Patterson@carleton.ca
:Modified: 2017-04-11
:Purpose:  Just makes points on a grid as well as the meshgrid
:
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
import numpy as np
import arcpy
from textwrap import dedent
from arcpytools_pnt import array_fc, fc_info, tweet


ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=3, linewidth=80, precision=2, suppress=True,
                    threshold=50, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


def mesh_xy(L=0, B=0, R=5, T=5, dx=1, dy=1, as_rec=True, top_down=True):
    """Create a mesh of coordinates within the specified X, Y ranges
    :Requires:
    :--------
    :  L(eft), R(igh), dx - coordinate min, max and delta x for X axis
    :  B(ott), T(op), dy  - same,  Y axis
    :  as_rec - produce a structured array (or convert to a record array)
    :Returns:
    :-------
    :  A list of coordinates of X,Y pairs and an ID if as_rec is True.
    :  A mesh grid X and Y coordinates is also produced.
    :-------------
    """
    dt = [('Pnt_num', '<i4'),
          ('X', '<f8'), ('Y', '<f8'),
          ('Xs', '<f8'), ('Ys', '<f8'), ('RowCol', 'U12')]
    x = np.arange(L, R + dx, dx, dtype='float64')
    if top_down:
        y = np.arange(T, B-dy, -dy, dtype='float64')
    else:
        y = np.arange(B, T+dy, dy, dtype='float64')
    mesh = np.meshgrid(x, y, sparse=False)
    xs = mesh[0].ravel()
    ys = mesh[1].ravel()
    rc = np.indices(mesh[0].shape)
    rows = rc[0].ravel()
    cols = rc[1].ravel()
    rcs = np.array(list(zip(rows, cols)))
    rclbl = ["r{:03.0f} c{:03.0f}".format(*i) for i in rcs]
    if as_rec:
        p = list(zip(np.arange(len(xs)), xs, ys, xs, ys, rclbl))
        pnts = np.array(p, dtype=dt)
    else:
        p = list(zip(xs, ys, xs, ys))
        pnts = np.array(p)
    return pnts


# ---- main section ----
#
if len(sys.argv) == 1:
    testing = True
    create_output = False
    fc = "/Point_tools.gdb/mesh_bounds"
    flder = "/".join(script.split("/")[:-2])
    extent_fc = flder + fc
    angle = 30.0
    out_fc = flder + "/Point_tools.gdb/mesh_pnts"
    dx = 250.0
    dy = 250.0
    top_down = True
else:
    testing = False
    create_output = True
    extent_fc = sys.argv[1]  # must be a featureclass
    dx = abs(float(sys.argv[2]))
    dy = abs(float(sys.argv[3]))
    top_down = sys.argv[4]
    out_fc = sys.argv[5]


arcpy.env.overwriteOutput = True
shp_fld, oid_fld, shp_type, SR = fc_info(extent_fc)
desc = arcpy.da.Describe(extent_fc)
xtent = (desc['extent'].__str__()).split(" ")[:4]
L, B, R, T = [float(i) for i in xtent]
fld_names = ['X', 'Y']
pnts = mesh_xy(L, B, R, T, dx, dy, as_rec=True, top_down=top_down)
# ---- create output
if create_output:
  array_fc(pnts, out_fc, fld_names, SR)
ln = "-"*70
frmt = """\n
:{}:
:Script... {}
:Output to..... {}
:using ........ {}
:Processing extent specified...
: L {}, B {}, R {}, T {}
:X spacing...{}
:Y spacing...{}
:Points......
{!r:}
:
:{}:"
"""
args = [ln, script, out_fc, SR.name, L, B, R, T, dx, dy, pnts, ln]
msg = frmt.format(*args)
tweet(msg)


# ----------------------------------------------------------------------
# __main__ .... code section

if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
#    pnts, mesh = _demo()
