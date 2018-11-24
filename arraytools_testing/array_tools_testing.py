# -*- coding: utf-8 -*-
"""
array_tools_testing
===================

Script :   array_tools_testing.py

Author :   Dan_Patterson@carleton.ca

Modified : 2018-09-19

Purpose:  tools for working with numpy arrays

Useage :

References
----------
`<http://pro.arcgis.com/en/pro-app/arcpy/data-access/numpyarraytotable.htm>`_.
`<http://pro.arcgis.com/en/pro-app/arcpy/data-access/tabletonumpyarray.htm>`_.
---------------------------------------------------------------------
"""

import sys
import numpy as np
#from arraytools.fc_tools._common import fc_info, fld_info
import arcpy


ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=5, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


def id_geom(in_fc, as_pnts=True):
    """The main code segment which gets the id and shape information and
    explodes the geometry to individual points
    """
    with arcpy.da.SearchCursor(in_fc, ['OBJECTID', 'SHAPE@']) as cursor:
        a = [[row[0], row[1]] for row in cursor]
    return a


def pip_arc(pnt_fc, poly_fc):
    """Naive point-in-polygon using arcpy

    pnt_fc : PointGeometry
        PointGeometry only supports `within` geometry operations
    poly_fc : Polygon
        Polgon geometry

    Notes:
    ------
    `within` options are `BOUNDARY`, `CLEMENTINI`, `PROPER`.
    - `BOUNDARY` on boundary
    - `CLEMENTINI` default, must be within, not on boundary
    """
    pnts = id_geom(pnt_fc)
    polys = id_geom(poly_fc)
    out = []
    for pnt in pnts:
        pid, p = pnt
        for poly in polys:
            plid, pol = poly
            if p.within(pol, "CLEMENTINI"):  # CLEMENTINI, PROPER
                out.append([pid, plid])  #[pid, p, plid, pol])
                break
#            else:
#                continue
    return out  #pnts, polys, out


def extend_tbl(arr, in_fc=None, join_id="PntID", col_id="PolyID"):
    """ExtendTable example
    """
    dt = [(join_id, '<i4'), (col_id, '<i4')]
    z = np.ndarray((len(arr), ), dtype=dt)
    z[join_id] = arr[:, 0]
    z[col_id] = arr[:, 1]
    arcpy.da.ExtendTable(in_fc, "OBJECTID", z, join_id)


def pip_demo():
    """Point in polygon demo
    """
    out = pip_arc(pnt_fc, poly_fc)
    out2 = np.array(out)
    flds = arcpy.ListFields(pnt_fc)
    out_fld = "PolyID"
    fnames = [i.name for i in flds]
    if out_fld in fnames:
        out_fld += "1"
    extend_tbl(out2, pnt_fc, "PntID", out_fld)  # uncomment to test extend
    print("output array\n{}".format(out2))


# ----------------------------------------------------------------------
# .... final code section producing the featureclass and extendtable
if len(sys.argv) == 1:
    testing = True
    pth = script.split("/")[:-1]
    pnt_fc = r"\array_tools.gdb\pnts_2000"  # 1994 of 2000 within clementini
    pnt_fc = "\\".join(pth) + pnt_fc
    poly_fc = r"\array_tools.gdb\SamplingGrids"
    poly_fc = "\\".join(pth) + poly_fc
else:
    testing = False
    # parameters here
#
if testing:
    print('\nScript source... {}'.format(script))
# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
