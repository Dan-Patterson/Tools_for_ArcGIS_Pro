# -*- coding: UTF-8 -*-
"""
Script :   dist_stats.py

Author :   Dan_Patterson@carleton.ca

Modified : 2018-07-14

Purpose:
-------
  Calculate standard distance and distance matrix for points grouped by an
  attribute/key field.

"""
# ---- imports, formats, constants ----
#

import sys
import numpy as np
import arcpy
from arcpytools_pnt import fc_info, tweet, make_row_format, _col_format, form_
from textwrap import dedent

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.1f}'.format}
np.set_printoptions(edgeitems=10, linewidth=100, precision=2,
                    suppress=True, threshold=120,
                    formatter=ft)
np.ma.masked_print_option.set_display('-')

script = sys.argv[0]


def dist_arr(a):
    """Minimum spanning tree prep... see main header
    : paths from given data set...
    """
    # idx = np.lexsort(a.T)  # sort y, then x
    idx = np.lexsort((a[:, 1], a[:, 0]))  # sort X, then Y
    # idx= np.lexsort((a[:,0], a[:,1]))  # sort Y, then X
    a_srt = a[idx, :]
    d = _e_dist(a_srt)
    frmt = """\n    {}\n    :Input array...\n    {}\n\n    :Sorted array...
    {}\n\n    :Distance...\n    {}
    """
    args = [dist_arr.__doc__, a, a_srt, d]  # d.astype('int')]
    print(dedent(frmt).format(*args))
    return idx, a_srt, d


def _e_dist(a):
    """Return a 2D square-form euclidean distance matrix.
    For other dimensions, use e_dist in arraytools, geom.py
    """
    b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
    diff = a - b
    d = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff)).squeeze()
    return d


def process(in_fc, id_fld, prn=True):
    """process the data for the tool or demo
    """
    frmt = """
    Group .... {}
    points in group {}
    center ... x = {:<8.2f} y = {:<10.2f}
    minimum .. x = {:<8.2f} y = {:<10.2f}
    maximum .. x = {:<8.2f} y = {:<10.2f}
    standard distance ... {:8.2f}
    distance stats ....
      mean:{:8.2f}  min:{:8.2f}  max:{:8.2f}  std:{:8.2f}
    """
    flds = ['SHAPE@X', 'SHAPE@Y', id_fld]
    a_in = arcpy.da.FeatureClassToNumPyArray(in_fc, flds)
    a_sort = np.sort(a_in, order=id_fld)
    a_split = np.split(a_sort, np.where(np.diff(a_sort[id_fld]))[0] + 1)
    msg = ""
    tbl = []
    for i in range(len(a_split)):
        a0 = a_split[i][['SHAPE@X', 'SHAPE@Y']]
        a = a0.copy()
        a = a.view((a.dtype[0], len(a.dtype.names)))  # art.geom._view_(a0)
        cent = np.mean(a, axis=0)
        min_ = np.min(a, axis=0)
        max_ = np.max(a, axis=0)
        var_x = np.var(a[:, 0])
        var_y = np.var(a[:, 1])
        stand_dist = np.sqrt(var_x + var_y)
        dm = _e_dist(a)
        dm_result = np.tril(dm, -1)
        vals = dm_result[np.nonzero(dm_result)]
        stats = [vals.mean(), vals.min(), vals.max(), vals.std()]
        # hdr = "Distance matrix...({}) ".format(i)
        # m = form_(dm_result, deci=1, wdth=80, title=hdr, prn=False)
        args = (i, len(a), *cent, *min_, *max_, stand_dist, *stats) #, m]
        tbl.append(args)
        msg += dedent(frmt).format(*args)
    if prn:
        tweet(msg)
    flds = ['ID', 'N_pnts', 'CentX', 'CentY', 'MinX', 'MinY', 'MaxX', 'MaxY',
             'Stand Dist', 'Mean_dist', 'Min_dist', 'Max_dist', 'Std_dist']
    dts = ['<i4', '<i4', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8',
           '<f8', '<f8', '<f8', '<f8']
    tbl = np.array(tbl)
    tbl = np.core.records.fromarrays(tbl.transpose(),
                                     names=flds,
                                     formats=dts)
    return a_in, a_split, msg, tbl


# ---- main section ----
#
if len(sys.argv) == 1:
    testing = True
    fc = "/Point_tools.gdb/pnts_in_mesh_Intersect"
    flder = "/".join(script.split("/")[:-2])
    in_fc = flder + fc
    id_fld = 'ID_poly'

else:
    testing = False
    in_fc = sys.argv[1]  # point layer
    id_fld = sys.argv[2]
    shp_fld, oid_fld, shp_type, SR = fc_info(in_fc)

a, a_split, msg, tbl = process(in_fc, id_fld)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    """Main section...   """
#    print("Script... {}".format(script))
#    _demo()
