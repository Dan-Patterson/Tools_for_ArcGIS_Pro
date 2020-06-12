# -*- coding: utf-8 -*-
"""
triangulate
===========

Script :   triangulate.py

Author :   Dan_Patterson@carleton.ca

Modified : 2018-08-24

Purpose:  triangulate poly* features

Useage :

References
----------
`<http://pro.arcgis.com/en/pro-app/arcpy/data-access/numpyarraytotable.htm>`_.
`<http://pro.arcgis.com/en/pro-app/arcpy/data-access/tabletonumpyarray.htm>`_.
`<https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/
scipy.spatial.Delaunay.html>`_.
---------------------------------------------------------------------
"""

import sys
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured as stu
from scipy.spatial import Delaunay, Voronoi
from arcpytools_plt import fc_info #, tweet  #, frmt_rec, _col_format
import arcpy

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

arcpy.env.overwriteOutput = True

# ----
#
def _xyID(in_fc, to_pnts=True):
    """Convert featureclass geometry (in_fc) to a simple 2D structured array
    :  with ID, X, Y values. Optionally convert to points, otherwise centroid.
    """
    flds = ['OID@', 'SHAPE@X', 'SHAPE@Y']
    args = [in_fc, flds, None, None, to_pnts, (None, None)]
    cur = arcpy.da.SearchCursor(*args)
    a = cur._as_narray()
    a.dtype = [('IDs', '<i4'), ('Xs', '<f8'), ('Ys', '<f8')]
    del cur
    return a


def poly(pnt_groups, SR):
    """Short form polygon creation
    """
    polygons = []
    for pnts in pnt_groups:
        for pair in pnts:
            arr = arcpy.Array([arcpy.Point(*xy) for xy in pair])
            pl = arcpy.Polygon(arr, SR)
            polygons.append(pl)
    return polygons


def vor_pnts(pnts, ri_type="Voronoi", testing=False):
    """not used with polygons"""
    out = []
    for ps in pnts:
        avg = np.mean(ps, axis=0)
        p =  ps - avg
        tri = Voronoi(p)
        for region in tri.regions:
            if not -1 in region:
                polygon = np.array([tri.vertices[i] + avg for i in region])
                out.append(polygon)
                if testing:
                    print("{}".format(polygon.T))
    #ts = [i for i in t if i.ndim == 2]
    return out


def tri_pnts(pnts, testing=False):
    """Triangulate the points and return the triangles

    Parameters:
    -----------
    pnts : np.array
        Points in array format.
    out : array
        an array of triangle points

    Notes:
    ------
    >>> pnts = pnts.reshape((1,) + pnts.shape)  # a 3D set of points (ndim=3)
    >>> [pnts]  # or pass in as a list
    """
    out = []
    for ps in pnts:
        ps = np.unique(ps, axis=0)  # get the unique points only
        avg = np.mean(ps, axis=0)
        p =  ps - avg
        tri = Delaunay(p)
        simps = tri.simplices
        new_pnts = [p[s]+avg for s in simps]
        if testing:
            print("{}".format(new_pnts))
        out.append(new_pnts)
    return out


def pnt_groups(in_fc):
    """Simple def to convert shapes to points from a featureclass
    """
    shp_fld, oid_fld, shp_type, SR = fc_info(in_fc)
    flds = ['OID@', 'SHAPE@X', 'SHAPE@Y']
    args = [in_fc, flds, None, None, True, (None, None)]
    cur = arcpy.da.SearchCursor(*args)
    a = cur._as_narray()
    a.dtype = [('IDs', '<i4'), ('Xs', '<f8'), ('Ys', '<f8')]
    del cur
    pts = []
    keys = np.unique(a['IDs'])
    for k in keys:
        w = np.where(a['IDs'] == k)[0]
        z = a[['Xs', 'Ys']][w[0]:w[-1] + 1]
        z = stu(z)
        #z = np.copy(z.view(np.float64).reshape(z.shape[0], 2))
        pts.append(z)
    return pts, a, SR

# ---- Do the work
#
if len(sys.argv) == 1:
    testing = True
    in_pth = script.split("/")[:-2] + ["Polygon_lineTools.gdb"]
    in_fc = "/".join(in_pth) + "/shapes_mtm9" #/Densified_4x"
    out_fc = "/".join(in_pth) + "/v3"
#    keep_flds = "*"
#    shp_fld, oid_fld, shp_type, SR = fc_info(in_fc)
#    pts, a, SR = pnt_groups(in_fc)
#    t = tri_pnts(pts, True)
#    polys = poly(t, SR)
#    arcpy.CopyFeatures_management(polys, "in_memory/temp")
#    arcpy.analysis.Clip("in_memory/temp", in_fc, out_fc, None)
else:
    testing = False
    in_fc = sys.argv[1]
    out_fc = sys.argv[2]

# finish up
#
#keep_flds = "*"
#shp_fld, oid_fld, shp_type, SR = fc_info(in_fc)
if not testing:
    pts, a, SR = pnt_groups(in_fc)
    t = tri_pnts(pts, True)
    polys = poly(t, SR)
    if arcpy.Exists(out_fc):
        arcpy.Delete_management(out_fc)
    arcpy.CopyFeatures_management(polys, "in_memory/temp")
    arcpy.MakeFeatureLayer_management("in_memory/temp", "temp")
    arcpy.management.SelectLayerByLocation("temp",
                                           "WITHIN_CLEMENTINI",
                                           in_fc, None,
                                           "NEW_SELECTION", "NOT_INVERT")
    arcpy.CopyFeatures_management("temp", out_fc)

    #arcpy.analysis.Clip("in_memory/temp", in_fc, out_fc, None)
    #
    #arcpy.analysis.Intersect(["in_memory/temp",in_fc], out_fc,
    #                         "ONLY_FID", None, "INPUT")


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
