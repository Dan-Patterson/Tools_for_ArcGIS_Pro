# -*- coding: UTF-8 -*-
r"""
split_by_area
===========

Script :   split_by_area.py

Author :   Dan_Patterson@carleton.ca

Modified:  2020-05-18

Purpose :  tools for working with numpy arrays

Notes
-----
The xs and ys form pairs with the first and last points being identical
The pairs are constructed using n-1 to ensure that you don't form a
line from identical points.

First split polygon is a sample of a multipart.  Added 0, 0 and 0, 80
back in

>>> xs = [0., 0., 80., 0, 0., 100., 100., 0.]
>>> ys = [0., 30., 30., 80., 100., 100., 0., 0.]
>>> a = np.array(list(zip(xs, ys))) * 1.0  # --- must be floats
>>> v = np.array([[50., 0], [50, 100.]])
>>> ext = np.array([[0., 0], [0, 100.],[100, 100.], [100., 0.], [0., 0.]])
return a, v

References
----------
`<https://stackoverflow.com/questions/3252194/numpy-and-line-intersections>`_.
`<https://community.esri.com/message/627051?commentID=627051#comment-627051>`
`<https://community.esri.com/message/779043-re-how-to-divide-irregular-
polygon-into-equal-areas-using-arcgis-105?commentID=779043#comment-779043>`

This is a good one
`<https://tereshenkov.wordpress.com/2017/09/10/dividing-a-polygon-into-a-given
-number-of-equal-areas-with-arcpy/>`

---------------------------------------------------------------------
"""
# ---- imports, formats, constants ----
import sys
import numpy as np
import warnings
# from arcpytools_plt import (cal_area, get_polys)
import arcpy

warnings.simplefilter('ignore', FutureWarning)

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=5, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


# ---- Do the work or run the demo ------------------------------------------
#

frmt = """
Input features.... {}
Output features... {}
Number of splits . {}
Split types ...... {}
"""


def tweet(msg):
    """Produce a message for both arcpy and python
    : msg - a text message
    """
    m = "{}".format(msg)
    arcpy.AddMessage(m)
    print(m)


def _describe(in_fc=None):
    """Simply return the arcpy.da.Describe object
    : desc.keys() an abbreviated list...
    : [... 'OIDFieldName'... 'areaFieldName', 'baseName'... 'catalogPath',
    :  ... 'dataType'... 'extent', 'featureType', 'fields', 'file'... 'hasM',
    :  'hasOID', 'hasZ', 'indexes'... 'lengthFieldName'... 'name', 'path',
    :  'rasterFieldName', ..., 'shapeFieldName', 'shapeType',
    :  'spatialReference',  ...]
    """
    if in_fc is not None:
        in_fc = r"C:\Git_Dan\npgeom\Project_npg\tests.gdb\sq"
        return arcpy.da.Describe(in_fc)
    else:
        return None


def fc_info(in_fc, prn=False):
    """Return basic featureclass information, including...

    Parameters
    ----------
    - shp_fld  :
        field name which contains the geometry object
    - oid_fld  :
        the object index/id field name
    - SR       :
        spatial reference object (use SR.name to get the name)
    - shp_type :
        shape type (Point, Polyline, Polygon, Multipoint, Multipatch)
    - others   :
        'areaFieldName', 'baseName', 'catalogPath','featureType','fields',
        'hasOID', 'hasM', 'hasZ', 'path'

     - all_flds :
         [i.name for i in desc['fields']]
    """
    desc = _describe(in_fc)
    args = ['shapeFieldName', 'OIDFieldName', 'shapeType', 'spatialReference']
    shp_fld, oid_fld, shp_type, SR = [desc[i] for i in args]
    if prn:
        frmt = "FeatureClass:\n   {}".format(in_fc)
        f = "\n{!s:<16}{!s:<14}{!s:<10}{!s:<10}"
        frmt += f.format(*args)
        frmt += f.format(shp_fld, oid_fld, shp_type, SR.name)
        tweet(frmt)
    else:
        return shp_fld, oid_fld, shp_type, SR


# ---- geometry related -----------------------------------------------------
#
def get_polys(in_fc):
    """Return polygons from a polygon featureclass."""
    out_polys = []
    out_ids = []
    with arcpy.da.SearchCursor(in_fc, ["SHAPE@", "OID@"]) as cursor:
        for row in cursor:
            out_polys.append(row[0])
            out_ids.append(row[1])
    return out_polys, out_ids


def _poly_ext(p):
    """Return poly* extent."""
    L, B = p.extent.lowerLeft.X, p.extent.lowerLeft.Y
    R, T = p.extent.upperRight.X, p.extent.upperRight.Y
    return L, B, R, T


def trans_rot(a, angle):
    """Return a simplified translate and rotate."""
    cent = a.mean(axis=0)
    angle = np.radians(angle)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array(((c, s), (-s, c)))
    return  np.einsum('ij,kj->ik', a-cent, R)+cent


def cal_area(poly, cuts, cutters, factor):
    """Calculate the areas."""
    tot_area = poly.area
    fract_areas = np.array([c.area/tot_area for c in cuts])
    c_sum = np.cumsum(fract_areas)
    f_list = np.linspace(0.0, 1.0, factor, endpoint=False)
    idxs = [np.argwhere(c_sum <= i)[-1][0] for i in f_list[1:]]
    out = []
    for idx in idxs:
        c_line = cutters[idx]
        left, right = poly.cut(c_line)
        out.append([idx, left.area, c_line])
    return out


def _cut_poly(poly, p_id, step=1.0, split_axis="X", split_fac=4, SR=None):
    """Perform the poly* cutting and return the result.

    step : number
        fractional step for division, 1.0 equates to 1%
    split_face : number
        number of areas to produce, 4, means split into 4 equal areas
    """
    L, B, R, T = _poly_ext(poly)
#    s_fac = math.ceil((R - L)/step)
#    lefts = np.linspace(L+dx, R, num=s_fac, endpoint=True)
    dx = step
    dy = step
    if split_axis == "X":
        lefts = np.arange(L+dx, R+dx, dx, dtype='float')
        splitters = np.array([[[l, B-1.0], [l, T+1.0]] for l in lefts])
    elif s_axis == 'Y':
        tops = np.arange(B+dy, T+dy, dy, dtype='float')
        splitters = np.array([[[R+1.0, t], [L-1.0, t]] for t in tops])
    cutters = []
    for s in splitters:
        s = s.tolist()
        c = arcpy.Polyline(arcpy.Array([arcpy.Point(*xy) for xy in s]), SR)
        cutters.append(c)
    # ----
    cuts = []
    for i in cutters:
        rght = poly
        if i.crosses(poly):
            try:
                left, rght = poly.cut(i)
                if rght is None:
                    cuts.append(left)
                cuts.append(left)
                poly = rght
                rght = left
            except RuntimeError:
                tweet("Issues with poly...{}".format(p_id))
                continue
        else:
            cuts.append(rght)
    return cuts, cutters


def final_cut(cutters, poly):
    """Produce the final cut."""
    cuts = []
    for i in cutters:
        rght = poly
        if i.crosses(poly):
            try:
                left, rght = poly.cut(i)
                if rght is None:
                    cuts.append(left)
                cuts.append(left)
                poly = rght
                rght = left
            except RuntimeError:
                tweet("Issues with poly...{}".format(p_id))
                continue
        else:
            cuts.append(rght)
    return cuts  # , cutters


# ---- demo and tool section -------------------------------------------------
#

if len(sys.argv) == 1:
    testing = False
    in_pth = script.split("/")[:-2] + ["Polygon_lineTools.gdb"]
    # in_fc = "/".join(in_pth) + "/shapes_mtm9"
    # out_fc = "/".join(in_pth) + "/c0"
    in_fc = r"C:\Git_Dan\npgeom\Project_npg\tests.gdb\sq"
    out_fc = r"C:\Git_Dan\npgeom\Project_npg\tests.gdb\x"
    s_axis = "Y"
    s_fac = 4
else:
    testing = False
    in_fc = sys.argv[1]
    out_fc = sys.argv[2]
    s_fac = int(sys.argv[3])
    s_axis = sys.argv[4]

# ---- for both
#
shp_fld, oid_fld, shp_type, SR = fc_info(in_fc)
out_polys, out_ids = get_polys(in_fc)
# old_ids = np.repeat(out_ids, s_fac)  # produce data for the output id field

# ---- instant bail
if SR.type == 'Projected':
    result_ = []
    for i, poly in enumerate(out_polys):
        p_id = out_ids[i]
        cuts, cutters = _cut_poly(poly, p_id, step=1,
                                  split_axis=s_axis,
                                  split_fac=4, SR=SR)
        out = cal_area(poly, cuts, cutters, s_fac)
        tweet(out)
        f_cutters = [cutters[i[0]] for i in out]  # idxs, area, poly
        r = final_cut(f_cutters, poly)
        result_.extend(r)
    if not testing:
        if arcpy.Exists(out_fc):
            arcpy.Delete_management(out_fc)
        arcpy.CopyFeatures_management(result_, out_fc)
        out_ids = np.repeat(out_ids, s_fac)
        id_fld = np.zeros((len(result_),),
                          dtype=[("key", "<i4"), ("Old_ID", "<i4")])
        id_fld["key"] = np.arange(1, len(result_) + 1)
        id_fld["Old_ID"] = out_ids
        arcpy.da.ExtendTable(out_fc, oid_fld, id_fld, "key")
else:
    msg = """
    -----------------------------------------------------------------
    Input data is not in a projected coordinate system....
    bailing...
    -----------------------------------------------------------------
    """
    tweet(msg)

# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
    in_fc = r"C:\Git_Dan\npgeom\Project_npg\tests.gdb\sq"
