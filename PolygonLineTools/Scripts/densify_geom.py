# -*- coding: UTF-8 -*-
"""
densify_geom
============

Script  :   densify_geom.py

Author  :   Dan.Patterson@carleton.ca

Modified :  2018-09-12

Purpose :   Densify geometry by a factor.

Notes :
  Uses functions from 'arraytools'.  These have been consolidated here.
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
import numpy as np
from arcpytools_plt import tweet, fc_info, fc_array
import arcpy

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


# ---- array functions -------------------------------------------------------
#
def _flat_(a_list, flat_list=None):
    """Change the isinstance as appropriate
    :  Flatten an object using recursion
    :  see: itertools.chain() for an alternate method of flattening.
    """
    if flat_list is None:
        flat_list = []
    for item in a_list:
        if isinstance(item, (list, tuple, np.ndarray, np.void)):
            _flat_(item, flat_list)
        else:
            flat_list.append(item)
    return flat_list


def _O_nd(obj, out=None):
    """Flatten type 'O' arrays to ndarray, using recursion
    :Note: append retains internal shape, extend will flatten
    :  nested lists into a list
    """
    if out is None:
        out = []
    sub_out = []
    for el in obj:
        el = np.asarray(el)
        if el.dtype.kind in ('O', 'V'):
            sub_out.append(_O_nd(el, out))  # ---- recursion needed ---
        else:
            out.extend(el)  # was append
    return out

def _densify_2D(a, fact=2):
    """Densify a 2D array using np.interp.
    :fact - the factor to density the line segments by
    :Notes
    :-----
    :original construction of c rather than the zero's approach
    :  c0 = c0.reshape(n, -1)
    :  c1 = c1.reshape(n, -1)
    :  c = np.concatenate((c0, c1), 1)
    """
    # Y = a changed all the y's to a
    a = np.squeeze(a)
    n_fact = len(a) * fact
    b = np.arange(0, n_fact, fact)
    b_new = np.arange(n_fact - 1)     # Where you want to interpolate
    c0 = np.interp(b_new, b, a[:, 0])
    c1 = np.interp(b_new, b, a[:, 1])
    n = c0.shape[0]
    c = np.zeros((n, 2))
    c[:, 0] = c0
    c[:, 1] = c1
    return c


# ---- featureclass functions ------------------------------------------------
#
def _get_shapes(in_fc):
    """Get shapes from a featureclass, in_fc, using SHAPE@ returning
    :  [<Polygon object at....>, ... (<Polygon object at....>]
    """
    with arcpy.da.SearchCursor(in_fc, 'SHAPE@') as cursor:
        a = [row[0] for row in cursor]
    return a


def obj_shapes(in_, SR):
    """object array of coordinates to shapes"""
    s = []
    for shps in in_:
        tmp = []
        if isinstance(shps, (list, tuple)):
            for shp in shps:
                shp = np.asarray(shp)
                shp = shp.squeeze()
                pnts = [arcpy.Point(*p) for p in shp]
                tmp.append(pnts)
            arr = arcpy.Array(pnts)
        else:
            arr = arcpy.Array([arcpy.Point(*p) for p in shps])
        #
        if out_type == 'Polyline':
            g = arcpy.Polyline(arr, SR)
        elif out_type == 'Polygon':
            g = arcpy.Polygon(arr, SR)
        s.append(g)
    return s


def arcpnts_poly(in_, out_type='Polygon', SR=None):
    """Convert arcpy Point lists to poly* features
    : out_type - either 'Polygon' or 'Polyline'
    :
    """
    s = []
    for i in in_:
        for j in i:
            if out_type == 'Polygon':
                g = arcpy.Polygon(arcpy.Array(j), SR)
            elif out_type == 'Polyline':
                g = arcpy.Polyline(arcpy.Array(j), SR)
            elif out_type == 'Points':
                j = _flat_(j)
                g = arcpy.Multipoint(arcpy.Array(j), SR)  # check
            s.append(g)
    return s


def _convert(a, fact=2):
    """Do the shape conversion for the array parts.  Calls to _densify_2D
    """
    out = []
    parts = len(a)
    for i in range(parts):
        sub_out = []
        p = np.asarray(a[i]).squeeze()
        if p.ndim == 2:
            shp = _densify_2D(p, fact=fact)  # call _densify_2D
            arc_pnts = [arcpy.Point(*p) for p in shp]
            sub_out.append(arc_pnts)
            out.extend(sub_out)
        else:
            for i in range(len(p)):
                pp = p[i]
                shp = _densify_2D(pp, fact=fact)
                arc_pnts = [arcpy.Point(*p) for p in shp]
                sub_out.append(arc_pnts)
            out.append(sub_out)
    return out


def densify(polys, fact=2, sp_ref=None):
    """Convert polygon objects to arrays, densify.
    :
    :Requires:
    :--------
    : _densify_2D - the function that is called for each shape part
    : _unpack - unpack objects
    """
    # ---- main section ----
    out = []
    for poly in polys:
        p = poly.__geo_interface__['coordinates']
        back = _convert(p, fact)
        out.append(back)
    return out


# ---- main block ------------------------------------------------------------
#
# (1) obtain fc information
# (2) convert multipart to singlepart
# (3) split the fc into two arrays, one geometry, the 2nd attributes
# (4) obtain the shapes and densify
# (5) optionally produce the output fc
# (6) join the attributes back

if len(sys.argv) == 1:
    in_pth = script.split("/")[:-2] + ["Polygon_lineTools.gdb"]
    in_fc = "/".join(in_pth) + "/shapes_mtm9"#    in_fc = r"C:\Git_Dan\a_Data\arcpytools_demo.gdb\xy1000_tree"
    out_fc = "/".join(in_pth) + '/x2'
    fact = 2
    out_type = 'Polygon'  # 'Polyline' or 'Points'
    testing = False
else:
    in_fc = sys.argv[1]  #
    out_fc = sys.argv[2]  #
    fact = int(sys.argv[3])  #
    out_type = sys.argv[4]  # Polygon, Polyline are options
    testing = False


shp_fld, oid_fld, shp_type, SR = fc_info(in_fc)
temp = out_fc + "tmp"
if arcpy.Exists(temp):
    arcpy.Delete_management(temp)
arcpy.MultipartToSinglepart_management(in_fc, temp)
polys = _get_shapes(temp)
a = densify(polys, fact=fact, sp_ref=SR)
b, _, _ = fc_array(in_fc, flds="*", allpnts=False) #_get_attributes(temp)
dt = b.dtype.descr
dtn = [(i[0].replace("@", "_"), i[1]) for i in dt]
b.dtype = np.dtype(dtn)
out_shps = arcpnts_poly(a, out_type=out_type, SR=SR)
#
# ---- if not testing, save the geometry and extend (join) the attributes
if not testing:
    if arcpy.Exists(out_fc):
        arcpy.Delete_management(out_fc)
    arcpy.CopyFeatures_management(out_shps, out_fc)
    arcpy.da.ExtendTable(out_fc, 'OBJECTID', b, 'OBJECTID', append_only=False)
# ---- cleanup
arcpy.Delete_management(temp)


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
