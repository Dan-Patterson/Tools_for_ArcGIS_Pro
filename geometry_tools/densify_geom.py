# -*- coding: UTF-8 -*-
"""
:Script:   densify_geom.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2018-01-16
:Purpose:  Densify geometry by a factor.
:Notes:
:  Uses functions from 'arraytools'.  These have been consolidated here.
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
import numpy as np
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
def fc_info(in_fc, prn=False):
    """Return basic featureclass information, including...
    :
    : shp_fld  - field name which contains the geometry object
    : oid_fld  - the object index/id field name
    : SR       - spatial reference object (use SR.name to get the name)
    : shp_type - shape type (Point, Polyline, Polygon, Multipoint, Multipatch)
    : - others: 'areaFieldName', 'baseName', 'catalogPath','featureType',
    :   'fields', 'hasOID', 'hasM', 'hasZ', 'path'
    : - all_flds =[i.name for i in desc['fields']]
    """
    desc = arcpy.da.Describe(in_fc)
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


def tweet(msg):
    """Print a message for both arcpy and python.
    : msg - a text message
    """
    m = "\n{}\n".format(msg)
    arcpy.AddMessage(m)
    print(m)
    print(arcpy.GetMessages())


def _get_shapes(in_fc):
    """Get shapes from a featureclass, in_fc, using SHAPE@ returning
    :  [<Polygon object at....>, ... (<Polygon object at....>]
    """
    with arcpy.da.SearchCursor(in_fc, 'SHAPE@') as cursor:
        a = [row[0] for row in cursor]
    return a


def _ndarray(in_fc, to_pnts=True, flds=None, SR=None):
    """Convert featureclass geometry (in_fc) to a structured ndarray including
    :  options to select fields and specify a spatial reference.
    :
    :Requires:
    :--------
    : in_fc - input featureclass
    : to_pnts - True, convert the shape to points. False, centroid returned.
    : flds - '*' for all, others: 'Shape',  ['SHAPE@X', 'SHAPE@Y'], or specify
    """
    if flds is None:
        flds = "*"
    if SR is None:
        desc = arcpy.da.Describe(in_fc)
        SR = desc['spatialReference']
    args = [in_fc, flds, None, SR, to_pnts, (None, None)]
    cur = arcpy.da.SearchCursor(*args)
    a = cur._as_narray()
    del cur
    return a


def _get_attributes(in_fc):
    """Get the attributes of features, returns the centroid coordinates
    :  as fields in the table.
    """
    dt_b = [('IDs', '<i4'), ('Xc', '<f8'), ('Yc', '<f8')]
    b = _ndarray(in_fc, to_pnts=False)
    dt = [[n, t] for n, t in b.dtype.descr[2:]]
    for i in dt:
        if i[0] in ["Shape_Length", "Shape_Area", "Shape"]:
            i[0] = i[0] + "_orig"
    dt = [tuple(i) for i in dt]
    dt_b.extend(dt)
    b.dtype = dt_b
    return b


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


# ------------------------------------------------------------------------
# (1) ---- Checks to see if running in test mode or from a tool ----------

def _demo():
    """run when script is in demo mode"""
    pth = script.replace('densify_geom.py', '')
    in_fc = pth + '/geom_data.gdb/polygon_demo'
#    in_fc = r"C:\Git_Dan\a_Data\testdata.gdb\Carp_5x5"   # full 25 polygons
#    in_fc = r"C:\Git_Dan\a_Data\arcpytools_demo.gdb\xy1000_tree"
    out_fc = pth + '/geom_data.gdb/x1'
    fact = 2
    out_type = 'Polygon'  # 'Polyline' or 'Points'
    testing = True
    return in_fc, out_fc, fact, out_type, testing


def _tool():
    """run when script is from a tool"""
    in_fc = sys.argv[1]  #
    out_fc = sys.argv[2]  #
    fact = int(sys.argv[3])  #
    out_type = sys.argv[4]  # Polygon, Polyline are options
    testing = False
    return in_fc, out_fc, fact, out_type, testing


# ---- main block ------------------------------------------------------------
#
# (1) check to see if in demo or tool mode
# (2) obtain fc information
# (3) convert multipart to singlepart
# (4) split the fc into two arrays, one geometry, the 2nd attributes
# (5) obtain the shapes and densify
# (6) optionally produce the output fc
# (7) join the attributes back

if len(sys.argv) == 1:
    in_fc, out_fc, fact, out_type, testing = _demo()
else:
    in_fc, out_fc, fact, out_type, testing = _tool()

shp_fld, oid_fld, shp_type, SR = fc_info(in_fc)

temp = out_fc + "tmp"
if arcpy.Exists(temp):
    arcpy.Delete_management(temp)
arcpy.MultipartToSinglepart_management(in_fc, temp)
polys = _get_shapes(temp)
a = densify(polys, fact=fact, sp_ref=SR)
b = _get_attributes(temp)
out_shps = arcpnts_poly(a, out_type=out_type, SR=SR)
if not testing:
    if arcpy.Exists(out_fc):
        arcpy.Delete_management(out_fc)
    arcpy.CopyFeatures_management(out_shps, out_fc)
    arcpy.da.ExtendTable(out_fc, 'OBJECTID', b, 'IDs')
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
