# -*- coding: utf-8 -*-
"""
=========
tbx_tools
=========

Script :
    tbx_tools.py for npgeom

Author :
    Dan_Patterson@carleton.ca

Modified :
    2019-10-19

Purpose :
    Tools for working with ``free`` ArcGIS Pro functionality

Notes :

References
----------

**Advanced license tools**

Some of the functions that you can replicate using this data class would
include:

**Containers**

`1 Bounding circles
<https://pro.arcgis.com/en/pro-app/tool-reference/data-management/minimum
-bounding-geometry.htm>`_.  minimum area bounding circle

`2 Convex hulls
<https://pro.arcgis.com/en/pro-app/tool-reference/data-management/minimum
-bounding-geometry.htm>`_.

`3 Feature Envelope to Polygon
<https://pro.arcgis.com/en/pro-app/tool-reference/data-management/feature
-envelope-to-polygon.htm>`_.  axis oriented envelope

**Conversion**

`1 Feature to Point
<https://pro.arcgis.com/en/pro-app/tool-reference/data-management/feature
-to-point.htm>`_.  centroid for point clusters, polylines or polygons

`2 Polygons to Polylines
<https://pro.arcgis.com/en/pro-app/tool-reference/data-management/
feature-to-polygon.htm>`_.  Simple conversion.

`3 Feature Vertices to Points
<https://pro.arcgis.com/en/pro-app/tool-reference/data-management/feature
-vertices-to-points.htm>`_.

**Alter geometry**

`Shift, move, translate features
<https://pro.arcgis.com/en/pro-app/tool-reference/editing/
transform-features.htm>`_.

`Sort Geometry
<https://pro.arcgis.com/en/pro-app/tool-reference/data-management/sort.htm>`_.

`Shift features
<https://pro.arcgis.com/en/pro-app/tool-reference/editing/
transform-features.htm>`_.

`4 Split Line at Vertices
<https://pro.arcgis.com/en/pro-app/tool-reference/data-management/
split-line-at-vertices.htm>`_.



`8 Frequency
<https://pro.arcgis.com/en/pro-app/tool-reference/analysis/frequency.htm>`_.

**To do**

`Feature to Line
<https://pro.arcgis.com/en/pro-app/tool-reference/data-management/feature
-to-line.htm>`_.

`Find Identical
<https://pro.arcgis.com/en/pro-app/tool-reference/data-management/
find-identical.htm>`_.

`Unsplit line
<https://pro.arcgis.com/en/pro-app/tool-reference/data-management/
unsplit-line.htm>`_.
"""
# pylint: disable=C0103  # invalid-name
# pylint: disable-E0611  # arcpy.da or arcgisscripting.da issue
# pylint: disable=E1101  # arcpy.da issue
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

import sys
from textwrap import dedent
# import importlib
import numpy as np
# from numpy.lib.recfunctions import structured_to_unstructured as stu
from numpy.lib.recfunctions import unstructured_to_structured as uts
from numpy.lib.recfunctions import append_fields

import arcgisscripting as ags
from arcgisscripting import da
from arcpy import (
        env, gp, AddMessage, Exists, ImportToolbox, Delete_management,
        MultipartToSinglepart_management, XYToLine_management
        )

from npg_io import (getSR, shape_K, fc_data, fc_geometry, geometry_fc)

from npGeo import Geo, Update_Geo

from npg_geom import _polys_to_unique_pnts_


env.overwriteOutput = True

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

# script = sys.argv[0]  # print this should you need to locate the script


tool_list = [
        'Bounding Circles', 'Convex Hulls', 'Extent Polys',
        'Features to Points', 'Polygons to Polylines', 'Vertices to Points',
        'Split at Vertices',
        'Shift Features', 'Rotate Features', 'Fill Holes',
        'Geometry Sort', 'Area Sort', 'Length Sort', 'Extent Sort',
        'Crosstabulate', 'Attribute sort'
        ]


# ===========================================================================
# ---- def section: def code blocks go here ---------------------------------
msg0 = """
----
Either you failed to specify the geodatabase location and filename properly
or you had flotsam, including spaces, in the path, like...\n
  {}\n
Create a safe path and try again...\n
`Filenames and paths in Python`
<https://community.esri.com/blogs/dan_patterson/2016/08/14/filenames-and
-file-paths-in-python>`_.
----
"""

msg_mp_sp = """
----
Multipart shapes have been converted to singlepart, so view any data
carried over during the extendtable join as representing those from
the original data.  Recalculate values where appropriate.
----
"""


def tweet(msg):
    """Print a message for both arcpy and python.
    """
    m = "\n{}\n".format(msg)
    AddMessage(m)
    print(m)


def check_path(fc):
    """
    ---- check_path ----

    Checks for a file geodatabase and a filename. Files and/or paths containing
    `flotsam` are flagged as being invalid.

    Check the geodatabase location and filename properly.  Flotsam, in the
    path or name, consists of one or more of these characters...::

       \'!"#$%&\'()*+,-;<=>?@[]^`{|}~  including the `space`

    Create a safe path and try again...

    References
    ----------
    `Lexical analysis
    <https://docs.python.org/3/reference/lexical_analysis.html>`_.

    `Filenames and paths in Python
    <https://community.esri.com/blogs/dan_patterson/2016/08/14/filenames-and
    -file-paths-in-python>`_.
    """
    msg = dedent(check_path.__doc__)
    _punc_ = '!"#$%&\'()*+,-;<=>?@[]^`~}{ '
    flotsam = " ".join([i for i in _punc_])  # " ... plus the `space`"
    fail = False
    if (".gdb" not in fc) or np.any([i in fc for i in flotsam]):
        fail = True
    pth = fc.replace("\\", "/").split("/")
    name = pth[-1]
    if (len(pth) == 1) or (name[-4:] == ".gdb"):
        fail = True
    if fail:
        tweet(msg)
        return (None, None)
    gdb = "/".join(pth[:-1])
    return gdb, name


# ---- Container tools -------------------------------------------------------
# ---- (1) bounding circles
#
def circles(in_fc, gdb, name, kind):
    """Minimum area bounding circles.  Change `angle=2` to a smaller value for
    denser points on circle perimeter.
    `getSR`, `shape_k`  and `fc_geometry` are from npg_io.
    """
    SR = getSR(in_fc)
    kind, k = shape_K(in_fc)
    tmp, IFT = fc_geometry(in_fc, SR=SR, IFT_rec=False)
    m = np.nanmin(tmp, axis=0)                   # shift to LB of whole extent
    info = "bounding circles"
    a = tmp - m
    g = Geo(a, IFT, k, info)                     # create the geo array
    out = g.bounding_circles(angle=2, return_xyr=False)
    circs = [arr + m for arr in out]
    k = 1
    if kind == 'Polygons':
        k = 2
    circs = Update_Geo(circs, K=k, id_too=None, Info=info)
    # produce the geometry
    p = kind.upper()
    geometry_fc(circs, circs.IFT, p_type=p, gdb=gdb, fname=name, sr=SR)
    return "{} completed".format("Circles")


# ---- (2) convex hulls
#
def convex_hull_polys(in_fc, gdb, name, kind):
    """Determine the convex hulls on a shape basis"""
    SR = getSR(in_fc)
    kind, k = shape_K(in_fc)
    tmp, IFT = fc_geometry(in_fc, SR=SR, IFT_rec=False)
    info = "convex hulls to polygons"
    g = Geo(tmp, IFT, k, info)                   # create the geo array
    ch_out = g.convex_hulls(by_part=False, threshold=50)
    ch_out = Update_Geo(ch_out, K=k, id_too=None, Info=info)
    # ---- produce the geometry
    p = kind.upper()
    geometry_fc(ch_out, ch_out.IFT, p_type=p, gdb=gdb, fname=name, sr=SR)
    return "{} completed".format("Convex Hulls")


# ---- (3) extent_poly section
#
def extent_poly(in_fc, gdb, name, kind):
    """Feature envelope to polygon demo.

    Parameters
    ----------
    in_fc : string
        Full geodatabase path and featureclass filename.
    kind : integer
        2 for polygons, 1 for polylines

    References
    ----------
    `Feature Envelope to Polygon
    <https://pro.arcgis.com/en/pro-app/tool-reference/data-management/feature
    -envelope-to-polygon.htm>`_.

    >>> data = fc_data(in_fc)
    """
    SR = getSR(in_fc)
    kind, k = shape_K(in_fc)
    tmp, IFT = fc_geometry(in_fc, SR=SR, IFT_rec=False)
    m = np.nanmin(tmp, axis=0)                   # shift to LB of extent
    info = "extent to polygons"
    a = tmp - m
    g = Geo(a, IFT, k, Info=info)   # create the geo array
    ext = g.extent_rectangles()   # create the extent array
    ext = ext + m                 # shift back, construct the output features
    ext = Update_Geo(ext, K=k, id_too=None, Info=info)
    # ---- produce the geometry
    p = kind.upper()
    geometry_fc(ext, ext.IFT, p_type=p, gdb=gdb, fname=name, sr=SR)
    return "{} completed".format("Extents")


# ---- Conversion Tools ------------------------------------------------------
# ---- (1) features to point
#
def f2pnts(in_fc):
    """Features to points.
    `getSR`, `shape_K` and `fc_geometry` from `npGeo_io`
    """
    SR = getSR(in_fc)
    kind, k = shape_K(in_fc)
    tmp, ift = fc_geometry(in_fc, SR=SR, IFT_rec=False)
    m = np.nanmin(tmp, axis=0)                   # shift to LB of whole extent
    info = "feature to points"
    a = tmp - m
    g = Geo(a, IFT=ift, Kind=k, Info=info)    # create the geo array
    cent = g.centroids + m                       # create the centroids
    dt = np.dtype([('Xs', '<f8'), ('Ys', '<f8')])
    cent = uts(cent, dtype=dt)
    return cent, SR


# ---- (2) polygon to polyline
#
def pgon_to_pline(in_fc, gdb, name):
    """Polygon to polyline conversion.  Multipart shapes are converted to
    singlepart.  The singlepart geometry is used to produce the polylines."""
#    gdb, name = check_path(out_fc)
#    if gdb is None:
#        return None
    SR = getSR(in_fc)
    tmp = MultipartToSinglepart_management(in_fc, r"memory\in_fc_temp")
    a, IFT = fc_geometry(tmp, SR=SR, IFT_rec=False)
    info = "pgon to pline"
    # create the geo array, convert it, then create the output featureclass
    a = Geo(a, IFT=IFT, Kind=1, Info=info)       # create the geo array
    geometry_fc(a, IFT, p_type="POLYLINE", gdb=gdb, fname=name, sr=SR)
    out = "{}/{}".format(gdb, name)
    if Exists(out):
        d = fc_data(tmp)
        import time
        time.sleep(1.0)
        da.ExtendTable(out, 'OBJECTID', d, 'OID_')
    tweet(dedent(msg_mp_sp))
    return


# ---- (3) vertices to points
#
def p_uni_pnts(in_fc):
    """Implements `_polys_to_unique_pnts_` in ``npg_helpers``.
    """
    SR = getSR(in_fc)
    tmp, IFT = fc_geometry(in_fc, SR=SR, IFT_rec=False)
    info = "unique points"
    a = Geo(tmp, IFT=IFT, Kind=0, Info=info)     # create the geo array
    out = _polys_to_unique_pnts_(a, as_structured=True)
    return out, SR


# ---- Alter Geometry --------------------------------------------------------
# ---- (1) rotate features
def rotater(in_fc, gdb, name, as_group, angle, clockwise):
    """Rotate features separately or as a group.
    """
    SR = getSR(in_fc)
    kind, k = shape_K(in_fc)                        # geometry type, 0, 1, 2
    a, IFT = fc_geometry(in_fc, SR=SR, IFT_rec=False)
    tmp = MultipartToSinglepart_management(in_fc, r"memory\in_fc_temp")
    info = "rotate features"
    a = Geo(a, IFT=IFT, Kind=k, Info=info)       # create the geo array
    s = a.rotate(as_group=as_group, angle=angle, clockwise=clockwise)
    p = kind.upper()
    geometry_fc(s, s.IFT, p_type=p, gdb=gdb, fname=name, sr=SR)
    out = "{}/{}".format(gdb, name)
    if Exists(out):
        import time
        time.sleep(1.0)
        d = fc_data(tmp)
        da.ExtendTable(out, 'OBJECTID', d, 'OID_')
    return


def fill_holes(in_fc, gdb, name):
    """Fill holes in a featureclass.  See the Eliminate part tool.
    """
    SR = getSR(in_fc)
    kind, k = shape_K(in_fc)                        # geometry type, 0, 1, 2
    tmp = MultipartToSinglepart_management(in_fc, r"memory\in_fc_temp")
    a, IFT = fc_geometry(tmp, SR=SR, IFT_rec=False)
    info = "fill holes"
    a = Geo(a, IFT=IFT, Kind=k, Info=info)       # create the geo array
    oring = a.outer_rings(True)
    p = kind.upper()
    geometry_fc(oring, oring.IFT, p_type=p, gdb=gdb, fname=name, sr=SR)
    out = "{}/{}".format(gdb, name)
    if Exists(out):
        import time
        time.sleep(1.0)
        d = fc_data(tmp)
        da.ExtendTable(out, 'OBJECTID', d, 'OID_')
    return


# ---- (2) sorty by area, length
#
def sort_geom(in_fc, gdb, name, sort_kind):
    """Sort features by area, length
    """
    SR = getSR(in_fc)
    kind, k = shape_K(in_fc)                        # geometry type, 0, 1, 2
    a, IFT = fc_geometry(in_fc, SR=SR, IFT_rec=False)
    info = "sort features"
    a = Geo(a, IFT=IFT, Kind=k, Info=info)
    if sort_kind == 'area':
        srt = a.sort_by_area(ascending=True, just_indices=False)
    elif sort_kind == 'length':
        srt = a.sort_by_length(ascending=True, just_indices=False)
    p = kind.upper()
    geometry_fc(srt, srt.IFT, p_type=p, gdb=gdb, fname=name, sr=SR)
    return


# ---- (3) sorty by extent
#
def sort_extent(in_fc, gdb, name, key):
    """Sort features by extent, area, length
    """
    SR = getSR(in_fc)
    kind, k = shape_K(in_fc)                        # geometry type, 0, 1, 2
    a, IFT = fc_geometry(in_fc, SR=SR, IFT_rec=False)
    info = "sort features"
    a = Geo(a, IFT=IFT, Kind=k, Info=info)
    srt = a.sort_by_extent(key, just_indices=False)
    p = kind.upper()
    geometry_fc(srt, srt.IFT, p_type=p, gdb=gdb, fname=name, sr=SR)
    return


# ---- (4) move/shift/translate features
#
def shifter(in_fc, gdb, name, dX, dY):
    """Shift features to a new location by delta X and Y values.  Multipart
    shapes are converted to singlepart shapes.
    """
    SR = getSR(in_fc)
    desc = da.Describe(in_fc)
    kind = desc['shapeType']
    kind, k = shape_K(in_fc)                        # geometry type, 0, 1, 2
    a, IFT = fc_geometry(in_fc, SR=SR, IFT_rec=False)
    tmp = MultipartToSinglepart_management(in_fc, r"memory\in_fc_temp")
    info = "shift features"
    # create the geo array, shift it, then create the output featureclass
    a = Geo(a, IFT=IFT, Kind=k, Info=info)
    s = a.shift(dX, dY)
    p = kind.upper()
    geometry_fc(s, s.IFT, p_type=p, gdb=gdb, fname=name, sr=SR)
    out = "{}/{}".format(gdb, name)
    if Exists(out):
        import time
        time.sleep(1.0)
        d = fc_data(tmp)
        da.ExtendTable(out, 'OBJECTID', d, 'OID_')
    return


# ---- (5) split line at vertices
#
def split_at_vertices(in_fc, out_fc):
    """Unique segments retained when poly geometry is split at vertices.
    """
    gdb, _ = check_path(out_fc)
    if gdb is None:
        return None
    SR = getSR(in_fc)
    tmp, IFT = fc_geometry(in_fc, SR=SR, IFT_rec=False)
    ag = Geo(tmp, IFT)
    od = ag.polys_to_segments(as_basic=False, as_3d=False)
    tmp = "memory/tmp"
    if Exists(tmp):
        Delete_management(tmp)
    ags.da.NumPyArrayToTable(od, tmp)
    xyxy = list(od.dtype.names[:4])
    args = [tmp, out_fc] + xyxy + ["GEODESIC", "Orig_id", SR]
    XYToLine_management(*args)
    return


# ---- Attribute tools -------------------------------------------------------
# ---- (1) frequency and statistics
#
def freq(a, cls_flds, stat_fld):
    """Frequency and crosstabulation.

    Parameters
    ----------
    a : array
        A structured array.
    flds : field
        Fields to use in the analysis.

    Notes
    -----
    1. Slice the input array by the classification fields.
    2. Sort the sliced array using the flds as sorting keys.
    3. Use unique on the sorted array to return the results and the counts.

    >>> np.unique(ar, return_index=False, return_inverse=False,
    ...           return_counts=True, axis=None)
    """
    if stat_fld is None:
        a = a[cls_flds]  # (1) It is actually faster to slice the whole table
    else:
        all_flds = cls_flds + [stat_fld]
        a = a[all_flds]
    idx = np.argsort(a, axis=0, order=cls_flds)  # (2)
    a_sort = a[idx]
    uni, inv, cnts = np.unique(a_sort[cls_flds], False,
                               True, return_counts=True)  # (3)
    out_flds = "Counts"
    out_data = cnts
    if stat_fld is not None:
        splitter = np.where(np.diff(inv) == 1)[0] + 1
        a0 = a_sort[stat_fld]
        splits = np.split(a0, splitter)
        sums = np.asarray([np.nansum(i.tolist()) for i in splits])
        nans = np.asarray([np.sum(np.isnan(i.tolist())) for i in splits])
        mins = np.asarray([np.nanmin(i.tolist()) for i in splits])
        means = np.asarray([np.nanmean(i.tolist()) for i in splits])
        maxs = np.asarray([np.nanmax(i.tolist()) for i in splits])
        out_flds = [out_flds, stat_fld + "_sums", stat_fld + "_NaN",
                    stat_fld + "_min", stat_fld + "_mean", stat_fld + "_max"]
        out_data = [out_data, sums, nans, mins, means, maxs]
    out = append_fields(uni, names=out_flds, data=out_data, usemask=False)
    return out


def attr_sort(a, oid_fld, sort_flds):
    """Return old and new id values for the sorted array.
    """
    idx = np.argsort(a, order=sort_flds)
    srted = a[idx]
    dt = [(oid_fld, '<i4'), ('Sorted_', '<i4')]
    out = np.zeros_like(srted, dtype=np.dtype(dt))  # create the new array
    out[oid_fld] = srted[oid_fld]
    out['Sorted_'] = np.arange(0, out.shape[0])
    return out


# ===========================================================================
# ---- main section: testing or tool run ------------------------------------
#
script = sys.argv[0]
pth = "/".join(script.split("/")[:-1])
tbx = pth + "/Free_tools.tbx"
tbx = ImportToolbox(tbx)

frmt = """
Source script... {}
Using :
    tool   : {}
    input  : {}
    output : {}
"""


def pick_tool(tool, in_fc, out_fc, gdb, name):
    """Pick the tool and run the option.
    """
    # ---- Geometry tools ----------------------------------------------------
    #
    # ---- Containers
    if tool in ['Bounding Circles', 'Convex Hulls', 'Extent Polys']:
        kind = sys.argv[4].upper()
        if tool == 'Bounding Circles':           # ---- (1) bounding circles
            circles(in_fc, gdb, name, kind)
        elif tool == 'Convex Hulls':             # ---- (2) convex hulls
            convex_hull_polys(in_fc, gdb, name, kind)
        elif tool == 'Extent Polys':             # ---- (3) extent_poly
            extent_poly(in_fc, gdb, name, kind)
        tweet("...\n{} as {}".format(tool, kind.title()))
    #
    # ---- Conversion
    elif tool in ['Features to Points', 'Vertices to Points']:
        if tool == 'Features to Points':         # ---- (1) features to point
            out, SR = f2pnts(in_fc)
        elif tool == 'Vertices to Points':       # ---- (2) feature to vertices
            out, SR = p_uni_pnts(in_fc)
        ags.da.NumPyArrayToFeatureClass(out, out_fc, ['Xs', 'Ys'], SR)
        tweet("...\n{} as {}".format(tool, 'Points'))
    elif tool == 'Polygons to Polylines':        # ---- (3) polygon to polyline
        tweet("...\nPolygons to Polylines...\n")
        pgon_to_pline(in_fc, gdb, name)
    elif tool == 'Split at Vertices':            # ---- (4) split at vertices
        tweet("...\n{} as {}".format(tool, 'Lines'))
        split_at_vertices(in_fc, out_fc)
    #
    # ---- Alter geometry
    elif tool == 'Rotate Features':              # ---- (1) rotate
        clockwise = False
        as_group = False
        rot_type = str(sys.argv[4])  # True, extent center, False, shape center
        angle = float(sys.argv[5])
        clockwise = str(sys.argv[6])
        if rot_type == "shape center":
            as_group = True
        if clockwise.lower() == "true":
            clockwise = True
        tweet("...\n{} {}".format(tool, 'Features'))
        rotater(in_fc, gdb, name, as_group, angle, clockwise)
    elif tool == 'Shift Features':               # ---- (5) shift
        dX = float(sys.argv[4])
        dY = float(sys.argv[5])
        tweet("...\n{} {}".format(tool, 'Features'))
        shifter(in_fc, gdb, name, dX=dX, dY=dY)
    elif tool == 'Fill Holes':
        tweet("not implemented yet")
        fill_holes(in_fc, gdb, name)
    #
    # ---- Sort geometry
    elif tool in ['Area Sort', 'Length Sort', 'Geometry Sort']:
        srt_type = tool.split(" ")[0].lower()
        tweet("...\n{} as {}".format(tool, 'input'))
        sort_geom(in_fc, gdb, name, srt_type)
    elif tool == 'Extent Sort':
        srt_type = int(sys.argv[4][0])
        tweet("...\n{} as {}".format(tool, 'input'))
        sort_extent(in_fc, gdb, name, srt_type)
    #
    # ---- Attribute tools --------------------------------------------------
    elif tool == 'Crosstabulate':                # ---- (1) freq and stats
        cls_flds = sys.argv[4]
        stat_fld = sys.argv[5]
        cls_flds = cls_flds.split(";")  # multiple to list, singleton a list
        if stat_fld in (None, 'NoneType', ""):
            stat_fld = None
        a = ags.da.TableToNumPyArray(in_fc, "*")  # use the whole array
        tweet("result...\n{}".format(a))
        out = freq(a, cls_flds, stat_fld)        # do freq analysis
        if Exists(out_fc) and env.overwriteOutput:
            Delete_management(out_fc)
        ags.da.NumPyArrayToTable(out, out_fc)
    elif tool == 'Attribute sort':
        sort_flds = str(sys.argv[4])  # just tool and in_fc, extend to existing
        sort_flds = sort_flds.split(";")
        msg = """\
        ------------------
        Sorting      : {}
        Using fields : {}
        Output field : {}
        -----------------
        """
        tweet(dedent(msg).format(in_fc, sort_flds, "Sorted_"))
        oid_fld = da.Describe(in_fc)['OIDFieldName']
        flds = [oid_fld] + sort_flds
        a = ags.da.TableToNumPyArray(in_fc, flds)
        out = attr_sort(a, oid_fld, sort_flds)    # do the work... attr_sort
        da.ExtendTable(in_fc, oid_fld, out, oid_fld, append_only=False)
    else:
        tweet("tool {} not found".format(tool))
        return None
    # ---- (


# ==== testing or tool run ===================================================
#
def _testing_():
    """Run in spyder
    """
    in_fc = "C:/Git_Dan/npgeom/npgeom.gdb/Polygons"
    out_fc = "C:/Git_Dan/npgeom/npgeom.gdb/x"
    tool = 'ShiftFeatures'  # None  #
    info_ = gp.getParameterInfo(tool)
    for param in info_:
        print("Name: {}, Type: {}, Value: {}".format(
            param.name, param.parameterType, param.value))
    print("Input {}\nOutput {}".format(in_fc, out_fc))
    return info_, in_fc, out_fc  # in_fc, out_fc, tool, kind


def _tool_(tools=tool_list):
    """Run from a tool in arctoolbox in ArcGIS Pro.  The tool checks to ensure
    that the path to the output complies and that the desired tool actually
    exists, so it can be parsed based on type.
    """
    tool = sys.argv[1]
    in_fc = sys.argv[2]
    out_fc = sys.argv[3]
    tweet("out_fc  {}".format(out_fc))
    if out_fc not in (None, 'None'):
        gdb, name = check_path(out_fc)               # ---- check the paths
        if gdb is None:
            tweet(msg0)
            return None
    else:
        gdb = None
        name = None
    if tool not in tools:                        # ---- check the tool
        tweet("Tool {} not implemented".format(tool))
        return None
    msg1 = "Tool   : {}\ninput  : {}\noutput : {}"
    tweet(msg1.format(tool, in_fc, out_fc))
    pick_tool(tool, in_fc, out_fc, gdb, name)    # ---- run the tool
    return  # tool, in_fc, out_fc


# ===========================================================================
# ---- main section: testing or tool run
#
if len(sys.argv) == 1:
    testing = True
    result = _testing_()
else:
    testing = False
    _tool_(tool_list)

# ===========================================================================
#
if __name__ == "__main__":
    """optional location for parameters"""
    info_ = _testing_()
