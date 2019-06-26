# -*- coding: utf-8 -*-
"""
============
extent_polys
============

Script : extent_polys.py
    Run various functions emulating arctoolbox tools that require an advanced
    license.  These tools use a geometry class and methods based on numpy.

Author :
    Dan_Patterson@carleton.ca

Modified : 2019-06-22
    Initial creation period 2019-06

Requires
--------
- ArcGIS Pro 2.4 + 
- modelled after fc_npGeo and npGeo from the arraytools toolset
"""

import sys
from textwrap import dedent  #, indent
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured as stu
from numpy.lib.recfunctions import unstructured_to_structured as uts

import arcpy
arcpy.env.overwriteOutput = True

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.1f}'.format}
np.set_printoptions(edgeitems=5, linewidth=120, precision=2, suppress=True,
                    threshold=100, formatter=ft)

script = sys.argv[0]

__all__ = ['extGeo', '_updateGeo', '_make_nulls_', 'fc_data',
           'getSR', 'fc_geometry', 'array_poly', 'geometry_fc',
           'check_path', 'extent_poly']
# ---- from npGeo -----------------------------------------------------------
# ==== update Geo array, or create one from a list of arrays ================
#
class extGeo(np.ndarray):
    """ from npGeo"""
    # ----
    def __new__(cls, arr=None, IFT=None, Kind=2, Info=""):
        """see npGeo"""
        msg = extGeo.__new__.__doc__
        arr = np.asarray(arr)
        IFT = np.asarray(IFT)
        cond = [(arr.ndim != 2),
                (IFT.ndim != 2), (IFT.shape[-1] != 3),
                (Kind not in (0, 1, 2))]
        if all(cond):
            print(dedent(msg))
            return None
        # ----
        self = arr.view(cls)     # view as Geo class
        self.IFT = IFT
        self.IDs = IFT[:, 0]
        self.FT = IFT[:, 1:]
        self.K = Kind
        self.Info = Info
        # --- other properties
        self.N = len(np.unique(self.IDs))  # sample size, unique shapes
        if self.shape[1] >= 2:             # X,Y and XY initialize
            self.X = arr[:, 0]
            self.Y = arr[:, 1]
            self.XY = arr[:, :2]
        if self.shape[1] >= 3:   # add Z, although not implemented
            self.Z = arr[:, 2]  # directly, but kept for future additions
        else:
            self.Z = None
        return self

    def __array_finalize__(self, src_arr):
        """see npGeo
        """
        if src_arr is None:
            return
        self.IFT = getattr(src_arr, 'IFT', None)
        self.IDs = getattr(src_arr, 'IDs', None)
        self.FT = getattr(src_arr, 'FT', None)
        self.K = getattr(src_arr, 'K', None)
        self.Info = getattr(src_arr, 'Info', None)
        self.N = getattr(src_arr, 'N', None)
        self.X = getattr(src_arr, 'X', None)
        self.Y = getattr(src_arr, 'Y', None)
        self.XY = getattr(src_arr, 'XY', None)

    def __array_wrap__(self, out_arr, context=None):
        """Wrap it up"""
        return np.ndarray.__array_wrap__(self, out_arr, context)

    # ------------------------------------------------------------------------
    # ------------------- End of class definition ----------------------------
    # ---- basic shape properties and methods to subdivide Geo
    @property
    def shapes(self):
        """Subdivide the array into shapes which may be singlepart or multipart
        Returns an object array or ndarray of points
        """
        uniq = np.unique(self.IDs)
        c = [self.FT[self.IDs == i].ravel() for i in uniq]
        c1 = [(min(i), max(i)) for i in c]
        return np.array([np.asarray(self[f:t]) for f, t in c1]).squeeze()

    @property
    def parts(self):
        """Deconstruct the 2D array into its parts, generally returning an
        object array.  The reverse is np.vstack(self)
        formally: return np.asarray([(self[f:t]) for f, t in self.FT]) but with
        additions to add the FT and IFT properties
        """
        xy = self.base
        if xy is None:
            xy = self.XY.view(np.ndarray)
        return np.asarray(np.split(xy, self.IFT[:, 2]))[:-1]

    # ---- methods -----------------------------------------------------------
    # ---- extents
    def extents(self, by_part=False):
        """Extents are returned as L(eft), B(ottom), R(ight), T(op)
        """
        def _extent_(i):
            """Extent of a sub-array in an object array"""
            return np.concatenate((np.nanmin(i, axis=0), np.nanmax(i, axis=0)))                                                                                      
        # ----
        if self.N == 1:
            by_part = True
        p_ext = [_extent_(i) for i in self.split(by_part)]
        return np.asarray(p_ext)

    def extent_rectangles(self):
        """Return extent polygons for all shapes.  Points are ordered clockwise
         from the bottom left, with the first and last points the same.
         Requires an Advanced license in Pro
         """
        ext_polys = []
        for ext in self.extents():
            L, B, R, T = ext
            poly = np.array([[L, B], [L, T], [R, T], [R, B], [L, B]])
            ext_polys.append(poly)
        return np.asarray(ext_polys)

    def split(self, by_part=False):
        """Split points by shape or by parts for each shape.
        Use self.parts or self.shapes directly"""
        if by_part:
            return self.parts
        return self.shapes

# ----------
#
def _updateGeo(a_2d, K=None, id_too=None, Info=None):
    """Create a new Geo from a list of arrays.

    Parameters
    ----------
    a_2d : list/tuple/array
        Some form of nested 2D array-like structure that can be stacked
    K : integer
        Points (0), polylines (1) or polygons (2)
    id_too : array-like
        If None, then the structure will be created.
    Info : text (optional)
        Provide any information that will help in identifying the array.

    Returns
    -------
    A new Geo array is returned given the inputs.
    """
    if K not in (0, 1, 2):
        print("Output type not specified, or not in (0, 1, 2)")
        return None
    if id_too is None:
        id_too = [(i, len(a)) for i, a in enumerate(a_2d)]
    a_2d = np.vstack(a_2d)
    id_too = np.array(id_too)
    I = id_too[:, 0]
    too = np.cumsum(id_too[:, 1])
    frum = np.concatenate(([0], too))
    IFT = np.array(list(zip(I, frum, too)))
    new_geo = extGeo(a_2d, IFT, K, Info)
    return new_geo


# ---- from fc_npGeo --------------------------------------------------------
# ---- Used to create the inputs for the Geo class
#
def _make_nulls_(in_fc, int_null=-999):
    """Return null values for a list of fields objects, excluding objectid
    and geometry related fields.

    See Also
    --------
    See ... arraytools/fc_tools/fc_npGeo.py for documentation
    """
    nulls = {'Double': np.nan, 'Single': np.nan, 'Float': np.nan,
             'Short': int_null, 'SmallInteger': int_null, 'Long': int_null,
             'Integer': int_null, 'String':str(None), 'Text':str(None),
             'Date': np.datetime64('NaT')}
    #
    desc = arcpy.da.Describe(in_fc)
    if desc['dataType'] != 'FeatureClass':
        print("Only Featureclasses are supported")
        return None, None
    in_flds = desc['fields']
    shp = desc['shapeFieldName']
    good = [f for f in in_flds if f.editable and f.name != shp]
    fld_dict = {f.name: f.type for f in good}
    fld_names = list(fld_dict.keys())
    null_dict = {f: nulls[fld_dict[f]] for f in fld_names}
    # ---- insert the OBJECTID field
    return null_dict, fld_names

def fc_data(in_fc):
    """Pull all editable attributes from a featureclass tables.  During the
    process, <null> values are changed to an appropriate type.

    See Also
    --------
    See ... arraytools/fc_tools/fc_npGeo.py for documentation.
    """
    flds = ['OID@', 'SHAPE@X', 'SHAPE@Y']
    null_dict, fld_names = _make_nulls_(in_fc, int_null=-999)
    fld_names = flds + fld_names
    new_names = ['OID_', 'X_cent', 'Y_cent']
    a = arcpy.da.FeatureClassToNumPyArray(in_fc, fld_names,
                                          skip_nulls=False,
                                          null_value=null_dict)
    a.dtype.names = new_names + fld_names[3:]
    return np.asarray(a)


def getSR(in_fc, verbose=False):
    """Return the spatial reference of a featureclass"""
    desc = arcpy.da.Describe(in_fc)
    SR = desc['spatialReference']
    if verbose:
        print("SR name: {}  factory code: {}".format(SR.name, SR.factoryCode))
    return SR


def fc_geometry(in_fc, SR=None, deg=5):
    """See fc_npGeo for details"""
    msg = """
    Use arcpy.FeatureClassToNumPyArray for Point files.
    MultiPoint, Polyline and Polygons and its variants are supported.
    """
    # ----
    def _multipnt_(in_fc, SR):
        """Convert multipoint geometry to array"""
        pnts = arcpy.da.FeatureClassToNumPyArray(
                   in_fc, ['OID@', 'SHAPE@X', 'SHAPE@Y'],
                   spatial_reference=SR,
                   explode_to_points=True)
        id_len = np.vstack(np.unique(pnts['OID@'], return_counts=True)).T
        a_2d = stu(pnts[['SHAPE@X', 'SHAPE@Y']])  # ---- use ``stu`` to convert
        return id_len, a_2d
    # ----
    def _polytypes_(in_fc, SR):
        """Convert polylines/polygons geometry to array"""
        def _densify_curves_(geom, deg=deg):
            """densify geometry for circle and ellipse (geom) at ``deg`` degree
            increments. deg, angle = (1, 361), (2, 181), (5, 73)
            """
            if 'curve' in geom.JSON:
                return geom.densify('ANGLE', 1, np.deg2rad(deg))
            return geom
        # ----
        null_pnt = (np.nan, np.nan)
        id_len = []
        a_2d = []
        with arcpy.da.SearchCursor(in_fc, ('OID@', 'SHAPE@'), None, SR) as cursor:
            for p_id, row in enumerate(cursor):
                sub = []
                IDs = []
                num_pnts = []
                p_id = row[0]
                geom = row[1]
                prt_cnt = geom.partCount
                p_num = geom.pointCount  # ---- added
                if (prt_cnt == 1) and (p_num <= 4):
                    geom = _densify_curves_(geom, deg=deg)
                for arr in geom:
                    pnts = [[pt.X, pt.Y] if pt else null_pnt for pt in arr]
                    sub.append(np.asarray(pnts))
                    IDs.append(p_id)
                    num_pnts.append(len(pnts))
                part_count = np.arange(prt_cnt)
                #too = np.cumsum(num_pnts)
                result = np.stack((IDs, part_count, num_pnts), axis=-1)
                id_len.append(result)
                a_2d.extend([j for i in sub for j in i])
        # ----
        id_len = np.vstack(id_len)  #np.array(id_len)
        a_2d = np.asarray(a_2d)
        return id_len, a_2d
    #
    # ---- Check and process section ----------------------------------------
    desc = arcpy.da.Describe(in_fc)
    fc_kind = desc['shapeType']
    SR = desc['spatialReference']
    if fc_kind == "Point":
        print(dedent(msg))
        return None
    if fc_kind == "Multipoint":
        id_len, a_2d = _multipnt_(in_fc, SR)
    else:
        id_len, a_2d = _polytypes_(in_fc, SR)
    # ---- Return and send out
    ids = id_len[:, 0]
    too = np.cumsum(id_len[:, 2])
    frum = np.concatenate(([0], too))
    from_to = np.array(list(zip(frum, too)))
    IFT = np.c_[ids, from_to]
    id_len2 = np.hstack((id_len, IFT[:, 1:]))
    dt = np.dtype({'names':['IDs', 'Part', 'Points', 'From_pnt', 'To_pnt'],
                   'formats': ['i4', 'i4', 'i4', 'i4', 'i4']})
    IFT_2 = uts(id_len2, dtype=dt)
    return a_2d, IFT, IFT_2


def array_poly(a, p_type=None, sr=None, IFT=None):
    """See fc_npGeo for details"""
    def _arr_poly_(arr, SR, as_type):
        """Slices the array where nan values appear, splitting them off during
        the process.
        """
        subs = []
        s = np.isnan(arr[:, 0])
        if np.any(s):
            w = np.where(s)[0]
            ss = np.split(arr, w)
            subs = [ss[0]]
            subs.extend(i[1:] for i in ss[1:])
        else:
            subs.append(arr)
        aa = []
        for sub in subs:
            aa.append([arcpy.Point(*pairs) for pairs in sub])
        if as_type == 'POLYGON':
            poly = arcpy.Polygon(arcpy.Array(aa), SR)
        elif as_type == 'POLYLINE':
            poly = arcpy.Polyline(arcpy.Array(aa), SR)
        return poly
    # ----
    ids = IFT[:, 0]
    from_to = IFT[:, 1:]
    chunks = [a[f:t] for f, t in from_to]  # ---- _poly_pieces_ chunks input
    polys = []
    for i in chunks:
        p = _arr_poly_(i, sr, p_type)  # ---- _arr_poly_ makes parts of chunks
        polys.append(p)
    out = list(zip(polys, ids))
    return out


def geometry_fc(a, IFT, p_type=None, gdb=None, fname=None, sr=None):
    """See fc_npGeo for details.  The work is done by ``array_poly``.
    """
    if p_type is None:
        p_type = "POLYGON"
    out = array_poly(a, p_type, sr=sr, IFT=IFT)   # call array_poly
    name = gdb + "\\" + fname
    wkspace = arcpy.env.workspace = 'memory'  # legacy is in_memory
    arcpy.management.CreateFeatureclass(wkspace, fname, p_type,
                                        spatial_reference=sr)
    arcpy.management.AddField(fname, 'ID_arr', 'LONG')
    with arcpy.da.InsertCursor(fname, ['SHAPE@', 'ID_arr']) as cur:
        for row in out:
            cur.insertRow(row)
    arcpy.management.CopyFeatures(fname, name)


# ============================================================================
# ---- extent_poly section
msg0 = """
Either you failed to specify the geodatabase location and filename properly
or you had flotsam, including spaces, in the path, like...\n
  {}\n
Create a safe path and try again...\n
`Filenames and paths in Python`
https://community.esri.com/blogs/dan_patterson/2016/08/14/filenames-and
-file-paths-in-python.
"""
def check_path(out_fc):
    """Check for a filegeodatabase and a filename"""
    _punc_ = '!"#$%&\'()*+,-;<=>?@[]^`{|}~ '
    flotsam = " ".join([i for i in _punc_]) + " ... plus the `space`"
    msg = msg0.format(flotsam)
    if np.any([i in out_fc for i in _punc_]):
        return (None, msg)
    pth = out_fc.split("\\")
    if len(pth) == 1:
        return (None, msg)
    name = pth[-1]
    gdb = "\\".join(pth[:-1])
    if gdb[-4:] != '.gdb':
        return (None, msg)
    return gdb, name


def extent_poly(in_fc, out_fc, kind):
    """Feature envelop to polygon demo.

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
    """
    result = check_path(out_fc)
    if result[0] is None:
        print(result[1])
        return result[1]
    gdb, name = result
    # ---- done checks       
    SR = getSR(in_fc)
    #data = fc_data(in_fc)
    tmp, IFT, IFT_2 = fc_geometry(in_fc, SR)
    m = np.nanmin(tmp, axis=0)    # shift to bottom left of extent
    info = "extent to polygons"
    a = tmp - m
    g = extGeo(a, IFT=IFT, Kind=kind, Info=info)   # create the geo array
    ext = g.extent_rectangles()   # create the extent array
    ext = ext + m                 # shift back, construct the output features
    ext = _updateGeo(ext, K=kind, id_too=None, Info=None)
    #
    # ---- produce the geometry
    p = "POLYGON"
    if kind == 1:
        p = "POLYLINE"
    geometry_fc(ext, ext.IFT, p_type=p, gdb=gdb, fname=name, sr=SR)
    return "{} completed".format(out_fc)

# ---- demo / tool section --------------------------------------------------
#
if len(sys.argv) == 1:
    testing = True
    in_fc = r"C:\Arc_projects\Free_Tools\Free_tools.gdb\Complex"
    kind = 2
    out_fc = r"C:\Arc_projects\Free_Tools\Free_tools.gdb\ex"
else:
    testing = False
    in_fc = sys.argv[1]
    kind = 2    
    if sys.argv[2] == 'Polyline':
        kind = 1
    out_fc = sys.argv[3]
# ----
result = check_path(out_fc)
if result[0] is None:
    msg = "...\n{}\n...".format(result[1])
    print(msg)
    arcpy.AddMessage(msg)
else:
    gdb, name = result
    if not testing:
        result = extent_poly(in_fc, out_fc, kind)
# ===========================================================================
#
if __name__ == "__main__":
    """optional location for parameters"""

    