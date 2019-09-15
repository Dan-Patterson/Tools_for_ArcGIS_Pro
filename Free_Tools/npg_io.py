# -*- coding: utf-8 -*-
"""
=========
npg_io.py
=========

Script :
    .../npgeom/npg_io.py

Author :
    Dan_Patterson@carleton.ca

Modified : 2019-09-06
    Creation date during 2019 as part of ``arraytools``.

Purpose : Tools for working with point and poly features as an array class
    Requires npGeo to implement the array geometry class.


See Also
--------

__init__ :
    `__init__.py` has further information on arcpy related functionality.
npGeo :
    A fuller description of the Geo class, its methods and properties is given
    there.  This script focuses on getting arcpy or geojson geometry into
    numpy arrays.

References
----------
**General**

`Subclassing ndarrays
<https://docs.scipy.org/doc/numpy/user/basics.subclassing.html>`_.


"""
# pylint: disable=C0330  # Wrong hanging indentation
# pylint: disable=C0103  # invalid-name
# pylint: disable=E0611  # stifle the arcgisscripting
# pylint: disable=E1101  # ditto for arcpy
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect
# pylint: disable=W0621  # redefining name
# pylint: disable=W0621  # redefining name
import sys
from textwrap import dedent, indent
import json

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured as stu
from numpy.lib.recfunctions import unstructured_to_structured as uts

from npGeo import Geo

import arcpy


__all__ = [
    'poly2array', 'load_geojson',   # shape to array and conversion
    'arrays_to_Geo', 'Geo_to_arrays', 'array_ift',
    '_make_nulls_', 'getSR', 'fc_composition',       # featureclass methods
    'fc_data', 'fc_geometry', 'fc_shapes', 'getSR', 'shape_to_K',
    'array_poly', 'geometry_fc',                     # convert back to fc
    'prn_q', '_check', 'prn_tbl', 'prn_geo'          # printing
    ]

# ---- Constants -------------------------------------------------------------
#
script = sys.argv[0]

FLOATS = np.typecodes['AllFloat']
INTS = np.typecodes['AllInteger']
NUMS = FLOATS + INTS

null_pnt = (np.nan, np.nan)  # ---- a null point

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=160, precision=2, suppress=True,
                    threshold=100, formatter=ft)


# ==== geometry =========================================================
# These are the main geometry to array conversions
#
# ---- for polyline/polygon featureclasses

def poly2array(polys):
    """Convert polyline or polygon shapes to arrays for use in the Geo class.

    Parameters
    ----------
    polys : tuple, list
        Polyline or polygons in a list/tuple
    """
    def _p2p_(poly):
        """Convert a single ``poly`` shape to numpy arrays or object"""
        sub = []
        for arr in poly:
            pnts = [[pt.X, pt.Y] if pt else null_pnt for pt in arr]
            sub.append(np.asarray(pnts))
        return sub
    # ----
    if not isinstance(polys, (list, tuple)):
        polys = [polys]
    out = []
    for poly in polys:
        out.append(_p2p_(poly))
    return out


# ---- json section
#
def load_geojson(pth, full=False, geometry=True):
    """Load a geojson file and convert to a Geo Array.  The geojson is from the
    Features to JSON tool listed in the references.

    Requires
    --------
    pth : file path
        Full file path to the geojson file.
    full : boolean
        True to return a formatted geojson file.
    geometry : boolean
        True returns just the geometry of the file.

    Returns
    -------
    data : dictionary
        The full geojson dictionary of the geometry and its attributes.  The
        result is a nested dictionary::

    >>> data
    ... {'type':
    ...  'crs': {'type': 'name', 'properties': {'name': 'EPSG:2951'}},
    ...  'features': [{'type': 'Feature',
    ...    'id': 1,
    ...    'geometry': {'type':  'MultiPolygon',
    ...     'coordinates': snip},  # coordinate values
    ...     'properties': snip }}, # attribute values from table
    ... {'type': ... repeat}

    geometry : list
        A list of lists representing the features, their parts *for multipart
        features) and inner holes (for polygons).

    References
    ----------
    `geojson specification in detail
    <https://geojson.org/>`_.

    `Features to JSON
    <https://pro.arcgis.com/en/pro-app/tool-reference/conversion/
    features-to-json.htm>`_.

    `JSON to Features
    <https://pro.arcgis.com/en/pro-app/tool-reference/conversion/
    json-to-features.htm>`_.
    """
    # import json
    with open(pth) as f:
        data = json.load(f)
    shapes = data['features']
    coords = [s['geometry']['coordinates'] for s in shapes]
    if full and geometry:
        return data, coords
    if full:
        return data
    if geometry:
        return coords


def geojson_Geo(pth, kind=2):
    """GeoJSON file to Geo array using `array_ift`.

    Parameters
    ----------
    pth : string
        Full path to the geojson file.
    kind : integer
        Polygon, Polyline or Point type are identified as either 2, 1, or 0.
    """
    coords = load_geojson(pth)
    a_2d, IFT = array_ift(coords)
    a_2d = np.vstack(a_2d)
    return Geo(a_2d, IFT, kind)


def fc_json(in_fc, SR=None):
    """Produce arrays from the json representation of fc_shapes shapes.
    """
    shapes = fc_shapes(in_fc, SR=SR)
    arr = []
    json_keys = [i for i in json.loads(shapes[0].JSON).keys()]
    geom_key = json_keys[0]
    for s in shapes:
        arr.append(json.loads(s.JSON)[geom_key])
    return arr


# ===========================================================================
# ---- Geo array construction
#    Construct the Geo array from a list of ndarrays or an ndarray and
#    deconstruct, the Geo array back to its origins
#
def arrays_to_Geo(in_arrays, Kind=2, Info=None):
    """Produce a Geo class object from a list/tuple of arrays.

    Parameters
    ----------
    in_arrays : list
        ``in_arrays`` can be created by adding existing 2D arrays to the list
        or produced from the conversion of poly features to arrays using
        ``poly2arrays``.
    Kind : integer
        Points (0), polylines (1) or polygons (2)

    Requires
    --------
    npg_io.array_ift

    Returns
    -------
    A ``Geo`` class object based on a 2D np.ndarray (a_2d) with an array of
    indices (IFT) delineating geometry from-to points for each shape and its
    parts.

    See Also
    --------
    **fc_geometry** to produce ``Geo`` objects directly from arcgis pro
    featureclasses.
    """
    a_2d, IFT = array_ift(in_arrays)     # ---- call npg_io.array_ift
    return Geo(a_2d, IFT, Kind, Info)


def Geo_to_arrays(in_geo):
    """Reconstruct the input arrays from the Geo array."""
    return np.asarray([np.asarray(in_geo.get(i))
                       for i in np.unique(in_geo.IDs).tolist()])


# ---- produce the stacked array and IFT values from the output of poly2array
#  or from the output of load_geojson
#
def array_ift(in_arrays):
    """Produce a 2D array stack and a I(d) F(rom) T(o) list of the coordinate
    pairs in the resultant.

    Parameters
    ----------
    in_arrays : list, array
        ``in_arrays`` can be created by adding existing 2D arrays to the list
         or produced from the conversion of poly features to arrays using
        ``poly2arrays``.

    Notes
    -----
    Called by ``npGeo_io.arrays_to_Geo``.
    Use ``fc_geometry`` to produce ``Geo`` objects directly from arcgis pro
    featureclasses.
    """
    null_pnt = np.array([[np.nan, np.nan]])
    id_too = []
    a_2d = []
    if isinstance(in_arrays, np.ndarray):
        if in_arrays.ndim == 2:
            in_arrays = [in_arrays]
    for cnt, p in enumerate(in_arrays):
        p = np.asarray(p)
        kind = p.dtype.kind
        if kind == 'O':
            bits = []
            for j in p:
                for i in j:
                    bits.append(np.asarray(i))
                    bits.append(null_pnt)
                bits = bits[:-1]
                stack = np.vstack(bits)
                id_too.append([cnt, len(stack)])
            sub = stack
        elif kind in NUMS:
            sub = []
            if len(p.shape) == 2:
                id_too.append([cnt, len(p)])
                sub.append(np.asarray(p))
            elif len(p.shape) == 3:
                id_too.extend([[cnt, len(k)] for k in p])
                sub.append([np.asarray(j) for i in p for j in i])
        subs = np.vstack(sub)
        a_2d.append(subs)
    a_2d = np.vstack(a_2d)
    id_too = np.array(id_too)
    ids = id_too[:, 0]
    too = np.cumsum(id_too[:, 1])
    frum = np.concatenate(([0], too))
    IFT = np.array(list(zip(ids, frum, too)))
    return a_2d, IFT


# ===========================================================================
# ---- featureclass section, arcpy dependent via arcgisscripting
#
def _make_nulls_(in_fc, include_oid=True, int_null=-999):
    """Return null values for a list of fields objects, excluding objectid
    and geometry related fields.  Throw in whatever else you want.

    Parameters
    ----------
    in_fc : featureclass or featureclass table
        Uses arcpy.ListFields to get a list of featureclass/table fields.
    int_null : integer
        A default to use for integer nulls since there is no ``nan`` equivalent
        Other options include

    >>> np.iinfo(np.int32).min # -2147483648
    >>> np.iinfo(np.int16).min # -32768
    >>> np.iinfo(np.int8).min  # -128

    >>> [i for i in cur.__iter__()]
    >>> [[j if j else -999 for j in i] for i in cur.__iter__() ]
    """
    nulls = {'Double': np.nan, 'Single': np.nan, 'Float': np.nan,
             'Short': int_null, 'SmallInteger': int_null, 'Long': int_null,
             'Integer': int_null, 'String': str(None), 'Text': str(None),
             'Date': np.datetime64('NaT'), 'Geometry': np.nan}
    #
    desc = arcpy.da.Describe(in_fc)
    if desc['dataType'] not in ('FeatureClass', 'Table'):
        print("Only Featureclasses and tables are supported")
        return None, None
    in_flds = desc['fields']
    good = [f for f in in_flds if f.editable and f.type != 'Geometry']
    fld_dict = {f.name: f.type for f in good}
    fld_names = list(fld_dict.keys())
    null_dict = {f: nulls[fld_dict[f]] for f in fld_names}
    # ---- insert the OBJECTID field
    if include_oid and desc['hasOID']:
        oid_name = 'OID@'  # desc['OIDFieldName']
        oi = {oid_name: -999}
        null_dict = dict(list(oi.items()) + list(null_dict.items()))
        fld_names.insert(0, oid_name)
    return null_dict, fld_names


def getSR(in_fc, verbose=False):
    """Return the spatial reference of a featureclass"""
    desc = arcpy.da.Describe(in_fc)
    SR = desc['spatialReference']
    if verbose:
        print("SR name: {}  factory code: {}".format(SR.name, SR.factoryCode))
    return SR


def shape_to_K(in_fc):
    """The shape type represented by the featureclass"""
    desc = arcpy.da.Describe(in_fc)
    s = desc['shapeType']
    if s == 'Polygon':
        return 2
    if s == 'Polyline':
        return 1
    if s in ('Point', 'Multipoint'):
        return 0


def fc_composition(in_fc, SR=None, prn=True, start=0, end=50):
    """Featureclass geometry composition in terms of shapes, shape parts, and
    point counts for each part.
    """
    if SR is None:
        SR = getSR(in_fc)
    with arcpy.da.SearchCursor(
            in_fc, ['OID@', 'SHAPE@'], spatial_reference=SR) as cur:
        len_lst = []
        for _, row in enumerate(cur):
            p_id = row[0]
            p = row[1]
            parts = p.partCount
            num_pnts = np.asarray([p[i].count for i in range(parts)])
            IDs = np.repeat(p_id, parts)
            part_count = np.arange(parts)
            too = np.cumsum(num_pnts)
            result = np.stack((IDs, part_count, num_pnts, too), axis=-1)
            len_lst.append(result)
    tmp = np.vstack(len_lst)
    too = np.cumsum(tmp[:, 2])
    frum = np.concatenate(([0], too))
    frum_too = np.array(list(zip(frum, too)))
    fc_comp = np.hstack((tmp[:, :3], frum_too))  # axis=0)
    dt = np.dtype({'names': ['IDs', 'Part', 'Points', 'From_pnt', 'To_pnt'],
                   'formats': ['i4', 'i4', 'i4', 'i4', 'i4']})
    fc = uts(fc_comp, dtype=dt)
    frmt = "\nFeatureclass...  {}" + \
        "\nShapes :{:>5.0f}\nParts  :{:>5.0f}\n  max  :{:>5.0f}" + \
        "\nPoints :{:>5.0f}\n  min  :{:>5.0f}\n  med  :{:>5.0f}" + \
        "\n  max  :{:>5.0f}"
    if prn:  # ':>{}.0f
        uni, cnts = np.unique(fc['IDs'], return_counts=True)
        a0, a1 = [fc['Part'] + 1, fc['Points']]
        args = [in_fc, len(uni), np.sum(cnts), np.max(a0),
                np.sum(a1), np.min(a1), int(np.median(a1)), np.max(a1)]
        msg = dedent(frmt).format(*args)
        print(msg)
        # ---- to structured and print
        frmt = "{:>8} "*5
        start, end = sorted([abs(int(i)) if isinstance(i, (int, float))
                             else 0 for i in [start, end]])
        end = min([fc.shape[0], end])
        print(frmt.format(*fc.dtype.names))
        for i in range(start, end):
            print(frmt.format(*fc[i]))
        return None
    return fc


# ===========================================================================
# ---- Create inputs for the Geo class
#  fc_data - just the data
#  fc_geometry - used to create the Geo class
#  fc_shapes - returns an object array (usually) of the geometry

def fc_data(in_fc):
    """Pull all editable attributes from a featureclass tables.  During the
    process, <null> values are changed to an appropriate type.

    Parameters
    ----------
    in_fc : text
        Path to the input featureclass

    Notes
    -----
    The output objectid and geometry fields are renamed to
    `OID_`, `X_cent`, `Y_cent`, where the latter two are the centroid values.
    """
    flds = ['OID@', 'SHAPE@X', 'SHAPE@Y']
    null_dict, fld_names = _make_nulls_(in_fc, include_oid=True, int_null=-999)
    if flds not in fld_names:
        new_names = out_flds = fld_names
    if fld_names[0] == 'OID@':
        out_flds = flds + fld_names[1:]
        new_names = ['OID_', 'X_cent', 'Y_cent'] + out_flds[3:]
    a = arcpy.da.FeatureClassToNumPyArray(in_fc, out_flds, skip_nulls=False,
                                          null_value=null_dict)
    a.dtype.names = new_names
    return np.asarray(a)


def fc_geometry(in_fc, SR=None, IFT_rec=False, true_curves=False, deg=5):
    """Derive, arcpy geometry objects from a FeatureClass searchcursor.

    Parameters
    ----------
    in_fc : text
        Path to the input featureclass.  Points not supported.
    SR : spatial reference
        Spatial reference object, name or id
    deg : integer
        Used to densify curves found for circles and ellipses. Values of
        1, 2, 5 and 10 deg(rees) are appropriate.  No error checking
    IFT_rec : boolean
        Return the ``IFT`` as a structured array as well.

    Returns
    -------
    ``a_2d, IFT`` (ids_from_to), where a_2d are the points as a 2D array,
    ``IFT``represent the id numbers (which are repeated for multipart shapes),
    and the from-to pairs of the feature parts.

    See Also
    --------
    Use ``array_ift`` to produce ``Geo`` objects directly pre-existing arrays,
    or arrays derived form existing arcpy poly objects which originated from
    esri featureclasses.

    Notes
    -----
    Multipoint, polylines and polygons and its variants are supported.

    **Point and Multipoint featureclasses**

    >>> cent = arcpy.da.FeatureClassToNumPyArray(pnt_fc,
                                             ['OID@', 'SHAPE@X', 'SHAPE@Y'])

    For multipoints, use

    >>> allpnts = arcpy.da.FeatureClassToNumPyArray(multipnt_fc,
                            ['OID@', 'SHAPE@X', 'SHAPE@Y'],
                            explode_to_points=True)

    **IFT array structure**

    To see the ``IFT`` output as a structured array, use the following.

    >>> dt = np.dtype({'names': ['ID', 'From', 'To'], 'formats': ['<i4']*3})
    >>> z = IFT.view(dtype=dt).squeeze()
    >>> prn_tbl(z)  To see the output in tabular form

    **Flatten geometry tests**

    >>> %timeit fc_geometry(in_fc2, SR)
    105 ms ± 1.04 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    ...
    >>> cur = arcpy.da.SearchCursor(in_fc, 'SHAPE@', None, SR)
    >>> polys = [row[0] for row in cur]
    >>> pts = [[(i.X, i.Y) if i else (np.nan, np.nan)
                for i in itertools.chain.from_iterable(shp)]
                for shp in polys]
    7.28 ms ± 105 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    """
    msg = """
    Use arcpy.FeatureClassToNumPyArray for Point files.
    MultiPoint, Polyline and Polygons and its variants are supported.
    """

    def _multipnt_(in_fc, SR):
        """Convert multipoint geometry to array"""
        pnts = arcpy.da.FeatureClassToNumPyArray(
                in_fc, ['OID@', 'SHAPE@X', 'SHAPE@Y'], spatial_reference=SR,
                explode_to_points=True
                )
        id_len = np.vstack(np.unique(pnts['OID@'], return_counts=True)).T
        a_2d = stu(pnts[['SHAPE@X', 'SHAPE@Y']])  # ---- use ``stu`` to convert
        return id_len, a_2d

    def _polytypes_(in_fc, SR, true_curves, deg):
        """Convert polylines/polygons geometry to array.

        >>> cur = arcpy.da.SearchCursor( in_fc, ('OID@', 'SHAPE@'), None, SR)
        >>> ids = [r[0] for r in cur]
        >>> arrs = [[j for j in r[1]] for r in cur]
        """
        def _densify_curves_(geom, deg=deg):
            """Densify geometry for circle and ellipse (geom) at ``deg`` degree
            increments. deg, angle = (1, 361), (2, 181), (5, 73)
            """
            if 'curve' in geom.JSON:
                return geom.densify('ANGLE', 1, np.deg2rad(deg))
            return geom
        # ----
        null_pnt = (np.nan, np.nan)
        id_len = []
        a_2d = []
        with arcpy.da.SearchCursor(
                in_fc, ('OID@', 'SHAPE@'), None, SR) as cursor:
            for row in cursor:
                sub = []
                IDs = []
                num_pnts = []
                p_id = row[0]
                geom = row[1]
                prt_cnt = geom.partCount
                if true_curves:
                    p_num = geom.pointCount  # ---- added
                    if (prt_cnt == 1) and (p_num <= 4):
                        geom = _densify_curves_(geom, deg=deg)
                for arr in geom:
                    pnts = [[pt.X, pt.Y] if pt else null_pnt for pt in arr]
                    sub.append(np.asarray(pnts))
                    IDs.append(p_id)
                    num_pnts.append(len(pnts))
                part_count = np.arange(prt_cnt)
                result = np.stack((IDs, part_count, num_pnts), axis=-1)
                id_len.append(result)
                a_2d.extend([j for i in sub for j in i])
        # ----
        id_len = np.concatenate(id_len, axis=0)
        a_2d = np.asarray(a_2d)
        return id_len, a_2d
    #
    # ---- Check and process section ----------------------------------------
    desc = arcpy.da.Describe(in_fc)
    fc_kind = desc['shapeType']
    SR = desc['spatialReference']
    if fc_kind == "Point":
        print(dedent(msg))
        return None, None
    if fc_kind == "Multipoint":
        id_len, a_2d = _multipnt_(in_fc, SR)
    else:
        id_len, a_2d = _polytypes_(in_fc, SR, true_curves, deg)
    # ---- Return and send out
    ids = id_len[:, 0]
    too = np.cumsum(id_len[:, 2])
    frum = np.concatenate(([0], too))
    from_to = np.concatenate((frum[:-1, None], too[:, None]), axis=1)
    IFT = np.concatenate((ids[:, None], from_to), axis=1)
    if IFT_rec:
        id_len2 = np.concatenate((id_len, IFT[:, 1:]), axis=1)
        dt = np.dtype(
                {'names': ['IDs', 'Part', 'Points', 'From_pnt', 'To_pnt'],
                 'formats': ['i4', 'i4', 'i4', 'i4', 'i4']})
        IFT_2 = uts(id_len2, dtype=dt)
        return a_2d, IFT, IFT_2
    return a_2d, IFT


def fc_shapes(in_fc, SR=None):
    """Featureclass to arcpy shapes.  Returns polygon, polyline, multipoint,
    or points.
    """
    if SR is None:
        SR = getSR(in_fc)
    with arcpy.da.SearchCursor(in_fc, 'SHAPE@', spatial_reference=SR) as cur:
        out = [row[0] for row in cur]
    return out


# ===========================================================================
# ---- back to featureclass
#
def array_poly(a, p_type=None, sr=None, IFT=None):
    """
    Used by ``geometry_fc`` to assemble the poly features from array(s).
    This can be used separately.

    Parameters
    ----------
    a : array
        Points array.
    p_type : text
        POLYGON or POLYLINE
    sr : spatial reference
        Spatial reference object, name or id.
    IFT : array
        An Nx3 array consisting of I(d)F(rom)T(o) points.

    Notes
    -----
    Polyline or polygon features can be created from the array data.  The
    features can be multipart with or without interior rings.

    Outer rings are ordered clockwise, inner rings (holes) are ordered
    counterclockwise.  For polylines, there is no concept of order.
    Splitting is modelled after _nan_split_(arr).
    """
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
        if as_type.upper() == 'POLYGON':
            poly = arcpy.Polygon(arcpy.Array(aa), SR)
        elif as_type.upper() == 'POLYLINE':
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
    """Reform poly features from the list of arrays created by ``fc_geometry``.

    Parameters
    ----------
    a : array or list of arrays
        Some can be object arrays, normally created by ``pnts_arr``
    IFT : list/array
        Identifies which feature each input belongs to.  This enables one to
        account for multipart shapes
    p_type : string
        Uppercase geometry type eg POLYGON.
    gdb : text
        Geodatabase path and name.
    fname : text
        Featureclass name.
    sr : spatial reference
        name or object

    Returns
    -------
    Singlepart and/or multipart featureclasses.

    Notes
    -----
    The work is done by ``array_poly``.
    """
    if p_type is None:
        p_type = "POLYGON"
    out = array_poly(a, p_type.upper(), sr=sr, IFT=IFT)   # call array_poly
    name = gdb + "/" + fname
    wkspace = arcpy.env.workspace = 'memory'  # legacy is in_memory
    arcpy.management.CreateFeatureclass(wkspace, fname, p_type,
                                        spatial_reference=sr)
    arcpy.management.AddField(fname, 'ID_arr', 'LONG')
    with arcpy.da.InsertCursor(fname, ['SHAPE@', 'ID_arr']) as cur:
        for row in out:
            cur.insertRow(row)
    arcpy.management.CopyFeatures(fname, name)
    return "geometry_fc complete"

#
# ============================================================================
# ---- array dependent


def prn_q(a, edges=3, max_lines=25, width=120, decimals=2):
    """Format a structured array by setting the width so it hopefully wraps.
    """
    width = min(len(str(a[0])), width)
    with np.printoptions(edgeitems=edges, threshold=max_lines, linewidth=width,
                         precision=decimals, suppress=True, nanstr='-n-'):
        print("\nArray fields/values...:")
        print("  ".join([n for n in a.dtype.names]))
        print(a)


# ---- printing based on arraytools.frmts.py using prn_rec and dependencies
#
def _check(a):
    """Check dtype and max value for formatting information"""
    return a.shape, a.ndim, a.dtype.kind, np.nanmin(a), np.nanmax(a)


def prn_tbl(a, rows_m=20, names=None, deci=2, width=100):
    """Format a structured array with a mixed dtype.  Derived from
    arraytools.frmts and the prn_rec function therein.

    Parameters
    ----------
    a : array
        A structured/recarray
    rows_m : integer
        The maximum number of rows to print.  If rows_m=10, the top 5 and
        bottom 5 will be printed.
    names : list/tuple or None
        Column names to print, or all if None.
    deci : int
        The number of decimal places to print for all floating point columns.
    width : int
        Print width in characters
    """
    def _ckw_(a, name, deci):
        """array `a` c(olumns) k(ind) and w(idth)"""
        c_kind = a.dtype.kind
        if (c_kind in FLOATS) and (deci != 0):  # float with decimals
            c_max, c_min = np.round([np.nanmin(a), np.nanmax(a)], deci)
            c_width = len(max(str(c_min), str(c_max), key=len))
        elif c_kind in NUMS:      # int, unsigned int, float wih no decimals
            c_width = len(max(str(np.nanmin(a)), str(np.nanmax(a)), key=len))
        elif c_kind in ('U', 'S', 's'):
            c_width = len(max(a, key=len))
        else:
            c_width = len(str(a))
        c_width = max(len(name), c_width) + deci
        return [c_kind, c_width]

    def _col_format(pairs, deci):
        """Assemble the column format"""
        form_width = []
        dts = []
        for c_kind, c_width in pairs:
            if c_kind in INTS:  # ---- integer type
                c_format = ':>{}.0f'.format(c_width)
            elif c_kind in FLOATS:  # and np.isscalar(c[0]):  # float rounded
                c_format = ':>{}.{}f'.format(c_width, deci)
            else:
                c_format = "!s:<{}".format(c_width)
            dts.append(c_format)
            form_width.append(c_width)
        return dts, form_width
    # ----
    dtype_names = a.dtype.names
    if dtype_names is None:
        print("Structured/recarray required")
        return None
    if names is None:
        names = dtype_names
    # ---- slice off excess rows, stack upper and lower slice using rows_m
    if a.shape[0] > rows_m*2:
        a = np.hstack((a[:rows_m], a[-rows_m:]))
    # ---- get the column formats from ... _ckw_ and _col_format ----
    pairs = [_ckw_(a[name], name, deci) for name in names]  # -- column info
    dts, wdths = _col_format(pairs, deci)                   # format column
    # ---- slice off excess columns
    c_sum = np.cumsum(wdths)               # -- determine where to slice
    N = len(np.where(c_sum < width)[0])    # columns that exceed ``width``
    a = a[list(names[:N])]
    # ---- Assemble the formats and print
    tail = ['', ' ...'][N < len(names)]
    row_frmt = "  ".join([('{' + i + '}') for i in dts[:N]])
    hdr = ["!s:<" + "{}".format(wdths[i]) for i in range(N)]
    hdr2 = "  ".join(["{" + hdr[i] + "}" for i in range(N)])
    header = " ... " + hdr2.format(*names[:N]) + tail
    header = "{}\n{}".format(header, "-"*len(header))
    txt = [header]
    for idx, i in enumerate(range(a.shape[0])):
        if idx == rows_m:
            txt.append("...")
        else:
            t = " {:>03.0f} ".format(idx) + row_frmt.format(*a[i]) + tail
            txt.append(t)
    msg = "\n".join([i for i in txt])
    print(msg)
    # return row_frmt, hdr2  # uncomment for testing


def prn_geo(a, rows_m=100, names=None, deci=2, width=100):
    """Format a structured array with a mixed dtype.  Derived from
    arraytools.frmts and the prn_rec function therein.

    Parameters
    ----------
    a : array
        A structured/recarray.
    rows_m : integer
        The maximum number of rows to print.  If rows_m=10, the top 5 and
        bottom 5 will be printed.
    names : list/tuple or None
        Column names to print, or all if None.
    deci : int
        The number of decimal places to print for all floating point columns.
    width : int
        Print width in characters.

    Notes
    -----
    >>> toos = s0.IFT[:,2]
    >>> nans = np.where(np.isnan(s0[:,0]))[0]  # array([10, 21, 31, 41]...
    >>> dn = np.digitize(nans, too)            # array([1, 2, 3, 4]...
    >>> ift[:, 0][dn]                          # array([1, 1, 2, 2])
    >>> np.sort(np.concatenate((too, nans)))
    ... array([ 5, 10, 16, 21, 26, 31, 36, 41, 48, 57, 65], dtype=int64)
    """
    def _ckw_(a, name, deci):
        """columns `a` kind and width"""
        c_kind = a.dtype.kind
        if (c_kind in FLOATS) and (deci != 0):  # float with decimals
            c_max, c_min = np.round([np.nanmin(a), np.nanmax(a)], deci)
            c_width = len(max(str(c_min), str(c_max), key=len))
        elif c_kind in NUMS:      # int, unsigned int, float wih no decimals
            c_width = len(max(str(np.nanmin(a)), str(np.nanmax(a)), key=len))
        else:
            c_width = len(name)
        c_width = max(len(name), c_width) + deci
        return [c_kind, c_width]

    def _col_format(pairs, deci):
        """Assemble the column format"""
        form_width = []
        dts = []
        for c_kind, c_width in pairs:
            if c_kind in INTS:  # ---- integer type
                c_format = ':>{}.0f'.format(c_width)
            elif c_kind in FLOATS:  # and np.isscalar(c[0]):  # float rounded
                c_format = ':>{}.{}f'.format(c_width, deci[-1])
            else:
                c_format = "!s:^{}".format(c_width)
            dts.append(c_format)
            form_width.append(c_width)
        return dts, form_width
    # ----
    if names is None:
        names = ['shape', 'part', 'X', 'Y']
    # ---- slice off excess rows, stack upper and lower slice using rows_m
    if not hasattr(a, 'IFT'):
        print("Requires a Geo array")
        return None
    ift = a.IFT
    c = [np.repeat(ift[i, 0], ift[i, 2] - ift[i, 1])
         for i, p in enumerate(ift[:, 0])]
    c = np.concatenate(c)
    # ---- p: __ shape end, p0: x parts, p1: o start of parts, pp: concatenate
    p = np.where(np.diff(c, append=0) == 1, "___", "")
    p0 = np.where(np.isnan(a[:, 0]), "x", "")
    p1 = np.asarray(["" if i not in ift[:, 2] else 'o' for i in range(len(p))])
    pp = np.asarray([p[i]+p0[i]+p1[i] for i in range(len(p))])
    if a.shape[0] > rows_m:
        a = a[:rows_m]
        c = c[:rows_m]
        p = p[:rows_m]
    # ---- get the column formats from ... _ckw_ and _col_format ----
    deci = [0, 0, deci, deci]
    flds = [c, pp, a[:, 0], a[:, 1]]
    pairs = [_ckw_(flds[n], names[n], deci[n])
             for n, name in enumerate(names)]  # -- column info
    dts, wdths = _col_format(pairs, deci)      # format column
    # ---- slice off excess columns
    c_sum = np.cumsum(wdths)               # -- determine where to slice the
    N = len(np.where(c_sum < width)[0])    # columns that exceed ``width``
    # ---- Assemble the formats and print
    row_frmt = " {:>03.0f} " + "  ".join([('{' + i + '}') for i in dts[:N]])
    hdr = ["!s:<" + "{}".format(wdths[i]) for i in range(N)]
    hdr2 = "  ".join(["{" + hdr[i] + "}" for i in range(N)])
    header = " pnt " + hdr2.format(*names[:N])
    header = "\n{}\n{}".format(header, "-"*len(header))
    txt = [header]
    for i in range(a.shape[0]):
        txt.append(row_frmt.format(i, c[i], pp[i], a[i, 0], a[i, 1]))
    msg = "\n".join([i for i in txt])
    print(msg)
    # return row_frmt, hdr2  # uncomment for testing


# ==== Extras ===============================================================
#
def shape_properties(a_shape, prn=True):
    """Get some basic shape geometry properties
    """
    coords = a_shape.__geo_interface__['coordinates']
    sr = a_shape.spatialReference
    props = ['type', 'isMultipart', 'partCount', 'pointCount', 'area',
             'length', 'length3D', 'centroid', 'trueCentroid', 'firstPoint',
             'lastPoint', 'labelPoint']
    props2 = [['Name', sr.name], ['Factory code', sr.factoryCode]]
    t = "\n".join(["{!s:<12}: {}".format(i, a_shape.__getattribute__(i))
                   for i in props])
    t = t + "\n" + "\n".join(["{!s:<12}: {}".format(*i) for i in props2])
    tc = '{!r:}'.format(np.array(coords))
    tt = t + "\nCoordinates\n" + indent(tc, '....')
    if prn:
        print(tt)
    else:
        return tt


def gms(arr):
    """Get maximum dimensions in a list/array

    Returns
    -------
    A list with the format - [3, 2, 4, 10, 2]. Representing the maximum
    expected value in each column::
      [ID, parts, pieces, points, pair]
    """
    from collections import defaultdict

    def get_dimensions(arr, level=0):
        yield level, len(arr)
        try:
            for row in arr:
                yield from get_dimensions(row, level + 1)
        except TypeError:  # not an iterable
            pass
    # ----
    dimensions = defaultdict(int)
    for level, length in get_dimensions(arr):
        dimensions[level] = max(dimensions[level], length)
    return [value for _, value in sorted(dimensions.items())]


def flatten_to_points(iterable):
    """Iteratively flattens an iterable containing potentially nested points
    down to X,Y pairs with feature ID, part, subpart/ring and point number.

    Requires
    --------
    iterable : list/array
        See notes

    Returns
    -------
    A structured array of coordinate geometry information as described by the
    array dtype.

    Notes
    -----
    `load_geojson`'s `coords` output is suitable for input or any ndarray or
    object array representing geometry coordinates.

    I added a x[0] check to try to prevent the flattening of the
    yield to beyond the final coordinate pair.

    References
    ----------
    `Stefan Pochmann on flatten nested lists with indices
    <https://stackoverflow.com/questions/48996063/python-flatten-nested-
    lists-with-indices>`_.
    """
    def gen(iterable):
        """Generator function to acquire the values."""
        stack = [None, enumerate(iterable)]
        pad = -1
        N = 6
        while stack:
            for stack[-2], x in stack[-1]:
                if isinstance(x[0], list):  # added [0] to check for pair
                    stack += None, enumerate(x)
                else:
                    z = [*x, *stack[::2]]
                    if len(z) < N:
                        z.extend([pad]*(N-len(z)))
                    yield z
                break
            else:
                del stack[-2:]

    # ----
    z = gen(iterable)
    dt = np.dtype({'names': ['Xs', 'Ys', 'a', 'b', 'c', 'd'],
                   'formats': ['<f8', '<f8', '<i4', '<i4', '<i4', '<i4']})
    z0 = np.vstack(list(z))
    return uts(z0, dtype=dt)


# ===========================================================================
# ---- main section
if __name__ == "__main__":
    """optional location for parameters"""
    print("\n{}".format(script))

"""
lists to dictionary

list1 =  [('84116', 1750),('84116', 1774),('84116', 1783),('84116',1792)]
list2 = [('84116', 1783),('84116', 1792),('84116', 1847),('84116', 1852),
         ('84116', 1853)]
Lst12 = list1 + list2
dt = [('Keys', 'U8'), ('Vals', '<i4')]
arr = np.asarray((list1 + list2), dtype=dt)
a0 =np.unique(arr)
k = np.unique(arr['Keys'])
{i : a0['Vals'][a0['Keys'] == i].tolist() for i in k}

"""
