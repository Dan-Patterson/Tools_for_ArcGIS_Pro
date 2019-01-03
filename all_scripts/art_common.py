# -*- coding: utf-8 -*-
"""
art_common
===========

Script :   art_common.py

Author :   Dan_Patterson@carleton.ca

Modified : 2018-09-24

Purpose:  tools for working with numpy arrays

art_common is a set of functions common to the implementation of array tools
in testing model

Useage :

References
----------
`<http://pro.arcgis.com/en/pro-app/arcpy/data-access/numpyarraytotable.htm>`_.
`<http://pro.arcgis.com/en/pro-app/arcpy/data-access/tabletonumpyarray.htm>`_.
---------------------------------------------------------------------
"""

import sys
from textwrap import dedent
import numpy as np
from arcpy import AddMessage, ListFields, Raster
from arcpy.da import Describe, TableToNumPyArray
ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


__all__ = ['tweet',
           'de_punc',
           '_describe',
           'fc_info',
           'fld_info',
           'null_dict',
           'tbl_arr'
           ]


def tweet(msg):
    """Print a message for both arcpy and python.

    msg - a text message
    """
    m = "\n{}\n".format(msg)
    AddMessage(m)
    print(m)


def de_punc(s, punc=None, no_spaces=True, char='_'):
    """Remove punctuation and/or spaces in strings and replace with
    underscores or nothing

    Parameters
    ----------
    s : string
        input string to parse
    punc : string
        A string of characters to replace ie. '@ "!\'\\[]'
    no_spaces : boolean
        True, replaces spaces with underscore.  False, leaves spaces
    char : string
        Replacement character
    """
    if (punc is None) or not isinstance(punc, str):
        punc = '!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~'  # _ removed
    if no_spaces:
        punc = " " + punc
    s = "".join([[i, char][i in punc] for i in s])
    return s

def _describe(in_fc):
    """Simply return the arcpy.da.Describe object.

    **desc.keys()** an abbreviated list::

    'OIDFieldName'... 'areaFieldName', 'baseName'... 'catalogPath',
    'dataType'... 'extent', 'featureType', 'fields', 'file'... 'hasM',
    'hasOID', 'hasZ', 'indexes'... 'lengthFieldName'... 'name', 'path',
    'rasterFieldName', ..., 'shapeFieldName', 'shapeType',
    'spatialReference'

    """
    return Describe(in_fc)


def fc_info(in_fc, prn=False):
    """Return basic featureclass information, including the following...

    Returns:
    --------
    shp_fld  :
        field name which contains the geometry object
    oid_fld  :
        the object index/id field name
    SR       :
        spatial reference object (use SR.name to get the name)
    shp_type :
        shape type (Point, Polyline, Polygon, Multipoint, Multipatch)

    Notes:
    ------
    Other useful parameters :
        'areaFieldName', 'baseName', 'catalogPath','featureType',
        'fields', 'hasOID', 'hasM', 'hasZ', 'path'

    Derive all field names :
        all_flds = [i.name for i in desc['fields']]
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
        return None
    return shp_fld, oid_fld, shp_type, SR


def fld_info(in_fc, prn=False):
    """Field information for a featureclass (in_fc).

    Parameters:
    -----------
    prn : boolean
        True - returns the values

        False - simply prints the results

    Field properties:
    -----------------
    'aliasName', 'baseName', 'defaultValue', 'domain', 'editable',
    'isNullable', 'length', 'name', 'precision', 'required', 'scale', 'type'
    """
    flds = ListFields(in_fc)
    f_info = [(i.name, i.type, i.length, i.isNullable, i.required)
              for i in flds]
    f = "{!s:<14}{!s:<12}{!s:>7} {!s:<10}{!s:<10}"
    if prn:
        frmt = "FeatureClass:\n   {}\n".format(in_fc)
        args = ["Name", "Type", "Length", "Nullable", "Required"]
        frmt += f.format(*args) + "\n"
        frmt += "\n".join([f.format(*i) for i in f_info])
        tweet(frmt)
        return None
    return f_info


def null_dict(flds):
    """Produce a null dictionary from a list of fields
    These must be field objects and not just their name.
    """
    dump_flds = ["OBJECTID","Shape_Length", "Shape_Area", "Shape"]
    flds_oth = [f for f in flds
                if f.name not in dump_flds]
#    oid_geom = ['OBJECTID', 'SHAPE@X', 'SHAPE@Y']
    nulls = {'Double':np.nan,
             'Single':np.nan,
             'Short':np.iinfo(np.int16).min,
             'SmallInteger':np.iinfo(np.int16).min,
             'Long':np.iinfo(np.int32).min,
             'Float':np.nan,
             'Integer':np.iinfo(np.int32).min,
             'String':str(None),
             'Text':str(None)}
    fld_dict = {i.name: i.type for i in flds_oth}
    nulls = {f.name:nulls[fld_dict[f.name]] for f in flds_oth}
    return nulls


def tbl_arr(pth):
    """Convert featureclass/table to a structured ndarray

    Requires
    --------
    pth : string
        path to input featureclass or table

    """
    flds = ListFields(pth)
    nulls = null_dict(flds)
    bad = ['OID', 'Geometry', 'Shape_Length', 'Shape_Area']
    f0 = ["OID@"]
    f1 = [i.name for i in flds if i.type not in bad]
    flds = f0 + f1
    a = TableToNumPyArray(pth,
                          field_names=flds,
                          skip_nulls=False,
                          null_value=nulls)
    dt = np.array(a.dtype.descr)
    nmes = dt[:, 0]
    sze = dt[:, 1]
    cleaned = []
    for i in nmes:
        i = de_punc(i)  # run de_punc to remove punctuation
        cleaned.append(i)
    a.dtype = list(zip(cleaned, sze))
    return a

# ---- raster section ----
#
def rasterfile_info(fname, prn=False):
    """Obtain raster stack information from the filename of an image
    :
    """
    #
    frmt = """
    File path   - {}
    Name        - {}
    Spatial Ref - {}
    Raster type - {}
    Integer?    - {}
    NoData      - {}
    Min         - {}
    Max         - {}
    Mean        - {}
    Std dev     - {}
    Bands       - {}
    Cell        - h {}   w {}
    Lower Left  - X {}   Y {}
    Upper Left  - X {}   Y {}
    Extent      - h {}   w {}
    """
    desc = Describe(fname)
    r_data_type = desc.datasetType  # 'RasterDataset'
    args = []
    if r_data_type == 'RasterDataset':
        r = Raster(fname)
        r.catalogPath            # full path name and file name
        pth = r.path             # path only
        name = r.name            # file name
        SR = r.spatialReference
        r_type = r.format        # 'TIFF'
        #
        is_int = r.isInteger
        nodata = r.noDataValue
        r_max = r.maximum
        r_min = r.minimum
        r_mean = "N/A"
        r_std = "N/A"
        if not is_int:
            r_mean = r.mean
            r_std = r.standardDeviation
        bands = r.bandCount
        cell_hght = r.meanCellHeight
        cell_wdth = r.meanCellWidth
        extent = desc.Extent
        LL = extent.lowerLeft  # Point (X, Y, #, #)
        hght = r.height
        wdth = r.width
        UL = r.extent.upperLeft
        args = [pth, name, SR.name, r_type, is_int, nodata, r_min, r_max,
                r_mean, r_std, bands, cell_hght, cell_wdth, LL.X, LL.Y,
                UL.X, UL.Y, hght, wdth]
    if prn:
        tweet(dedent(frmt).format(*args))
    else:
        return args

# ----------------------------------------------------------------------
# .... final code section producing the featureclass and extendtable
if len(sys.argv) == 1:
    testing = True
    # parameters here
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
