# -*- coding: UTF-8 -*-
"""
_common.py
==========

Script :   _common.py

Author :   Dan.Patterson@carleton.ca

Modified : 2018-09-13

Purpose :
    Common tools for working with numpy arrays and featureclasses

Requires:
---------
numpy and arcpy

Tools :
-------
    '_describe', '_flatten', 'fc_info', 'flatten_shape', 'fld_info',\
    'pack', 'tweet', 'unpack'

References :

---------------------------------------------------------------------
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

__all__ = ['tweet',
           'de_punc',
           '_describe',
           'fc_info',
           'fld_info',
           'null_dict',
           ]


def tweet(msg):
    """Print a message for both arcpy and python.

    msg - a text message
    """
    m = "\n{}\n".format(msg)
    arcpy.AddMessage(m)
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

# ----------------------------------------------------------------------------
# ---- Geometry objects and generic geometry/featureclass functions ----------
# ----------------------------------------------------------------------------
def _describe(in_fc):
    """Simply return the arcpy.da.Describe object.

    **desc.keys()** an abbreviated list::

    'OIDFieldName'... 'areaFieldName', 'baseName'... 'catalogPath',
    'dataType'... 'extent', 'featureType', 'fields', 'file'... 'hasM',
    'hasOID', 'hasZ', 'indexes'... 'lengthFieldName'... 'name', 'path',
    'rasterFieldName', ..., 'shapeFieldName', 'shapeType',
    'spatialReference'

    """
    return arcpy.da.Describe(in_fc)


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
    flds = arcpy.ListFields(in_fc)
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
    flds = arcpy.ListFields(pth)
    nulls = null_dict(flds)
    bad = ['OID', 'Geometry', 'Shape_Length', 'Shape_Area']
    f0 = ["OID@"]
    f1 = [i.name for i in flds if i.type not in bad]
    flds = f0 + f1
    a = arcpy.da.TableToNumPyArray(pth,
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


def arr_csv(a):
    """Format a structured/recarray to csv format
    """
    pass
# ---- extras ----------------------------------------------------------------



# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally... print the script source name. run the _demo """
#    print("Script... {}".format(script))
