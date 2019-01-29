# -*- coding: UTF-8 -*-
"""
table_rotate
============

Script :   table_rotate.py

Author :   Dan.Patterson@carleton.ca

Modified : 2018-12-30

Purpose:  tools for working with numpy arrays
Useage:

References:

:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
import numpy as np
import numpy.lib.recfunctions as rfn
import arcpy

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}

np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


def tweet(msg):
    """Produce a message for both arcpy and python
    : msg - a text message
    """
    m = "\n{}\n".format(msg)
    arcpy.AddMessage(m)
    print(m)
    print(arcpy.GetMessages())


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


def tbl_arr(in_tbl, in_flds=None):
    """Convert a table to an array

    Requires:
    --------
    in_tbl : table
        a table from within arcmap
    in_flds : either None, a list/tuple of field names.
        If None or an empty list or tuple, then all fields are returned.
    """
    flds = arcpy.ListFields(in_tbl)
    dump_flds = ["Shape_Length", "Shape_Area", "Shape"]
    flds = [f for f in flds if f.name not in dump_flds]
    in_flds = [f.name for f in flds]
    nulls = null_dict(flds)
    if not isinstance(flds, (list, tuple, type(None), "")):
        return "Input is not correct"
    if flds is None:
        in_flds = "*"
    elif isinstance(in_flds, (list, tuple)):
        if len(in_flds) == 0:
            in_flds = "*"
        else:
            in_flds = [f.name for f in flds]
    a = arcpy.da.TableToNumPyArray(in_tbl,
                                   in_flds,
                                   skip_nulls=False,
                                   null_value=nulls)
    return a


def rotate_tbl(a, max_cols=20, line_wdth=79):
    """Rotate a structured array to offer another view of the data.

    max_cols : integer
        maximum number of columns to print
    line_wdth : integer
        slice the line after this width

    Notes:
    ------
    Be reasonable... the 1000 record table just isn't going to work. The
    maximum number of rows can be specified and the column widths are
    determined from the data therein.  Hopefully everything will fit within
    the margin width... If not, reduce the number of rows, or roll your own.

    dt = ", ".join(["('C{}', '<U{}')".format(i, j) for i, j in enumerate(w)])
    w is the widths below and e is the empty object array
    arcpy.Tabletools.RotateTable("polygon_demo",
                              "OBJECTID;file_part;main_part;Test;Pnts;Shape")
    """
    cut = min(a.shape[0], max_cols)
    rc = (len(a[0]), cut + 1)
    a = a[:cut]
    e = np.empty(rc, dtype=np.object)
    e[:, 0] = a.dtype.names
    types = (list, tuple, np.ndarray)
    u0 = [[[j, 'seq'][isinstance(j, types)] for j in i] for i in a]
    u = np.array(u0, dtype=np.unicode_)
    e[:, 1:] = u[:].T
    widths = [max([len(i) for i in e[:, j]]) for j in range(e.shape[1])]
    f = ["{{!s: <{}}} ".format(width + 1) for width in widths]
    txt = "".join(i for i in f)
    txt = "\n".join([txt.format(*e[i, :])[:line_wdth]
                     for i in range(e.shape[0])])
    hdr_txt = "Attribute | Records....\n{}".format(txt)
    tweet(hdr_txt)
    return txt  #, e, widths

# ---- main section ----

script = sys.argv[0]
if len(sys.argv) > 1:
    in_tbl = sys.argv[1]
    in_flds = sys.argv[2]
    out_txt = sys.argv[3]
    a = tbl_arr(in_tbl, in_flds=None)[:5]  # in_flds)
    txt = rotate_tbl(a)  # rotate table demo
    f = open(out_txt, 'w')
    print(txt, file=f)
    f.close()    
else:
    in_tbl = r"C:\Git_Dan\arraytools\array_tools_testing\array_tools.gdb\pnts_2000"
    a = tbl_arr(in_tbl, in_flds=None)[:5]  # in_flds)
    out_txt = "C:/Temp/rot_.txt"
    txt = rotate_tbl(a)  # rotate table demo
    f = open(out_txt, 'w')
    print(txt, file=f)
    f.close()
# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))

