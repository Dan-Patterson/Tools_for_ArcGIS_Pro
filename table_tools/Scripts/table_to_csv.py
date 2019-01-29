# -*- coding: utf-8 -*-
"""
table_to_csv
============

Script :   table_to_csv.py

Author :   Dan_Patterson@carleton.ca

Modified : 2018-12-29

Purpose :  tools for working with numpy arrays and geometry

Notes:

References:

"""
import sys
from textwrap import dedent
import numpy as np
import arcpy

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
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
    """Convert a table to text

    Requires:
    --------
    in_tbl : table
        a table from within arcmap
    in_flds : either None, a list/tuple of field names.
        If None or an empty list or tuple, then all fields are returned.
    """
    flds = arcpy.ListFields(in_tbl)
    nulls = null_dict(flds)
    if not isinstance(in_flds, (list, tuple, type(None), "")):
        return "Input is not correct"
    if in_flds is None:
        in_flds = "*"
    elif isinstance(in_flds, (list, tuple)):
        if len(in_flds) == 0:
            in_flds = "*"
    a = arcpy.da.TableToNumPyArray(in_tbl,
                                   in_flds,
                                   skip_nulls=False,
                                   null_value=nulls)
    return a


# ----------------------------------------------------------------------
# (3) save_txt .... code section ---
def save_txt(a, name="arr.txt", sep=", ", dt_hdr=True):
    """Save a NumPy structured, recarray to text.

    Requires:
    --------
    a     : array
        input array
    fname : filename
        output filename and path otherwise save to script folder
    sep   : separator
        column separater, include a space if needed
    dt_hdr: boolean
        if True, add dtype names to the header of the file
    """
    a_names = ", ".join(i for i in a.dtype.names)
    hdr = ["", a_names][dt_hdr]  # use "" or names from input array
    s = np.array(a.tolist(), dtype=np.unicode_)
    widths = [max([len(i) for i in s[:, j]])
              for j in range(s.shape[1])]
#    frmt = sep.join(["%{}s".format(i) for i in widths])
    frmt = sep.join(["%s" for i in widths])  # stripped out space padding
    # vals = ", ".join([i[1] for i in a.dtype.descr])
    np.savetxt(name, a, fmt=frmt, header=hdr, comments="")
    print("\nFile saved...")


# ---- main section ----
if len(sys.argv) > 1:
    script = sys.argv[0]
    in_tbl = sys.argv[1]
    out_csv = str(sys.argv[2]).replace("\\", "/")
    
    frmt = """\n
    :---------------------------------------------------------------------:
    Running.... {}
    Input table ....... {}
    Output csv file... {}\n
    :---------------------------------------------------------------------:
    """
    args = [script, in_tbl, out_csv]
    msg = dedent(frmt).format(*args)
    tweet("Input parameters {}".format(msg))
    #
    flds = arcpy.ListFields(in_tbl)
    fnames = [i.name for i in flds if i.type not in ('Geometry', 'Raster')]
#    fnames = ";".join([i for i in fnames])
    a = tbl_arr(in_tbl, in_flds=fnames)  # call tble_arr to get array
    #
    save_txt(a, name=out_csv)
# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """

