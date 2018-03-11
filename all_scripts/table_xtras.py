# -*- coding: UTF-8 -*-
"""
:Script:   .py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-xx-xx
:Purpose:  tools for working with numpy arrays
:Useage:
:
:References:
:
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


def tbl_txt(in_tbl, in_flds=None):
    """Convert a table to text
    :Requires
    :--------
    :  in_tbl - a table from within arcmap
    :  in_flds - either None, a list/tuple of field names.  If None or an
    :            empty list or tuple, then all fields are returned.
    """
    if not isinstance(in_flds, (list, tuple, type(None), "")):
        return "Input is not correct"
    if in_flds is None:
        in_flds = "*"
    elif isinstance(in_flds, (list, tuple)):
        if len(in_flds) == 0:
            in_flds = "*"
    a = arcpy.da.TableToNumPyArray(in_tbl, in_flds)
    return a


def rotate_tbl(a, max_cols=20, line_wdth=79):
    """Rotate a structured array to offer another view of the dsta
    :  Be reasonable... the 1000 record table just isn't going to work.
    :  The maximum number of rows can be specified and the column
    :  widths are determined from the data therein.  Hopefully everything
    :  will fit within the margin width... If not, reduce the number
    :  of rows, or roll your own.
    :
    : max_cols - maximum number of columns to print
    : line_wdth - slice the line after this width
    :
    :Notes:
    : dt = ", ".join(["('C{}', '<U{}')".format(i, j) for i, j in enumerate(w)])
    : w is the widths below and e is the empty object array
    :arcpy.Tabletools.RotateTable("polygon_demo",
    :                          "OBJECTID;file_part;main_part;Test;Pnts;Shape")
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
    txt = "Attribute | Records....\n{}".format(txt)
    tweet(txt)
#    return txt, e, widths

# ---- main section ----
'''
script = sys.argv[0]
if len(sys.argv) > 1:
    in_tbl = sys.argv[1]
    in_flds = sys.argv[2]
#    out_txt = str(sys.argv[3]).replace("\\", "/")
else:
    in_tbl = r"C:\GIS\Tools_scripts\Table_tools\Table_tools.gdb\polygon_demo"
    # in_flds = "OBJECTID, Shape, Id, Area, file_part, X_c"
# in_flds = in_flds.split(";")
a = tbl_txt(in_tbl, in_flds=None)  # in_flds)
# rotate_tbl(a)  # rotate table demo
nms = a.dtype.names
b = np.array([a[i] for i in nms if a[i].dtype.kind == 'i'])  # int fields
'''

#f = 'C:/GIS/Tools_scripts/Data/sample_20.npy'
#f = 'C:/GIS/Tools_scripts/Data/sample_1000.npy'
#f = 'C:/GIS/Tools_scripts/Data/sample_10K.npy'
#f =  'C:/GIS/Tools_scripts/Data/array_100K.npy'
f = 'C:/GIS/Tools_scripts/Data/sample_100K.npy'

a = np.load(f)
nms = a.dtype.names
sze = [i[1] for i in a.dtype.descr]
uni = np.unique(a[['Town', 'County']], return_counts=True)
final = rfn.append_fields(uni[0], names='Count', data=uni[1], usemask=False)
#n_0 = [(nm, a[nm]) for nm in nms if a[nm].dtype.kind == 'i']  # int fields
sums = [(nm, a[nm].sum()) for nm in nms if a[nm].dtype.kind == 'i']  # sums

# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
'''
dt = [('Game1', '<i4'), ('Game2', '<i4'), ('Game3', '<i4'),
       ('Game4', '<i4'), ('Game5', '<i4')]
a = np.array([(2, 6, 5, 2, 2),
              (6, 4, 1, 8, 4),
              (8, 3, 2, 1, 5),
              (4, 9, 4, 7, 9)], dtype= dt)
x.view(np.float64).reshape(len(x), -1))

nms = a.dtype.names
b = [(nm, a[nm]) for nm in nms if a[nm].dtype.kind == 'i']  # int fields
b = [(nm, a[nm].sum()) for nm in nms if a[nm].dtype.kind == 'i']  # sums
b = [(nm, a[nm].size) for nm in nms if a[nm].dtype.kind == 'i']  # size
b = [(nm, a[nm].min()) for nm in nms if a[nm].dtype.kind == 'i']  # min

c = [(nm, a[nm]) for nm in nms if a[nm].dtype.kind == 'f']  # float fields
c = [(nm, a[nm].mean(axis=0)) for nm in nms if a[nm].dtype.kind == 'f']  #mean
c =>
[('Shape', array([ 23.000,  8.788])),
 ('X_min', 19.0),
 ('Y_min', 5.0),
 ('X_max', 27.0),
 ('Y_max', 12.5),
 ('Shape_Length', 39.618033988749893),
 ('Shape_Area', 55.899999999999999)]

or...
txt = "\n".join(["{}: mean= {}".format(*c[i]) for i in range(len(c))])
print(txt)
Shape: mean= [ 23.000  8.788]
X_min: mean= 19.0
Y_min: mean= 5.0
X_max: mean= 27.0
Y_max: mean= 12.5
Shape_Length: mean= 39.61803398874989

Now compare to a['Shape'][:,0].mean() => 23

'''
