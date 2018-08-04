# -*- coding: UTF-8 -*-
"""
:Script:   table2numpyarray.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2018-03-18
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
import arcpy

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


def tweet(msg):
    """Print a message for both arcpy and python.
    : msg - a text message
    """
    m = "\n{}\n".format(msg)
    arcpy.AddMessage(m)
    print(m)


# ---- Run options: _demo or from _tool
#
def _demo():
    """Code to run if in demo mode
    """
    a = np.array(['1, 2, 3, 4, 5', 'a, b, c', '6, 7, 8, 9',
                  'd, e, f, g, h', '10, 11, 12, 13'])
    return a


def _tool():
    """run when script is from a tool
    """
    in_tbl = sys.argv[1]
    in_flds = sys.argv[2]
    out_folder = sys.argv[3]  # output folder name
    out_filename = sys.argv[4]
    out_name = "\\".join([out_folder, out_filename])
    # ---- main tool section
    desc = arcpy.da.Describe(in_tbl)
    args = [in_tbl, in_flds, out_name]
    msg = "Input table.. {}\nfields...\n{}\nOutput arr  {}".format(*args)
    tweet(msg)
    #
    # ---- call section for processing function
    #
    oid = 'OBJECTID'
    in_flds = in_flds.split(";")
    if oid in in_flds:
        vals = in_flds
    else:
        vals = [oid] + in_flds
    #
    # ---- create the field dictionary
    f_info = np.array([[i.name, i.type] for i in arcpy.ListFields(in_tbl)])
    f_dict = {'OBJECTID': -1}
    for f in in_flds:
        if f in f_info[:, 0]:
            n, t = f_info[f_info[:, 0] == f][0]
            if t in ('Integer', 'Short', 'Long'):
                t = np.iinfo(np.int32).min
            elif t in ('Double', 'Float'):
                t = np.nan
            elif t in ('String', 'Text'):
                t = np.unicode_(None)
            else:
                t = np.iinfo(np.int32).min
            f_dict[n] = t
    # ---- where_clause= skip_nulls=  null_value=)
    arr = arcpy.da.TableToNumPyArray(in_tbl, vals, "#", False, f_dict)
    #
    np.save(out_name, arr)


# ----------------------------------------------------------------------
# .... final code section producing the featureclass and extendtable
if len(sys.argv) == 1:
    testing = True
    arrs = _demo()
    frmt = "Result...\n{}"
    print(frmt.format(arrs))
else:
    testing = False
    _tool()
#
if not testing:
    print('Demo done...')


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
    a = _demo()

#in_tbl = 'C:\\GIS\\A_Tools_scripts\\Numpy_arc\\Numpy_arc.gdb\\sample_1000'
#arr = arcpy.da.TableToNumPyArray(in_tbl, '*"'
#uni = [i[0] for i in np.unique(arr)]
#uni = ['A', 'B', 'C', 'D']
#
#np.array(uni)  # array(['A', 'B', 'C', 'D'],  dtype='<U1')
#
##uni_u = [a.encode('utf8') for a in uni]
#
##np.array(uni_u) #  array([b'A', b'B', b'C', b'D'],  dtype='|S1')
#
## ---- compressed file demo ----
#in_npy = r'C:\GIS\A_Tools_scripts\Numpy_arc\Data\sample_1000b.npy'
#arr = np.load(in_npy)
#
#out_npy = r'C:\GIS\A_Tools_scripts\Numpy_arc\Data\sample_1000b_compressed'
#np.savez_compressed(out_npy, arr)
#
#arr_z = np.load(out_npy + r'.npz')  # ---- note added the *.npz extension
#arr_z.keys()  # => ['arr_0']
#arr_2 = arr_z['arr_0']
#np.all(arr == arr_2)  # ---- both arrays equal? ==> True
#
## another
#a = np.random.randint(0, 100, size=(2000, 1500)) * 1.0
#
#a.shape  # => (2000, 1500) a.dtype => dtype('float64')
#out_npy = r'C:\GIS\A_Tools_scripts\Numpy_arc\Data\float_2000x1500_comp.npz'
#
#np.savez_compressed(out_npy, a)
#r = arcpy.NumPyArrayToRaster(a)
#
#r.save(r'C:\GIS\A_Tools_scripts\Numpy_arc\Data\float_1500x2000.tif')
