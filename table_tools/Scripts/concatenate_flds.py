# -*- coding: UTF-8 -*-
"""
:Script:   concatenate_flds.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2018-01-30
:Purpose:  tools for working with numpy arrays
:  Concatenate fields from fields in a geodatabase table.
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
np.set_printoptions(edgeitems=3, linewidth=80, precision=2, suppress=True,
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
    print(arcpy.GetMessages())


def _cleanup(arrs, strip_list):
    """clean the arrays if needed"""
    cleaned = []
    for ar in arrs:
        if ar.dtype.kind in ('f', 'i'):
            tmp = ar.astype(np.unicode_)
        else:
            tmp = ar
        for i in strip_list:
            tmp = np.char.strip(tmp)
            tmp = np.char.replace(tmp, str(i), "")
        cleaned.append(tmp)
    return cleaned


def concat_flds(arrs, sep='space', name=None, strip_list=None, with_ids=True):
    """Concatenate a sequence of arrays to string format and return a
    :  structured array or ndarray
    :  arrs - a list of single arrays of the same length
    :  sep - the separator to separate the arrays
    :  name - used for structured array
    """
    def cleanup(arrs, strip_list):
        """clean the arrays if needed"""
        cleaned = []
        for ar in arrs:
            if ar.dtype.kind in ('f', 'i'):
                tmp = ar.astype(np.unicode_)
            else:
                tmp = ar
            for i in strip_list:
                tmp = np.char.replace(tmp, str(i), "")
                tmp = np.char.strip(tmp)
            cleaned.append(tmp)
        return cleaned
    # ---- Main section
    N = len(arrs)
    if sep == 'space':
        sep = ' '
    elif sep == 'comma':
        sep = ', '
    elif sep == 'none':
        sep = ''
    if N < 2:
        return arrs
    if strip_list is None:
        cleaned = arrs
    else:
        cleaned = cleanup(arrs, strip_list)
    a, b = cleaned[0], cleaned[1]
    c = ["{}{}{}".format(i, sep, j) for i, j in list(zip(a, b))]
    if N > 2:
        for i in range(2, len(cleaned)):
            c = ["{}{}{}".format(i, sep, j)
                 for i, j in list(zip(c, cleaned[i]))]
    c = np.asarray(c)
    sze = c.dtype.str
    if name is not None:
        c.dtype = [(name, sze)]
    else:
        name = 'concat'
        c.dtype = [(name, sze)]
    if with_ids:
        tmp = np.copy(c)
        dt = [('IDs', '<i8'), (name, sze)]
        c = np.empty((tmp.shape[0], ), dtype=dt)
        c['IDs'] = np.arange(1, tmp.shape[0] + 1)
        c[name] = tmp
    return c


# ---- Run options: _demo or from _tool
#
def _demo():
    """Code to run if in demo mode
    """
    # in_tbl = r"C:\Git_Dan\arraytools\Data\numpy_demos.gdb\sample_10k"
    in_tbl = r"C:\GIS\Joe_address\Joe_address\Joe_address.gdb\Addr_summary"
    in_flds = ['Street', 'Len_range', 'Test_txt', 'Len_range2']
    nv = np.iinfo(np.int32).min  # use smallest int...it gets cast as needed
    a = arcpy.da.TableToNumPyArray(in_tbl, in_flds,
                                   skip_nulls=False,
                                   null_value=nv)
    a0 = a['Street']
    a1 = a['Len_range']
    a2 = a['Test_txt']
    a3 = a['Len_range2']
    arrs = [a0, a1, a2, a3]
#    a = [np.arange(5, dtype='int'),
#            np.arange(10, 5, -1, dtype='float'),
#            np.array(['a', 'b', 'c', 'd', 'e'])]
    strip_list = [nv, 'None', None, "", ","]
    c = concat_flds(arrs, sep=" ", name="Test",
                    strip_list=strip_list, with_ids=True)
#    arcpy.da.ExtendTable(in_tbl, 'OBJECTID', out_array, 'IDs')
    return arrs, c


def _tool():
    """run when script is from a tool
    """
    in_tbl = sys.argv[1]
    in_flds = sys.argv[2]
    fld_sep = "{}".format(sys.argv[3])
    strip_list = sys.argv[4]
    out_fld = sys.argv[5]

    if ';' in in_flds:
        in_flds = in_flds.split(';')
    else:
        in_flds = [in_flds]

    desc = arcpy.da.Describe(in_tbl)
    tbl_path = desc['path']
    fnames = [i.name for i in arcpy.ListFields(in_tbl)]
    if out_fld in fnames:
        out_fld += 'dup'
    out_fld = arcpy.ValidateFieldName(out_fld, tbl_path)
    args = [in_tbl, in_flds, out_fld, tbl_path]
    msg = "in_tbl {}\nin_fld {}\nout_fld  {}\ntbl_path  {}".format(*args)
    tweet(msg)
    oid = 'OBJECTID'
    vals = [oid] + in_flds
    nv = np.iinfo(np.int32).min  # use smallest int...it gets cast as needed
    arr = arcpy.da.TableToNumPyArray(in_tbl, vals,
                                     skip_nulls=False,
                                     null_value=nv)
    tweet("{!r:}".format(arr))
    #
    # ---- process arrays from the fields, concatenate, and ExtendTable ----
    arrs = [arr[i] for i in in_flds]
    out_array = concat_flds(arrs, sep=fld_sep, name=out_fld,
                            strip_list=strip_list, with_ids=True)
    arcpy.da.ExtendTable(in_tbl, 'OBJECTID', out_array, 'IDs')
    del in_tbl, arr, out_array


# ----------------------------------------------------------------------
# .... final code section producing the featureclass and extendtable
if len(sys.argv) == 1:
    testing = True
    arrs, c = _demo()
    frmt = "Result...\n{}"
    print(frmt.format(c))
else:
    testing = False
    _tool()
#
if not testing:
    print('Concatenation done...')

# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
