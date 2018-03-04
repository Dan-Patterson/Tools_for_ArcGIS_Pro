# -*- coding: UTF-8 -*-
"""
:Script:   strip_stuff.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2018-02-10
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


punc = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.',
        '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`',
        '{', '|', '}', '~']
whitesp = [' ', '\t', '\n', '\r', '\x0b', '\x0c']


def tweet(msg):
    """Print a message for both arcpy and python.
    : msg - a text message
    """
    m = "\n{}\n".format(msg)
    arcpy.AddMessage(m)
    print(m)


def clean_fld(a, strip_list, new_value=""):
    """clean the arrays if needed"""
    tmp = np.copy(a)
    for i in strip_list:
        tmp = np.char.replace(tmp, str(i), new_value)
    cleaned = tmp
    return cleaned


# ---- Run options: _demo or from _tool
#
def _demo():
    """Code to run if in demo mode
    """
    a = np. array(['1, 2, 3, 4, 5', 'a, b, c', '6, 7, 8, 9',
                   'd, e, f, g, h', '10, 11, 12, 13'])
    cleaned = clean_fld(a, punc)
    return a, cleaned


def _tool():
    """run when script is from a tool
    """
    in_tbl = sys.argv[1]
    in_fld = sys.argv[2]
    out_fld = sys.argv[3]
    all_punc = sys.argv[4]
    all_white = sys.argv[5]
    all_extra = sys.argv[6]
    all_others = sys.argv[7]

    a0 = [[], punc][all_punc in (True, 'True', 'true')]
    a1 = [[], whitesp][all_white in (True, 'True', 'true')]
    if len(all_others) == 1:
        a2 = list(all_others)
    elif len(all_others) > 1:
        if ";" in all_others:
            a2 = all_others.replace(";", "xx")
            a2 = a2.split('xx')[:-1]
    else:
        a2 = []
    #
    strip_list = a0 + a1 + a2
    desc = arcpy.da.Describe(in_tbl)
    tbl_path = desc['path']
    is_gdb_tbl = tbl_path[-4:] == '.gdb'
    fnames = [i.name for i in arcpy.ListFields(in_tbl)]
    if out_fld in fnames:
        out_fld += 'dup'
    out_fld = arcpy.ValidateFieldName(out_fld, tbl_path)
    args = [in_tbl, in_fld, out_fld, tbl_path]
    msg = "in_tbl {}\nin_fld {}\nout_fld  {}\ntbl_path  {}".format(*args)
    tweet(msg)
    tweet("Removing .... {}".format(strip_list))
    oid = 'OBJECTID'
    vals = [oid, in_fld]
    #
    # ---- do the work
    #
    arr = arcpy.da.TableToNumPyArray(in_tbl, vals)
    tweet("{!r:}".format(arr))
    a0 = arr[in_fld]
    #
    cleaned = clean_fld(a0, strip_list)  # punc
    #
    if all_extra in (True, 'True', 'true'):
        sps = ['    ',  '   ', '  ']
        for i in sps:
            cleaned = np.char.replace(cleaned, i, " ")
    sze = cleaned.dtype.str
    dt = [('IDs', '<i8'), (out_fld, sze)]
    out_array = np.empty((arr.shape[0], ), dtype=dt)
    out_array['IDs'] = np.arange(1, arr.shape[0] + 1)
    out_array[out_fld] = cleaned
    #
    #
    arcpy.da.ExtendTable(in_tbl, 'OBJECTID', out_array, 'IDs')


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
    a, c = _demo()
