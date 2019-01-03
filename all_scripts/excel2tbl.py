# -*- coding: utf-8 -*-
"""
excel2tbl
===========

Script :   excel2tbl.py

Author :   Dan_Patterson@carleton.ca

Modified : 2018-11-23

Purpose:  tools for working with numpy arrays

Useage :

References
----------
`<http://pro.arcgis.com/en/pro-app/arcpy/data-access/numpyarraytotable.htm>`_.
`<http://pro.arcgis.com/en/pro-app/arcpy/data-access/tabletonumpyarray.htm>`_.
---------------------------------------------------------------------
"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

import sys
import numpy as np
import xlrd
import arcpy.da
from arcpy import env

env.overwriteOutput = True
#from arcpytools import fc_info, tweet  #, frmt_rec, _col_format
#import arcpy

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

# ----------------------------------------------------------------------
# ---- excel_np
def excel_np(path, sheet_num=0, int_null=-999):
    """Read excel files to numpy structured/record arrays.  Your spreadsheet
    must adhere to simple rules::
      - first row must contain the field names for the output array
      - no blank rows or columns, basically, no fluff or formatting
      - if you have nodata values, put them in, since blank cells will be
        'corrected' as best as possible.
      - text and numbers in a column, results in a text column

    See arraytools.a_io for excel_np for complete description
    """
    def isfloat(a):
        """float check"""
        try:
            i = float(a)
            return i
        except ValueError:
            return np.nan

    def punc_space(name):
        """delete punctuation and spaces and replace with '_'"""
        punc = list('!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~ ')
        return "".join([[i, '_'][i in punc] for i in name])

    # import xlrd
    w = xlrd.open_workbook(path)        # xlrd.book.Book class
    sheets = len(w.sheets())
    if sheet_num > sheets:
        return None
    sheet = w.sheet_by_index(sheet_num) # sheet by number
    # sheet = w.sheet_by_name('test')   # case sensitive, not implemented
    names = sheet.row_values(0)         # clean these up later
    cols = sheet.ncols
    rows = sheet.nrows
    col_data = [sheet.col_values(i, 1, rows) for i in range(cols)]
    row_guess = sheet.row_values(1)
    row_dts = [np.asarray(i).dtype.kind for i in row_guess]
    col_dts = [np.asarray(col_data[i]).dtype.kind
               for i in range(cols)]
    clean = []
    for i in range(len(row_dts)):
        c = col_data[i]
        if row_dts[i] == col_dts[i]:    # same dtype... send to array
            ar = np.asarray(c)
        if row_dts[i] == 'f':           # float? if so, substitute np.nan
            ar = np.array([isfloat(i) for i in c])
            is_nan = np.isnan(ar)       # find the nan values, then check
            not_nan = ar[~is_nan]       # are the floats == ints?
            if np.all(np.equal(not_nan, not_nan.astype('int'))):  # integer?
                ar[is_nan] = int_null   # assign the integer null
                ar = ar.astype('int')
        elif row_dts[i] in ('U', 'S'):  # unicode/string... send to array
            ar = np.char.strip(ar)
            ar = np.where(np.char.str_len(ar) == 0, 'None', ar)
        else:
            ar = np.asarray(c)
        clean.append(ar)
    # ---- assemble the columns for the array ----
    dt_str = [i.dtype.str for i in clean]
    names = [i.strip() for i in names]      # clean up leading/trailing spaces
    names = [punc_space(i) for i in names]  # replace punctuation and spaces
    dts_name = list(zip(names, dt_str))
    arr = np.empty((rows-1,), dtype=dts_name)
    cnt = 0
    for i in names:
        arr[i] = clean[cnt]
        cnt += 1
    return arr


# ----------------------------------------------------------------------
# .... final code section producing the featureclass and extendtable
if len(sys.argv) == 1:
    testing = True
    in_excel = script.rpartition("/")[0] + "/Data/test.xlsx"
    sheet_num = 0
    int_null = -999
    arr = excel_np(in_excel, sheet_num=sheet_num, int_null=int_null)
    print("Array returned...\n{}".format(arr))
    # parameters here
else:
    testing = False
    in_excel = sys.argv[1]
    sheet_num = int(sys.argv[2])
    int_null = sys.argv[3]
    if int_null in ('-2147483648', '-32768', '-128', '-9', '-1'):
        int_null == int(int_null)
    else:
        int_null = '-2147483648'
    out_tbl = sys.argv[4]
    arr = excel_np(in_excel, sheet_num=sheet_num, int_null=int_null)
    if arr is None:
        print("not a sheet number")
    else:
        arcpy.da.NumPyArrayToTable(arr, out_tbl)

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
