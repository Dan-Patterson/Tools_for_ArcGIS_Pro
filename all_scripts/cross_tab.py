 # -*- coding: UTF-8 -*-
"""
:Script:   cross_tab.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-09-23
:Purpose:  Crosstabulate data
:Notes:
:
:References:
: https://stackoverflow.com/questions/12983067/how-to-find-unique-vectors-of
:         -a-2d-array-over-a-particular-axis-in-a-vectorized
: https://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
: http://stackoverflow.com/questions/38030054/
:      create-adjacency-matrix-in-python-for-large-dataset
: np.unique
: in the newer version, they use flags to get the sums
:
"""
import sys
import numpy as np
import arcpy
from textwrap import indent

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=50, linewidth=80, precision=2,
                    suppress=True, threshold=100,
                    formatter=ft)
np.ma.masked_print_option.set_display('-')

script = sys.argv[0]


def tweet(msg):
    """Print a message for both arcpy and python.
    : msg - a text message
    """
    m = "\n{}\n".format(msg)
    arcpy.AddMessage(m)
    print(m)
    print(arcpy.GetMessages())


def _prn(r, c, a):
    """fancy print formatting.
    """
    r_sze = max(max([len(i) for i in r]), 8)
    c_sze = [max(len(str(i)), 5) for i in c]
    f_0 = '{{!s:<{}}} '.format(r_sze)
    f_1 = ('{{!s:>{}}} '*len(c)).format(*c_sze)
    frmt = f_0 + f_1
    hdr = 'Result' + '_' * (r_sze-7)
    txt = [frmt.format(hdr, *c)]
    txt2 = txt + [frmt.format(r[i], *a[i]) for i in range(len(r))]
    result = "\n".join(txt2)
    return result


def crosstab(row, col, verbose=False):
    """Crosstabulate 2 data arrays, shape (N,), using np.unique.
    :  scipy.sparse has similar functionality and is faster for large arrays.
    :
    :Requires:  A 2D array of data with shape(N,) representing two variables
    :--------
    : row - row variable
    : col - column variable
    :
    :Returns:
    : ctab - the crosstabulation result as row, col, count array
    : a - the crosstabulation in a row, col, count, but filled out whether a
    :     particular combination exists or not.
    : r, c - unique values/names for the row and column variables
    :
    """
    dt = np.dtype([('row', row.dtype), ('col', col.dtype)])
    rc = np.asarray(list(zip(row, col)), dtype=dt)
    r = np.unique(row)
    c = np.unique(col)
    u, idx, inv, cnt = np.unique(rc, return_index=True, return_inverse=True,
                                 return_counts=True)
    rcc_dt = u.dtype.descr
    rcc_dt.append(('Count', '<i4'))
    ctab = np.asarray(list(zip(u['row'], u['col'], cnt)), dtype=rcc_dt)
    a = np.zeros((len(r), len(c)), dtype=np.int_)
    rc = [[(np.where(r == i[0])[0]).item(),
           (np.where(c == i[1])[0]).item()] for i in ctab]
    for i in range(len(ctab)):
        rr, cc = rc[i]
        a[rr, cc] = ctab[i][2]
    result = _prn(r, c, a)
    if verbose:
        tweet(result)
    return ctab, a, result, r, c


def _demo():
    """run a test using a file
    : TableToNumPyArray(in_table, field_names, {where_clause},
    :                   {skip_nulls}, {null_value})
    """
    # using the UG1 and Employer_group that is not null in both cases
    # 10,734 records
    # f = r'C:\FPA_2\D_gdb_files\FPA_final_Sept_13.gdb\UG1_EmpGroup_notnull'
    #f = r'C:\FPA_2\D_gdb_files\FPA_final_Sept_13.gdb\UG1_Title_notnull'
    f = r'C:\FPA_2\D_gdb_files\FPA_final_Sept_13.gdb\grad_only_employer_group'
    flds = [i.name for i in arcpy.ListFields(f)]
    null_dict = {'OBJECTID': -9, 'Student_num': -9, 'Code': -9, 'Year_': -9}
    t = arcpy.da.TableToNumPyArray(in_table=f, field_names=flds,
                                   skip_nulls=False, null_value=null_dict)
#    rows = t['UG1']  # for undergrad
#    rows = t['Grad1']  # for grad
#    cols = t['Sector']
    rows =  t['Employer_Group']  # t['Title_Position']
    cols = t['Grad1'] # t['UG1']
#    f = r'C:\GIS\Tools_scripts\Statistics\Stats_demo_01.gdb\pnts_2K_normal'
#    flds = [i.name for i in arcpy.ListFields(f)]
#    t = arcpy.da.TableToNumPyArray(in_table=f, field_names=flds,
#                                   skip_nulls=False)  # , null_value=null_dict)
#    rows = t['Text01']
#    cols = t['Text02']
    ctab, a, result, r, c = crosstab(rows, cols, verbose=True)
    return ctab, a, result, r, c


def _demo2():
    """run a test using 2K of normally distributed points
    : Files

    """
    f = r'C:\GIS\Tools_scripts\Statistics\Stats_demo_01.gdb\pnts_2K_normal'
    flds = [i.name for i in arcpy.ListFields(f)]
    t = arcpy.da.TableToNumPyArray(in_table=f, field_names=flds,
                                   skip_nulls=False)  # , null_value=null_dict)
    rows = t['Text01']
    cols = t['Text02']
    ctab, a, result, r, c = crosstab(rows, cols, verbose=False)
    return ctab, a, result, r, c


def _demo3():
    """load a npy file
    :  C:\GIS\Tools_scripts\Data\sample_20.npy
    :  C:\GIS\Tools_scripts\Data\sample_1000.npy
    :  C:\GIS\Tools_scripts\Data\sample_10K.npy
    :  C:\GIS\Tools_scripts\Data\sample_100K.npy
    :
    :    dtype=[('OBJECTID', '<i4'), ('f0', '<i4'), ('County', '<U2'),
    :           ('Town', '<U6'), ('Facility', '<U8'), ('Time', '<i4')])
    """
    f = r'C:\GIS\Tools_scripts\Data\sample_100K.npy'
    t = np.load(f)
    rows = t['County']  #t['Text01']
    cols = t['Town']
    ctab, a, result, r, c = crosstab(rows, cols, verbose=False)
    return ctab, a, result, r, c


frmt = """
Crosstab results ....
{}\n
The array of counts/frequencies....
{}\n
Row and column headers...
{}
{}\n
And as a fancy output which can be saved to a csv file using
....np.savetxt('c:/path/file_name.csv', array_name, fmt= '%s', delimiter=', ')
{}
"""


if len(sys.argv) == 1:
    ctab, a, result, r, c = _demo()
    pre = '    '
    msg = frmt.format(indent(str(ctab), pre), indent(str(a), pre),
                      indent(str(r), pre), indent(str(c), pre),
                      indent(result, pre))
    print("\n{}{}".format('---- cross_tab.py ', "-"*60))
    print(msg)
else:
    in_tbl = sys.argv[1]
    row_fld = sys.argv[2]
    col_fld = sys.argv[3]
    flds = [row_fld, col_fld]
    t = arcpy.da.TableToNumPyArray(in_table=in_tbl, field_names=flds,
                                   skip_nulls=False)  # , null_value=null_dict)
    rows = t[row_fld]
    cols = t[col_fld]
    ctab, a, result, r, c = crosstab(rows, cols, verbose=True)
    args = [in_tbl, row_fld, col_fld]
    msg = "\nTable {}\nrow field {}\ncol field {}".format(*args)
    tweet(msg)

if __name__ == "__main__":
    """run crosstabulation with data"""
#    ctab, a, result, r, c = _demo()
