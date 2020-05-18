 # -*- coding: UTF-8 -*-
"""
=========
cross_tab
=========

Script : cross_tab.py

Author: Dan_Patterson@carleton.ca

Modified : 2019-02-23

Purpose : Crosstabulate data

References
----------

`<https://stackoverflow.com/questions/12983067/how-to-find-unique-vectors-of
-a-2d-array-over-a-particular-axis-in-a-vectorized>`_.

`<https://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy
-array>`_.

`<http://stackoverflow.com/questions/38030054/create-adjacency-matrix-in-
python-for-large-dataset>`_.
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
    r = r.tolist()
    r.append('Total')
    c = c.tolist()
    c.append('Total')
    r_sze = max(max([len(str(i)) for i in r]), 8)
    c_sze = [max(len(str(i)), 5) for i in c]
    f_0 = '{{!s:<{}}} '.format(r_sze)
    f_1 = ('{{!s:>{}}} '*len(c)).format(*c_sze)
    frmt = f_0 + f_1
    hdr = 'Result' + '_'*(r_sze-7)
    txt = [frmt.format(hdr, *c)]
    txt2 = txt + [frmt.format(r[i], *a[i]) for i in range(len(r))]
    result = "\n".join(txt2)
    return result


def crosstab(row, col, verbose=False):
    """Crosstabulate 2 data arrays, shape (N,), using np.unique.
    scipy.sparse has similar functionality and is faster for large arrays.

    Parameters
    ----------
    a : array
        A 2D array of data with shape(N,) representing two variables
    row : text
        row variable
    col : text
        column variable

    Returns
    -------
    ctab : the crosstabulation result as row, col, count array
    a : the crosstabulation in a row, col, count, but filled out whether a
        particular combination exists or not.
    r, c : unique values/names for the row and column variables
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
    a = np.zeros((len(r)+1, len(c)+1), dtype=np.int_)
    rc = [[(np.where(r == i[0])[0]).item(),
           (np.where(c == i[1])[0]).item()] for i in ctab]
    for i in range(len(ctab)):
        rr, cc = rc[i]
        a[rr, cc] = ctab[i][2]
    a[-1,:] = np.sum(a, axis=0)
    a[:, -1] = np.sum(a, axis=1)
    result = _prn(r, c, a)
    if verbose:
        tweet(result)
    return ctab, a, result, r, c


def tbl_2_np_array(in_tbl, flds, skip_nulls=False, null_value=None):
    """Form the TableToNumPyArray to account for nulls for various dtypes

    """
    int_min = np.iinfo(np.int32).min
    float_min = np.finfo(np.float64).min
    str_val = "None"
    nulls = {'Double':float_min, 'Integer':int_min, 'String':str_val}
    #
    fld_dict = {i.name: i.type for i in arcpy.ListFields(in_tbl)}
    null_dict = {f:nulls[fld_dict[f]] for f in flds}
    t = arcpy.da.TableToNumPyArray(in_table=in_tbl, field_names=flds,
                                   skip_nulls=False,
                                   null_value=null_dict)
    return t


def _demo():
    """run a test using 2K of normally distributed points
    : TableToNumPyArray(in_table, field_names, {where_clause},
    :                   {skip_nulls}, {null_value})
    """
    # Load the table, with 2 fields, produce the table and the crosstabulation
    f = "/".join(script.split("/")[:-2]) + '/Table_tools.gdb/pnts_2K_normal'
    flds = ['Text01', 'Sequences']
    t = tbl_2_np_array(in_tbl=f, flds=flds)
    #
    row = t[flds[0]]
    col = t[flds[1]]
    ctab, a, result, r, c = crosstab(row, col, verbose=True)
    return ctab, a, result, r, c


def _demo2():
    """load a npy file

    - /Data/sample_20.npy
    - /Datasample_1000.npy
    - /Data/sample_10K.npy
    - /Data/sample_100K.npy

    dtype=[('OBJECTID', '<i4'), ('f0', '<i4'), ('County', '<U2'),
           ('Town', '<U6'), ('Facility', '<U8'), ('Time', '<i4')])
    """
    f = "/".join(script.split("/")[:-3]) + "/Data/sample_100K.npy"
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
    out_tbl = sys.argv[4]
    txt_file = sys.argv[5]
    flds = [row_fld, col_fld]
    #
    t = tbl_2_np_array(in_tbl=in_tbl, flds=flds)
    #
    rows = t[row_fld]
    cols = t[col_fld]
    ctab, a, result, r, c = crosstab(rows, cols, verbose=True)
    args = [in_tbl, row_fld, col_fld]
    pre = '    '
    msg = frmt.format(indent(str(ctab), pre), indent(str(a), pre),
                      indent(str(r), pre), indent(str(c), pre),
                      indent(result, pre))
    tweet("{}{}".format('---- cross_tab.py ', "-"*60))
    tweet(msg)
    if not (out_tbl in ['#', '', None]):
        arcpy.da.NumPyArrayToTable(ctab, out_tbl)
    if not (txt_file in ['#', '', None]):
        with open(txt_file, 'w') as f:
            f.write("Crosstab for ... {}\n".format(in_tbl))
            f.write(result)

if __name__ == "__main__":
    """run crosstabulation with data"""
#    ctab, a, result, r, c = _demo()
