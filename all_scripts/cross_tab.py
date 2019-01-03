# -*- coding: UTF-8 -*-
"""
cross_tab
=========

Script :   cross_tab.py

Author :   Dan.Patterson@carleton.ca

Modified : 2018-10-22

Purpose :  Crosstabulate data

Notes:

References:
-----------

`<https://stackoverflow.com/questions/12983067/how-to-find-unique-vectors-of
-a-2d-array-over-a-particular-axis-in-a-vectorized>`_.

`<https://stackoverflow.com/questions/16970982/find-unique-rows-
in-numpy-array>`_.

`<http://stackoverflow.com/questions/38030054/create-adjacency-matrix-in-
python-for-large-dataset>`_.

np.unique - in the newer version, they use flags to get the sums
:
"""
import sys
import numpy as np
from arraytools.fc_tools._common import * #(tweet, tbl_arr)
# ---- others from above , de_punc, _describe, fc_info, fld_info, null_dict,
#from arcpy import AddMessage, ListFields
#from textwrap import indent

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2,
                    suppress=True, threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')

script = sys.argv[0]


def crosstab(row, col, r_name=None, c_name=None, verbose=False):
    """Crosstabulate 2 data arrays, shape (N,), using np.unique.
    scipy.sparse has similar functionality and is faster for large arrays.

    Requires:
    --------
    A 2D array of data with shape(N,) representing two variables

    row : field/column
        row variable
    col : field/column
        column variable

    Useage:
    ------
    >>> float_min = np.finfo(np.float).min
    >>> float_max = np.finfo(np.float).max
    >>> int_min = np.iinfo(np.int_).min
    >>> int_max = np.iinfo(np.int_).max
    >>> f = r'C:\some\path\your.gdb\your_featureclass'
    >>> null_dict = {'Int_fld': int_min, 'Float_fld': float_min}  # None strings
    >>> flds = ['Int_field', 'Txt01', 'Txt02']  # 2 text fields
    >>> t = arcpy.da.TableToNumPyArray(in_table=f, field_names=flds,
                                      skip_nulls=False)
                                      # , null_value=null_dict) if needed
    >>> rows = t['Txt01']
    >>> cols = t['Txt02']
    >>> ctab, a, result, r, c = crosstab(rows, cols, verbose=False)

    Returns:
    --------
      ctab :
          the crosstabulation result as row, col, count array
      a :
          the crosstabulation in a row, col, count, but filled out whether a
          particular combination exists or not.
      r, c :
          unique values/names for the row and column variables
    """
    def _prn(r, c, r_name, c_name, a):
        """fancy print formatting.
        """
        r = r.tolist()
        r.append('Total')
        c = c.tolist()
        c.append('Total')
        r_sze = max([len(str(i)) for i in r]) + 2
        c_sze = [max(len(str(i)), 5) for i in c]
        f_0 = '{{!s:<{}}} '.format(r_sze)
        f_1 = ('{{!s:>{}}} '*len(c)).format(*c_sze)
        frmt = f_0 + f_1
        hdr = 'Row: {}\nCol: {}\n'.format(r_name, c_name) + '_' * (r_sze)
        txt = [frmt.format(hdr, *c)]
        txt2 = txt + [frmt.format(r[i], *a[i]) for i in range(len(r))]
        result = "\n".join(txt2)
        return result
    #
    r_name = [str(r_name), "Row"][r_name is None]
    c_name = [str(c_name), "Col"][c_name is None]
    dt = np.dtype([(r_name, row.dtype), (c_name, col.dtype)])
    rc = np.asarray(list(zip(row, col)), dtype=dt)
    r = np.unique(row)
    c = np.unique(col)
    u, idx, cnt = np.unique(rc, return_index=True, return_counts=True)
    rcc_dt = u.dtype.descr
    rcc_dt.append(('Count', '<i4'))
    ctab = np.asarray(list(zip(u[r_name], u[c_name], cnt)), dtype=rcc_dt)
    c0 = np.zeros((len(r), len(c)), dtype=np.int_)
    rc = [[(np.where(r == i[0])[0]).item(),
           (np.where(c == i[1])[0]).item()] for i in ctab]
    for i in range(len(ctab)):
        rr, cc = rc[i]
        c0[rr, cc] = ctab[i][2]
    tc = np.sum(c0, axis=0)
    c1 = np.vstack((c0, tc))
    tr = np.sum(c1, axis=1)
    counts = np.hstack((c1, tr.reshape(tr.shape[0], 1)))
    out_tbl = _prn(r, c, r_name, c_name, counts)
    if verbose:
        tweet(out_tbl)
    return ctab, counts, out_tbl

def tabular_sum(a, r_name, c_name, val_name):
    """array, row, col and value fields
    """
    rc = a[[r_name, c_name]]
    sum_name = val_name +'Sum'
    dt = rc.dtype.descr + [(sum_name, '<i4')]
    uniq = np.unique(rc)
    out_ = []
    for u in uniq:
        c0, c1 = u
        idx = np.logical_and(a[r_name]==c0, a[c_name]==c1)
        val = np.nansum(a[val_name][idx])
        out_.append([c0, c1, val])
    out_ = np.array(out_)
    z = np.empty((len(out_),), dtype=dt)
    z[r_name] = out_[:, 0]
    z[c_name] = out_[:, 1]
    z[sum_name] = out_[:, 2].astype('int32')
    return z
#        cond =(np.where((a[row]=='A') & (a['Town'] == 'A_'), a['Time'], 0))




frmt = """\
Crosstab results ....
{}\n
The array of counts/frequencies....
{}\n
Row field:  {}
Col field:  {}\n
Row and column headers...
{}
{}\n
And as a fancy output which can be saved to a csv file using
....np.savetxt('c:/path/name.csv', array, fmt= '%s', delimiter=', ')\n
{}
"""
# ---- crosstab from tool, uncomment for testing or tool use
#if len(sys.argv) == 1:
##    in_tbl = r"C:\Git_Dan\arraytools\array_tools_testing\array_tools.gdb\pnts_2000"
#    in_tbl = 'C:/Git_Dan/arraytools//Data/sample_10K.npy'
#    a = tbl_arr(in_tbl)  #arcpy.da.TableToNumPyArray(in_tbl, "*")
#    row_fld = 'County'
#    col_fld = 'Town'
#    rows = a[row_fld]
#    cols = a[col_fld]
##    ctab, counts, out_tbl = crosstab(rows, cols, r_name=row_fld, c_name=col_fld, verbose=False)
##    tweet(frmt.format(in_tbl, ctab, row_fld, col_fld, r, c, result))
#else:
#    in_tbl = sys.argv[1]
#    row_fld = sys.argv[2]
#    col_fld = sys.argv[3]
#    flds = [row_fld, col_fld]
#    t = tbl_arr(in_tbl)
#    # arcpy.da.TableToNumPyArray(in_table=in_tbl, field_names=flds,
#    #                            skip_nulls=False)  # , null_value=null_dict)
#    rows = t[row_fld]
#    cols = t[col_fld]
#    ctab, a, result, r, c = crosstab(rows, cols, verbose=True)
#    tweet(frmt.format(in_tbl, ctab, row_fld, col_fld, r, c, result))

if __name__ == "__main__":
    """run crosstabulation with data"""
#    ctab, counts, out_tbl = crosstab(a['County'], a['Town'], r_name='County', c_name='Town', verbose=False)
#    ctab, a, result, r, c = _demo()
