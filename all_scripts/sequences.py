# -*- coding: UTF-8 -*-
"""
sequences
================

Script:   sequential_funcs.py
Author:   Dan.Patterson@carleton.ca
Modified: 2018-06-02
Purpose :
    Calculating sequential patterns for fields in geodatabase tables
Useage :

References:
-----------
  http://pro.arcgis.com/en/pro-app/arcpy/functions/
       numpyarraytoraster-function.htm
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
import numpy as np
import arcpy
from arcpytools import fc_info, tweet, frmt_rec, _col_format

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


def sequences(data, stepsize=0):
    """Return a list of arrays of sequences values denoted by stepsize

    data :
        List/array of values in 1D
    stepsize :
        Separation between the values.  If stepsize=0, sequences of equal
        values will be searched.  If stepsize is 1, then sequences incrementing
        by 1... etcetera.  Stepsize can be both positive or negative

    >>> # check for incrementing sequence by 1's
    d = [1, 2, 3, 4, 4, 5]
    s, o = sequences(d, 1, True)
    # s = [array([1, 2, 3, 4]), array([4, 5])]
    # o = array([[1, 4, 4],
    #            [4, 2, 6]])

    Notes:
    ------
    For strings, use

    >>> partitions = np.where(a[1:] != a[:-1])[0] + 1

    Variants:
    ---------
    Change `N` in the expression to find other splits in the data

    >>> np.split(data, np.where(np.abs(np.diff(data)) >= N)[0]+1)

    References:
    -----------

    `<https://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-
    sequences-elements-from-an-array-in-numpy>`_.

    `<https://stackoverflow.com/questions/50551776/python-chunk-array-on-
    condition#50551924>`_.
    """
    #
    a = np.array(data)
    a_dt = a.dtype.kind
    if a_dt in ('U', 'S'):
        seqs = np.split(a, np.where(a[1:] != a[:-1])[0] + 1)
    elif a_dt in ('i', 'f'):
        seqs = np.split(a, np.where(np.diff(a) != stepsize)[0] + 1)
    vals = [i[0] for i in seqs]
    cnts = [len(i) for i in seqs]
    seq_num = np.arange(len(cnts))
    too = np.cumsum(cnts)
    frum = np.zeros_like(too)
    frum[1:] = too[:-1]
    dt = [('ID', '<i4'), ('Value', a.dtype.str), ('Count', '<i4'),
          ('From_', '<i4'), ('To_', '<i4')]
    out = np.array(list(zip(seq_num, vals, cnts, frum, too)), dtype=dt)
    return out


# ---- Run options: _demo or from _tool
#
def _demo():
    """Code to run if in demo mode
    Requires:
        arcpytools fc_info, tweet
    """
    tbl = "Table_tools.gdb/pnts_2k_normal"
    in_tbl = "/".join(script.split("/")[:-2] + [tbl])
    #
    _, oid_fld, _, _ = fc_info(in_tbl, prn=False)  # run fc_info
    #
    in_fld = 'SequenceTxt'  #'Sequences2'  # 'SequenceTxt'
    stepsize = 0
    in_flds = [oid_fld, in_fld]   # OBJECTID, plus another field
    a = arcpy.da.TableToNumPyArray(in_tbl, in_flds, skip_nulls=False,
                                   null_value=-1)
#    a = [1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 5, 4, 4, 3, 3, 3, 2, 1]
    a = a[in_fld]
    out_tbl = None
    return in_tbl, a, in_fld, stepsize, out_tbl


def _tool():
    """run when script is from a tool
    """
    in_tbl = sys.argv[1]
    in_fld = sys.argv[2]
    stepsize = int(sys.argv[3])
    out_tbl = sys.argv[4]  # output field name
    #
    # ---- main tool section
    _, oid_fld, _, _ = fc_info(in_tbl, prn=False)  # run fc_info
    #
    flds = [oid_fld, in_fld]
    in_arr = arcpy.da.TableToNumPyArray(in_tbl, flds, skip_nulls=False,
                                   null_value=-1)
    a = in_arr[in_fld]  # do stuff with array
    tweet("{!r:}".format(a))
    return in_tbl, a, in_fld, stepsize, out_tbl


# ----------------------------------------------------------------------
# .... final code section producing the featureclass and extendtable
#
if len(sys.argv) == 1:
    testing = True
    in_tbl, a, in_fld, stepsize, out_tbl = _demo()
else:
    testing = False
    in_tbl, a, in_fld, stepsize, out_tbl = _tool()


msg = """
---- sequences ------------------------------------------------------
Processing ... {}
input field .. {}
step size  ... {} (difference between adjacent values)
output table . {}

----
Value : value in the field
Count : number of observations in that sequence
From_ : start location of the sequence (includes this index)
To_   : end location of the sequence (up to but not including)
NoData: -1
"""

tweet(msg.format(in_tbl, in_fld, stepsize, out_tbl))

out = sequences(a, stepsize=0)
if out_tbl not in ("#", "", " ", None, 'None'):
    arcpy.da.NumPyArrayToTable(out, out_tbl)
prn = frmt_rec(out[:50], 0, True, False)
tweet(prn)
#
## ---- reassemble the table for extending ----


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
