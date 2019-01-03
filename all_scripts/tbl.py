# -*- coding: UTF-8 -*-
"""
tbl
===

Script :   tbl.py   tools for working with text array in table form

Author :   Dan.Patterson@carleton.ca

Modified : 2018-11-12

Purpose :  Tabulate data

- Unique counts on 2 or more variables.
- Sums, mins, max etc on variable classes

Requires:
---------
`frmts.py` is required since it uses print functions from there

`prn` is used for fancy printing if it loads correctly

Useage:
-------
To convert esri geodatabase tables or shapefile tables to arrays, use the
following guidelines.

>>> float_min = np.finfo(np.float).min
>>> float_max = np.finfo(np.float).max
>>> int_min = np.iinfo(np.int_).min
>>> int_max = np.iinfo(np.int_).max
>>> f = r'C:\some\path\your.gdb\your_featureclass'
>>> null_dict = {'Int_fld': int_min, 'Float_fld': float_min}  # None strings
>>> flds = ['Int_field', 'Txt01', 'Txt02']  # 2 text fields
>>> a = arcpy.da.TableToNumPyArray(in_table=f, field_names=flds,
                                  skip_nulls=False,
                                  null_value=null_dict)  # if needed
>>> row = 'Txt01'
>>> col = 'Txt02'
>>> ctab = crosstab(a, row, col, verbose=False)

Notes:
------
Useful tip:

`...install folder.../Lib/site-packages/numpy/core/numerictypes.py`

>>> # "import string" is costly to import!
>>> # Construct the translation tables directly
>>> #   "A" = chr(65), "a" = chr(97)
>>> _all_chars = [chr(_m) for _m in range(256)]
>>> _ascii_upper = _all_chars[65:65+26]
>>> _ascii_lower = _all_chars[97:97+26]
>>> _just_numbers = _all_chars[48:58]
>>> LOWER_TABLE = "".join(_all_chars[:65] + _ascii_lower + _all_chars[65+26:])
>>> UPPER_TABLE = "".join(_all_chars[:97] + _ascii_upper + _all_chars[97+26:])

- np.char.split(s, ' ')
- np.char.startswith(s, 'S')
- np.char.strip()
- np.char.str_len(s)

- np.sum(np.char.startswith(s, ' '))  # check for leading spaces
- np.sum(np.char.endswith(s0, ' '))   # check for trailing spaces
- s0 = np.char.rstrip(s0)

**Partitioning**:
::
    lp = np.char.partition(s0, ' ')[:, 0]   # get the left-most partition
    rp = np.char.rpartition(s0, ' ')[:, -1] # get the right-most partition
    lpu, lpcnts= np.unique(lp, return_counts=True)
    rpu, rpcnts= np.unique(rp, return_counts=True)

Queries: d
    np.char.find(c, query) >= 0


References:
-----------

`<https://stackoverflow.com/questions/12983067/how-to-find-unique-vectors-of
-a-2d-array-over-a-particular-axis-in-a-vectorized>`_.

`<https://stackoverflow.com/questions/16970982/find-unique-rows-
in-numpy-array>`_.

`<http://stackoverflow.com/questions/38030054/create-adjacency-matrix-in-
python-for-large-dataset>`_.

np.unique - in the newer version, they use flags to get the sums

"""
# pylint: disable=C0103
# pylint: disable=R1710
# pylint: disable=R0914

import sys
from textwrap import dedent
import numpy as np

# ---- others from above , de_punc, _describe, fc_info, fld_info, null_dict,

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=5, linewidth=80, precision=2,
                    suppress=True, threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')

script = sys.argv[0]

if 'prn' not in locals().keys():
    try:
        from arraytools.frmts import prn
    except:
        prn = print

__all__ = ['find_in',
           'tbl_count',
           'tbl_sum']

# ---- text columns... via char arrays
#
#def tbl_replace(a, col=None, from_=None, to_=None):
#    """table replace
#    """
#    #np.char.replace(
#    pass
#    return None

def find_in(a, col=None, what=None, where='in', any_case=True, pull='all'):
    """Query a recarray/structured array for values

    a : recarray/structured array
        Only text columns can be queried
    col : column/field to query
        Only 1 field can be queried at a time for the condition.
    what : string or number
        The query.  If a number, the field is temporarily converted to a
        text representation for the query.
    where : string
        s, i, eq, en .... `st`(arts with), `in`, `eq`(ual), `en`(ds with)
    any_case : boolean
        True, will find records regardless of `case`, applies to text fields
    extract: text or list
        `all`:  extracts all records where the column case is found
        `list`: extracts the records for only those fields in the list

    >>> find_text(a, col='FULLNAME', what='ABBEY', pull=a.dtype.names[:2])
    """
    # ---- error checking section ----
    e0 = """
    :Query error: You provided...
    :  dtype: {}  col: {} what: {}  where: {}  any_case: {}  extract: {}

    :Required...\n
    {}
    """
    err1 = "\nField not found:\nQuery fields: {}\nArray fields: {}"
    errors = [a.dtype.names is None,
              col is None, what is None,
              where.lower()[:2] not in ('en', 'eq', 'in', 'st'),
              col not in a.dtype.names]
    if sum(errors) > 0:
        arg = [a.dtype.kind, col, what, where, any_case, pull, find_in.__doc__]
        print(dedent(e0).format(*arg))
        return None
    if isinstance(pull, (list, tuple)):
        names = a.dtype.names
        r = [i in names for i in pull]
        if sum(r) != len(r):
            print(err1.format(pull, names))
            return None
    # ---- query section
    # convert column values and query to lowercase, if text, then query
    c = a[col]
    if c.dtype.kind in ('i', 'f', 'c'):
        c = c.astype('U')
        what = str(what)
    elif any_case:
        c = np.char.lower(c)
        what = what.lower()
    where = where.lower()[0]
    if where == 'i':
        q = np.char.find(c, what) >= 0   # ---- is in query ----
    elif where == 's':
        q = np.char.startswith(c, what)  # ---- startswith query ----
    elif where == 'eq':
        q = np.char.equal(c, what)
    elif where == 'en':
        q = np.char.endswith(c, what)    # ---- endswith query ----
    if q.sum() == 0:
        print("none found")
        return None
    if pull == 'all':
        return a[q]
    pull = np.unique([col] + list(pull))
    return a[q][pull]


def tbl_count(a, row=None, col=None, verbose=False):
    """Crosstabulate 2 fields data arrays, shape (N,), using np.unique.
    scipy.sparse has similar functionality and is faster for large arrays.

    Requires:
    --------
    A 2D array of data with shape(N,) representing two variables.

    row : field/column
        The table column/field to use for the row variable
    col : field/column
        The table column/field to use for thecolumn variable

    Notes:  See useage section above for converting Arc* tables to arrays.

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
    names = a.dtype.names
    assert row in names, "The.. {} ..column, not found in array.".format(row)
    assert col in names, "The.. {} ..column, not found in array.".format(col)
    r_vals = a[row]
    c_vals = a[col]
    dt = np.dtype([(row, r_vals.dtype), (col, c_vals.dtype)])
    rc = np.asarray(list(zip(r_vals, c_vals)), dtype=dt)
    u, idx, cnt = np.unique(rc, return_index=True, return_counts=True)
    rcc_dt = u.dtype.descr
    rcc_dt.append(('Count', '<i4'))
    ctab = np.asarray(list(zip(u[row], u[col], cnt)), dtype=rcc_dt)
    if verbose:
        prn(ctab)
    else:
        return ctab


def tbl_sum(a, row=None, col=None, val_fld=None):
    """Tabular sum of values for two attributes

    Parameters:
    ----------
    a : array
        Structured/recarray
    row, col : string
        The fields to be used as the table rows and columns
    val_fld : string
        The field that will be summed for the unique combinations of
        row/column classes

    Returns:
    --------
    A table summarizing the sums for the row/column combinations.
    """
    # ---- Slice the input array using the row/column fields, determine the
    # unique combinations of their attributes.  Create the output dtype
    names = a.dtype.names
    assert row in names, "The.. {} ..column, not found in array.".format(row)
    assert col in names, "The.. {} ..column, not found in array.".format(col)
    val_kind = a[val_fld].dtype.kind
    if val_kind not in ('i', 'f'):
        print("\nThe value field must be numeric")
        return None
    if val_kind == 'f':
        val_type = '<f8'
    elif val_kind == 'i':
        val_type = '<i4'
    rc = a[[row, col]]
    sum_name = val_fld +'_sum'
    dt = rc.dtype.descr + [(sum_name, val_type)]
    uniq = np.unique(rc)
    #
    # ----
    out_ = []
    for u in uniq:
        c0, c1 = u
        idx = np.logical_and(a[row] == c0, a[col] == c1)
        val = np.nansum(a[val_fld][idx])
        out_.append([c0, c1, val])
    out_ = np.array(out_)
    z = np.empty((len(out_),), dtype=dt)
    z[row] = out_[:, 0]
    z[col] = out_[:, 1]
    z[sum_name] = out_[:, 2].astype(val_kind)
    return z


# ---- crosstab from tool, uncomment for testing or tool use
def _demo():
    """Load the sample file for testing
    """
    # script = sys.argv[0]  # the script path defined earlier
    in_tbl = script.rpartition("/")[0] + '/Data/sample_20.npy'  # sample_20.npy
    a = np.load(in_tbl)
    ctab = tbl_count(a, row='County', col='Town', verbose=True)
    return a, ctab

def _data():
    """base file"""
    in_tbl = script.rpartition("/")[0] + '/Data/points_2000.npy'
    a = np.load(in_tbl)
    return a

if __name__ == "__main__":
    """run crosstabulation with data"""
#    ctab, counts, out_tbl = tab_count(a['County'], a['Town'],
#    r_fld='County', c_fld='Town', verbose=False)
#    ctab, a, result, r, c = _demo()
