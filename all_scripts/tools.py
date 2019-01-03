# -*- coding: UTF-8 -*-
"""
arraytools tools
================

Script :   tools.py

Author :   Dan_Patterson@carleton.ca

Modified : 2018-11-23

Purpose :  tools for working with numpy arrays

Useage:
-------

>>> import arraytools as art

- `tools.py` and other scripts are part of the arraytools package.
- Access in other programs using .... art.func(params) ....

**Requires**
-------------
  see import section and __init__.py in the `arraytools` folder

**Notes**
---------
**Basic array information**

- `np.typecodes`
- `np.sctypeDict`
- `np.sctypes`

>>> for i in np.sctypes:
    print("{!s:<8} : {}".format(i, np.sctypes[i]))

**Classes available**
::

int      : class
    'numpy.int8', 'numpy.int16', 'numpy.int32', 'numpy.int64'
uint     : class
    'numpy.uint8', 'numpy.uint16', 'numpy.uint32, 'numpy.uint64'
float    : class
    'numpy.float16', 'numpy.float32', 'numpy.float64'
complex  : class
    'numpy.complex64', 'numpy.complex128'
others   : class
    'bool', 'object', 'bytes', 'str', 'numpy.void'

**Typecodes**

`np.typecodes.items()`  ... `np.typecodes['AllInteger']`
::
    All: '?bhilqpBHILQPefdgFDGSUVOMm'
    |__AllFloat: 'efdgFDG'
       |__Float: 'efdg'
       |__Complex: 'FDG'
    |__AllInteger: 'bBhHiIlLqQpP'
    |  |__UnsignedInteger: 'BHILQP'
    |  |__Integer: 'bhilqp'
    |__Datetime': 'Mm'
    |__Character': 'c'
    |__Other:  'U', '?', 'S', 'O', 'V'  Determined from the above

`np.sctypes.keys` and `np.sctypes.values`
::
    numpy classes
    |__complex  complex64, complex128, complex256
    |__float    float16, float32, float64, float128
    |__int      int8, int16, int32, int64
    |__uint     uint8, uint16, uint32, uint63
    |__others   bool, object, str, void
                   ?,   O,    S U,  V

**Numbers**
::
   np.inf, -np.inf
   np.iinfo(np.int8).min  or .max = -128, 128
   np.iinfo(np.int16).min or .max = -32768, 32768
   np.iinfo(np.int32).min or .max = -2147483648, max=2147483647
   np.finfo(np.float64)
   np.finfo(resolution=1e-15, min=-1.7976931348623157e+308,
            max=1.7976931348623157e+308, dtype=float64)


**Functions**
-------------
Tool function examples follow...

**1. arr2xyz** : convert 2/3D arrays into xyz values (ie 2D)

arr2xyz(a, verbose=False)** : convert an array to x,y,z values, using
row/column values for x and y
::
    a= np.arange(2*3).reshape(2,3)
    arr2xyz(a)
    array([[0, 0, 0],
           [1, 0, 1],
           [2, 0, 2],
           [0, 1, 3],
           [1, 1, 4],
           [2, 1, 5]])

**2. make_blocks** : create array blocks

>>> make_blocks(rows=2, cols=4, r=2, c=2, dt='int')
array([[0, 0, 1, 1, 2, 2, 3, 3],
       [0, 0, 1, 1, 2, 2, 3, 3],
       [4, 4, 5, 5, 6, 6, 7, 7],
       [4, 4, 5, 5, 6, 6, 7, 7]])

**3. group_vals(seq, stepsize=1)**
::
    seq = [1, 2, 4, 5, 8, 9, 10]
    stepsize = 1
    [array([1, 2]), array([4, 5]), array([ 8,  9, 10])]

**4. reclass(z, bins, new_bins, mask=False, mask_val=None)**

Reclass an array using existing class breaks (bins) and new bins both must be
in ascending order.
::
      z = np.arange(3*5).reshape(3,5)
      bins = [0, 5, 10, 15]
      new_bins = [1, 2, 3, 4]
      z_recl = reclass(z, bins, new_bins, mask=False, mask_val=None)
      ==> .... z                     ==> .... z_recl
      array([[ 0,  1,  2,  3,  4],   array([[1, 1, 1, 1, 1],
             [ 5,  6,  7,  8,  9],          [2, 2, 2, 2, 2],
             [10, 11, 12, 13, 14]])         [3, 3, 3, 3, 3]])

**5. scale(a, x=2, y=2)** : scale an array by x, y factors
::
      a = np.array([[0, 1, 2], [3, 4, 5]]
      b = scale(a, x=2, y=2)
        =  array([[0, 0, 1, 1, 2, 2],
                  [0, 0, 1, 1, 2, 2],
                  [3, 3, 4, 4, 5, 5],
                  [3, 3, 4, 4, 5, 5]])

using scale with np.tile
::
      art.scale(a, 2,2)         np.tile(art.scale(a, 2, 2), (2, 2))
      array([[0, 0, 1, 1],      array([[0, 0, 1, 1, 0, 0, 1, 1],
             [0, 0, 1, 1],             [0, 0, 1, 1, 0, 0, 1, 1],
             [2, 2, 3, 3],             [2, 2, 3, 3, 2, 2, 3, 3],
             [2, 2, 3, 3]])            [2, 2, 3, 3, 2, 2, 3, 3],
                                       [0, 0, 1, 1, 0, 0, 1, 1],
                                       [0, 0, 1, 1, 0, 0, 1, 1],
                                       [2, 2, 3, 3, 2, 2, 3, 3],
                                       [2, 2, 3, 3, 2, 2, 3, 3]])

**6. split_array(a, fld='Id')**
::
     array 'b'
     array([(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11)],
           dtype=[('A', '<i4'), ('B', '<i4'), ('C', '<i4'), ('D', '<i4')])
    - split_array(b, fld='A')
    [array([(0, 1, 2, 3)],
          dtype=[('A', '<i4'), ('B', '<i4'), ('C', '<i4'), ('D', '<i4')]),
     array([(4, 5, 6, 7)],
          dtype=[('A', '<i4'), ('B', '<i4'), ('C', '<i4'), ('D', '<i4')]),
     array([(8, 9, 10, 11)],
          dtype=[('A', '<i4'), ('B', '<i4'), ('C', '<i4'), ('D', '<i4')])]

**7.  make_flds(n=1, as_type=names=None, default="col")** : example

>>> from numpy.lib._iotools import easy_dtype as easy
>>> make_flds(n=1, as_type='float', names=None, def_name="col")
dtype([('col_00', '<f8')])

>>> make_flds(n=2, as_type='int', names=['f01', 'f02'], def_name="col")
dtype([('f01', '<i8'), ('f02', '<i8')])

**8.  nd_rec** : ndarray to structured array or recarray

**9.  nd_struct** :

**10. nd2struct(a)**

Keep the dtype the same
::
    aa = nd2struct(a)       # produce a structured array from inputs
    aa.reshape(-1,1)   # structured array
    array([[(0, 1, 2, 3, 4)],
           [(5, 6, 7, 8, 9)],
           [(10, 11, 12, 13, 14)],
           [(15, 16, 17, 18, 19)]],
       dtype=[('A', '<i4'), ... snip ... , ('E', '<i4')])

Upcast the dtype
::
    a_f = nd2struct(a.astype('float'))  # note astype allows a view
    array([(0.0, 1.0, 2.0, 3.0, 4.0), ... snip... ,
           (15.0, 16.0, 17.0, 18.0, 19.0)],
          dtype=[('A', '<f8'), ... snip ... , ('E', '<f8')])

**11. np2rec** : shell around above

**12. rc_vals(a)**

**13. xy_vals(a) ... array to x, y, values**

**14. array_cols**

**15. change_arr(a, order=[], prn=False)** : merely a convenience function
::
    a = np.arange(4*5).reshape((4, 5))
    change(a, [2, 1, 0, 3, 4])
    array([[ 2,  1,  0,  3,  4],
           [ 7,  6,  5,  8,  9],
           [12, 11, 10, 13, 14],
           [17, 16, 15, 18, 19]])

**shortcuts**
::
    b = a[:, [2, 1, 0, 3, 4]]    # reorder the columns, keeping the rows
    c = a[:, [0, 2, 3]]          # delete columns 1 and 4
    d = a[[0, 1, 3, 4], :]       # delete row 2, keeping the columns
    e = a[[0, 1, 3], [1, 2, 3]]  # keep [0, 1], [1, 2], [3, 3]
                                   => ([ 1, 7, 18])

**16. concat_arrs**

**17. pad__(a, pad_with=None, size=(1, 1))**

**18. stride(a, r_c=(3, 3))**

Produce a strided array using a window of r_c shape.

Calls _check(a, r_c, subok=False) to check for array compliance
::
      a =np.arange(15).reshape(3,5)
      s = stride(a)    stride     ====>   slide    =====>
      array([[[ 0,  1,  2],  [[ 1,  2,  3],  [[ 2,  3,  4],
              [ 5,  6,  7],   [ 6,  7,  8],   [ 7,  8,  9],
              [10, 11, 12]],  [11, 12, 13]],  [12, 13, 14]]])

`pad_`  to pad an array prior to striding or blocking

`block`  calls stride with non-overlapping blocks with no padding


**19. block(a, win=(3, 3))**

**20. sliding_window_view**

**21.  block_arr(a, win=[3, 3], nodata=-1)**

Block an array given an input array, a window and a nodata value.
::
    a = np.arange(16).reshape(4,4)
    block_arr(a, win=[4, 3], nodata=-1)
    array([[ 0,  1,  2,  3, -1, -1],
           [ 4,  5,  6,  7, -1, -1],
           [ 8,  9, 10, 11, -1, -1],
           [12, 13, 14, 15, -1, -1]]),
    masked_array(data =
        [[[0 1 2]
          [4 5 6]
          [8 9 10]
          [12 13 14]]

         [[3 -- --]
          [7 -- --]
          [11 -- --]
          [15 -- --]]],
    mask .... snipped ....

**22. rolling_stats() : stats for a strided array**

    min, max, mean, sum, std, var, ptp

**23. find(a, func, this=None, count=0, keep=[], prn=False, r_lim=2)**

    func - (cumsum, eq, neq, ls, lseq, gt, gteq, btwn, btwni, byond)
           (        ==,  !=,  <,   <=,  >,   >=,  >a<, =>a<=,  <a> )

**23a. _func(fn, a, this)**

    called by 'find' see details there
    (cumsum, eq, neq, ls, lseq, gt, gteq, btwn, btwni, byond)

Note  see ``find1d_demo.py`` for examples


**24. group_pnts(a, key_fld='ID', keep_flds=['X', 'Y', 'Z'])**

**25. uniq(ar, return_index=False, return_inverse=False, return_counts=False,
          axis=0)**

**26. is_in(find_in, using, not_in=False)**

**27. running_count**

**28. sequences(data, stepsize)**

**29. sort_rows_by_col(a, col=0, descending=False)**

Sort 2d ndarray by column
::
      a                           col_sort(a, col=1, descending=False)
      array([[2, 3, 2, 2],        array([[2, 1, 2, 4],
             [1, 4, 1, 3],               [2, 3, 2, 2],
             [2, 1, 2, 4]])              [1, 4, 1, 3]])

**30. sort_cols_by_row**

**31. radial_sort(pnts, cent=None)**

**32. pack_last_axis**


References:
----------
general

`<https://github.com/numpy/numpy>`_.
`<https://github.com/numpy/numpy/blob/master/numpy/lib/_iotools.py>`_.

striding

`<https://github.com/numpy/numpy/blob/master/numpy/lib/stride_tricks.py>`_.
`<http://www.johnvinyard.com/blog/?p=268>`_.

for strided arrays

`<https://stackoverflow.com/questions/47469947/as-strided-linking-stepsize-
strides-of-conv2d-with-as-strided-strides-paramet#47470711>`_.

`<https://stackoverflow.com/questions/48097941/strided-convolution-of-2d-in-
numpy>`_.

`<https://stackoverflow.com/questions/45960192/using-numpy-as-strided-
function-to-create-patches-tiles-rolling-or-sliding-w>`_.

numpy  # stride for convolve 4d

`<https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-
column>`_.

Functions
---------
Alphabetical listing
::

 '_func', '_tools_help_', 'arr2xyz', 'arrays_cols', 'block', 'block_arr',
 'change_arr', 'concat_arrs', 'find', 'group_pnts', 'group_vals', 'is_in',
 'make_blocks', 'make_flds', 'n_largest', 'n_smallest', 'nd2rec',
 'nd2struct', 'nd_rec', 'nd_struct', 'num_to_mask', 'num_to_nan',
 'pack_last_axis', 'pad_', 'radial_sort', 'rc_vals', 'reclass',
 'rolling_stats', 'running_count', 'scale', 'sequences',
 'sliding_window_view', 'sort_cols_by_row', 'sort_rows_by_col',
 'split_array', 'stride', 'uniq', 'view_sort', 'xy_sort','xy_vals'

---------------------------------------------------------------------
"""
# pylint: disable=C0103
# pylint: disable=R1710
# pylint: disable=R0914

# ---- imports, formats, constants -------------------------------------------
import sys
from textwrap import dedent, indent
import warnings
import numpy as np
from numpy.lib.stride_tricks import as_strided

warnings.simplefilter('ignore', FutureWarning)

__all__ = ['_tools_help_',
           'arr2xyz', 'make_blocks',     # (1-6) ndarrays ... make arrays,
           'group_vals', 'reclass',      #     change shape, arangement
           'scale', 'split_array',
           'make_flds', 'nd_rec',        # (7-14) structured/recdarray
           'nd_struct', 'nd2struct',
           'nd2rec', 'rc_vals', 'xy_vals',
           'arrays_struct',
           'change_arr', 'concat_arrs',  # (15-16) change/modify arrays
           'pad_', 'stride', 'block',    # (17-22) stride, block and pad
           'sliding_window_view',
           'block_arr', 'rolling_stats',
           '_func', 'find', 'group_pnts', # (23-28) querying, analysis
           'uniq', 'is_in',
           'running_count', 'sequences',
           'sort_cols_by_row',            # (29-31) column and row sorting
           'sort_rows_by_col',
           'radial_sort',
           'view_sort', 'xy_sort',
           'pack_last_axis'  # extras -------
           ]


ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 8.2f}'.format}

np.set_printoptions(
        edgeitems=3,
        threshold=120,
        floatmode='maxprec',
        precision=2, suppress=True, linewidth=100,
        nanstr='nan', infstr='inf', sign='-',
        formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script



# ---- (1) ndarrays ... code section .... ----
# ---- make arrays, change shape, arrangement
# ---- arr2xyz, makeblocks, rc_vals, xy_vals ----
#
def arr2xyz(a, keep_masked=False, verbose=False):
    """Produce an array such that the row, column values are used for x,y
    and array values for z.  Masked arrays are sorted.

    Returns
    --------
    A mesh grid with values, dimensions and shapes are changed so
    that ndim=2, ie shape(3,4,5), ndim=3 becomes shape(12,5), ndim=2

    >>> a = np.arange(9).reshape(3, 3)
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> arr2xyz(am, keep_masked=True)   # keep the masked values...
    masked_array(data =
    [[0 0 0]
     [1 0 -]
     [2 0 2]
     [0 1 -]
     [1 1 4]
     [2 1 -]
     [0 2 6]
     [1 2 7]
     [2 2 8]],
             mask =
     [[False False False]... snip
     [False False False]],
           fill_value = 999999)
    >>>
    >>> arr2xyz(am, keep_masked=False)  # remove the masked values
    array([[0, 0, 0],
           [2, 0, 2],
           [1, 1, 4],
           [0, 2, 6],
           [1, 2, 7],
           [2, 2, 8]])

    See also
    --------
    `xy_vals(a)` and `rc_vals(a)` for simpler versions.

    `num_to_mask(a)` and  `num_to_nan(a)` if you want structured arrays,
    to produce masks prior to conversion.

    """
    if a.ndim == 1:
        a = a.reshape(a.shape[0], 1)
    if a.ndim > 2:
        a = a.reshape(np.product(a.shape[:-1]), a.shape[-1])
    r, c = a.shape
    XX, YY = np.meshgrid(np.arange(c), np.arange(r))
    XX = XX.ravel()
    YY = YY.ravel()
    if isinstance(np.ma.getmask(a), np.ndarray):
        tbl = np.ma.vstack((XX, YY, a.ravel()))
        tbl = tbl.T
        if not keep_masked:
            m = tbl[:, 2].mask
            tbl = tbl[~m].data
    else:
        tbl = np.stack((XX, YY, a.ravel()), axis=1)
    if verbose:
        frmt = """
        ----------------------------
        Meshgrid demo: array to x,y,z table
        :Formulation...
        :  XX,YY = np.meshgrid(np.arange(x.shape[1]),np.arange(x.shape[0]))
        :Input table
        {!r:<}
        :Raveled array, using x.ravel()
        {!r:<}
        :XX in mesh: columns shape[1]
        {!r:<}
        :YY in mesh: rows shape[0]
        {!r:<}
        :Output:
        {!r:<}
        :-----------------------------
        """
        print(dedent(frmt).format(a, a.ravel(), XX, YY, tbl))
    else:
        return tbl


def make_blocks(rows=3, cols=3, r=2, c=2, dt='int'):
    """Make a block array with rows * cols containing r*c sub windows.
    Specify the rows, columns, then the block size as r, c and dtype
    Use `scale`, if you want specific values during array construction.

    Requires
    --------
    rows : integer
        rows in initial array
    cols : integer
        columns in the initial array
    r : integer
        rows in sub window
    c : integer
        columns in sub window
    dt : np.dtype
        array data type

    Returns
    --------
    The defaults produce an 8 column by 8 row array numbered from
    0 to (rows*cols) - 1

    >>> array.shape = (rows * r, cols * c)  # (6, 6)

    >>> make_blocks(rows=3, cols=3, r=2, c=2, dt='int')
    array([[0, 0, 1, 1, 2, 2],
           [0, 0, 1, 1, 2, 2],
           [3, 3, 4, 4, 5, 5],
           [3, 3, 4, 4, 5, 5],
           [6, 6, 7, 7, 8, 8],
           [6, 6, 7, 7, 8, 8]])

    """
    a = np.arange(rows*cols, dtype=dt).reshape(rows, cols)
    a = scale(a, x=r, y=c)
    return a


def group_vals(seq, delta=0, oper='!='):
    """Group consecutive values separated by no more than delta

    Parameters
    ----------
    seq : array, list tuple
        sequence of values
    delta :
        difference between consecutive values
    oper :
        'eq', '==', 'ne', '!=', 'gt', '>', 'lt', '<'

    Reference
    ---------
    `<https://stackoverflow.com/questions/7352684/
    how-to-find-the-groups-of-consecutive-elements-from-an-array-in-numpy>`_.

    Notes
    -----
    >>> a = [1, 1, 1, 2, 2, 3, 1, 1, 1]
    >>> group_vals(a, delta=0, oper='!=')  # sequential difference !=0
    [array([1, 1, 1]), array([2, 2]), array([3]), array([1, 1, 1])]

    See also
    --------
    split_array : form structured or recarrays
    """
    valid = ('eq', '==', 'ne', '!=', 'gt', '>', 'lt', '<')
    if oper not in valid:
        raise ValueError("operand not in {}".format(valid))
    elif oper in ('==', 'eq'):
        s = np.split(seq, np.where(np.diff(seq) == delta)[0]+1)
    elif oper in ('!=', 'ne'):
        s = np.split(seq, np.where(np.diff(seq) != delta)[0]+1)
    elif oper in ('>', 'gt'):
        s = np.split(seq, np.where(np.diff(seq) > delta)[0]+1)
    elif oper in ('<', 'lt'):
        s = np.split(seq, np.where(np.diff(seq) < delta)[0]+1)
    else:
        s = seq
    return s


def reclass(a, bins=None, new_bins=None, mask_=False, mask_val=None):
    """Reclass an array of integer or floating point values.

    Requires:
    --------
    bins : list/tuple
        sequential list/array of the lower limits of each class
        include one value higher to cover the upper range.
    new_bins : list/tuple
        new class values for each bin
    mask : boolean
        whether the raster contains nodata values or values to
        be masked with mask_val
    mask_val: number
        value to be masked

    Array dimensions will be squeezed.

    Example
    -------
    inputs::

        z = np.arange(3*5).reshape(3,5)
        bins = [0, 5, 10, 15]
        new_bins = [1, 2, 3, 4]
        z_recl = reclass(z, bins, new_bins, mask=False, mask_val=None)

    outputs::

        ==> .... z                     ==> .... z_recl
        array([[ 0,  1,  2,  3,  4],   array([[1, 1, 1, 1, 1],
               [ 5,  6,  7,  8,  9],          [2, 2, 2, 2, 2],
               [10, 11, 12, 13, 14]])         [3, 3, 3, 3, 3]])

    """
    a_rc = np.zeros_like(a)
    c_0 = isinstance(bins, (list, tuple))
    c_1 = isinstance(new_bins, (list, tuple))
    err = "Bins = {} new = {} won't work".format(bins, new_bins)
    if not c_0 or not c_1:
        print(err)
        return a
    if len(bins) < 2:  # or (len(new_bins <2)):
        print(err)
        return a
    if len(new_bins) < 2:
        new_bins = np.arange(1, len(bins)+2)
    new_classes = list(zip(bins[:-1], bins[1:], new_bins))
    for rc in new_classes:
        q1 = (a >= rc[0])
        q2 = (a < rc[1])
        a_rc = a_rc + np.where(q1 & q2, rc[2], 0)
    return a_rc


def scale(a, x=2, y=2, num_z=None):
    """Scale the input array repeating the array values up by the
    x and y factors.

    Parameters:
    ----------
    `a` : An ndarray, 1D arrays will be upcast to 2D.

    `x y` : Factors to scale the array in x (col) and y (row).  Scale factors
    must be greater than 2.

    `num_z` : For 3D, produces the 3rd dimension, ie. if num_z = 3 with the
    defaults, you will get an array with shape=(3, 6, 6),

    Examples:
    --------
    >>> a = np.array([[0, 1, 2], [3, 4, 5]]
    >>> b = scale(a, x=2, y=2)
    array([[0, 0, 1, 1, 2, 2],
           [0, 0, 1, 1, 2, 2],
           [3, 3, 4, 4, 5, 5],
           [3, 3, 4, 4, 5, 5]])

    Notes:
    -----
    >>> a = np.arange(2*2).reshape(2,2)
    array([[0, 1],
           [2, 3]])

    >>> frmt_(scale(a, x=2, y=2, num_z=2))
    Array... shape (3, 4, 4), ndim 3, not masked
      0, 0, 1, 1    0, 0, 1, 1    0, 0, 1, 1
      0, 0, 1, 1    0, 0, 1, 1    0, 0, 1, 1
      2, 2, 3, 3    2, 2, 3, 3    2, 2, 3, 3
      2, 2, 3, 3    2, 2, 3, 3    2, 2, 3, 3
      sub (0)       sub (1)       sub (2)

    """
    if (x < 1) or (y < 1):
        print("x or y scale < 1... read the docs\n{}".format(scale.__doc__))
        return None
    a = np.atleast_2d(a)
    z0 = np.tile(a.repeat(x), y)  # repeat for x, then tile
    z1 = np.hsplit(z0, y)         # split into y parts horizontally
    z2 = np.vstack(z1)            # stack them vertically
    if a.shape[0] > 1:            # if there are more, repeat
        z3 = np.hsplit(z2, a.shape[0])
        z3 = np.vstack(z3)
    else:
        z3 = np.vstack(z2)
    if num_z not in (0, None):
        d = [z3]
        for i in range(num_z):
            d.append(z3)
        z3 = np.dstack(d)
        z3 = np.rollaxis(z3, 2, 0)
    return z3


def split_array(a, fld='ID'):
    """Split a structured or recarray array using unique values in the
    `fld` field.  It is assumed that there is a sequential ordering to
    the values in the field.  If there is not, use np.where in conjunction
    with np.unique or sort the array first.

    Parameters
    ----------
    `a` : A structured or recarray.

    `fld` : A numeric field assumed to be sorted which indicates which group
    a record belongs to.

    Returns
    -------
    A list of arrays split on the categorizing field

    """
    return np.split(a, np.where(np.diff(a[fld]))[0] + 1)



# ----------------------------------------------------------------------
# ---- (2) structured/recdarray section, change format or arrangement ----
# ----------------------------------------------------------------------
# ---- make_flds, nd_rec, nd_struct, nd_struct, np2rec, rc_vals, xy_vals
#
def make_flds(n=2, as_type='float', names=None, def_name="col"):
    """Create float or integer fields for statistics and their names.

    Requires
    --------
    n : integer
        number of fields to create excluding the names field
    def_name : string
        base name to use, numeric values will be produced for each dimension
        for the 3D array, ie Values_00... Values_nn

    Returns
    -------
    a dtype : which contains the necessary fields to contain the values.

    >>> from numpy.lib._iotools import easy_dtype as easy
    >>> make_flds(n=1, as_type='float', names=None, def_name="col")
    dtype([('col_00', '<f8')])

    >>> make_flds(n=2, as_type='int', names=['f01', 'f02'], def_name="col")
    dtype([('f01', '<i8'), ('f02', '<i8')])

    Don't forget the above, a cool way to create fields quickly

    """
    from numpy.lib._iotools import easy_dtype as easy
    if as_type in ['float', 'f8', '<f8']:
        as_type = '<f8'
    elif as_type in ['int', 'i4', 'i8', '<i4', '<i8']:
        as_type = '<i8'
    else:
        as_type = 'str'
    f = ",".join([as_type for i in range(n)])
    if names is None:
        names = ", ".join(["{}_{:>02}".format(def_name, i) for i in range(n)])
        dt = easy(f, names=names, defaultfmt=def_name)
    else:
        dt = easy(f, names=names)
    return dt


def nd_rec(a, flds=None, types=None):
    """Change a uniform array to an array of mixed dtype as a recarray

    Requires:
    ---------
    flds : string or None
        flds='a, b, c'
    types : string or None
        types='U8, f8, i8'

    See also:
    ---------
    nd_struct : alternate using lists rather than string inputs

    Notes:
    -----
    The a.T turns the columns to rows so that each row can be assigned a
    separate data type.

    Example::

       a = np.arange(9).reshape(3, 3)
       a_r = nd_rec(a, flds='a, b, c', types='U8, f8, i8')
       a_r
       rec.array([('0',  1., 2), ('3',  4., 5), ('6',  7., 8)],
          dtype=[('a', '<U8'), ('b', '<f8'), ('c', '<i8')])

    """
    _, c = a.shape
    if flds is None:
        flds = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[:c]
        flds = ", ".join([n for n in flds])
    if types is None:
        types = a.dtype.str  # a.dtype.descr[0][1]
        types = ", ".join(["{}".format(types) for i in range(c)])
    a_r = np.core.records.fromarrays(a.transpose(),
                                     names=flds,
                                     formats=types)
    return a_r


def nd_struct(a, flds=None, types=None):
    """"Change an array with uniform dtype to an array of mixed dtype as a
    structured array.

    Requires:
    ---------
    flds : list or None
        flds=['A', 'B', 'C']
    types : list or None
        types=['U8', 'f8', 'i8']

    See also:
    ---------
    nd_rec : alternate using strings rather than list inputs

    Example::

        a = np.arange(9).reshape(3, 3)
        a_s = nd_struct(a, flds=['A', 'B', 'C'], types=['U8', 'f8', 'i8'])
        a_s
        array([('0',  1., 2), ('3',  4., 5), ('6',  7., 8)],
              dtype=[('A', '<U8'), ('B', '<f8'), ('C', '<i8')])

    Timing of nd_rec and nd_struct

    >>> %timeit nd_rec(a, flds='a, b, c', types='U8, f8, i8')
    465 µs ± 53 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

    >>> %timeit nd_struct(a, flds=['A', 'B', 'C'], types=['U8', 'f8', 'i8'])
    253 µs ± 27.1 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    """
    _, c = a.shape
    dt_base = [a.dtype.str] * c  # a.dtype.descr[0][1]
    if flds is None:
        flds = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[:c]
    if types is None:
        types = dt_base
    dt0 = np.dtype(list(zip(flds, dt_base)))
    dt1 = list(zip(flds, types))
    a_s = a.view(dtype=dt0).squeeze(axis=-1).astype(dt1)
    return a_s


def nd2struct(a, fld_names=None):
    """Return a view of an ndarray as structured array with a uniform dtype/

    Parameters
    ----------
    a : array
        ndarray with a uniform dtype.
    fld_names : list
        A list of strings one for each column/field.  If none are provided,
        then the field names are assigned from an alphabetical list up to 26
        fields.  The dtype of the input array is retained, but can be upcast.

    Examples
    --------
    >>> a = np.arange(2*3).reshape(2,3)
    array([[0, 1, 2],
           [3, 4, 5]])  # dtype('int64')
    >>> b = nd2struct(a)
    array([(0, 1, 2), (3, 4, 5)],
          dtype=[('A', '<i8'), ('B', '<i8'), ('C', '<i8')])
    >>> c = nd2struct(a.astype(np.float64))
    array([( 0.,  1.,  2.), ( 3.,  4.,  5.)],
          dtype=[('A', '<f8'), ('B', '<f8'), ('C', '<f8')])

    See Also
    --------
    pack_last_axis(arr, names=None) at the end

    """
    if a.dtype.names:  # return if a structured array already
        return a
    alph = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if a.ndim != 2:
        frmt = "Wrong array shape... read the docs..\n{}"
        print(frmt.format(nd2struct.__doc__))
        return a
    _, cols = a.shape
    if fld_names is None:
        names = list(alph)[:cols]
    elif (len(fld_names) == cols) and (cols < 26):
        names = fld_names
    else:  # from... pack_last_axis
        names = ['f{:02.0f}'.format(i) for i in range(cols)]
    return a.view([(n, a.dtype) for n in names]).squeeze(-1)


def nd2rec(a, fld_names=None):
    """Shell to nd2struct but yielding a recarray.
    """
    a = nd2struct(a, fld_names=fld_names)
    return a.view(type=np.recarray)


def rc_vals(a):
    """Convert array to rcv, for 2D arrays.  See xy_val for details.
    """
    r, c = a.shape
    n = r * c
    x, y = np.meshgrid(np.arange(c), np.arange(r))
    dt = [('Row', '<i8'), ('Col', '<i8'), ('Val', a.dtype.str)]
    out = np.zeros((n,), dtype=dt)
    out['Row'] = x.ravel()
    out['Col'] = y.ravel()
    out['Val'] = a.ravel()
    return out


def xy_vals(a):
    """Convert array to xyz, for 2D arrays

    Parameters:
    -----------
    a : array
        2D array of values
    Returns:
    --------
    Triplets of x, y and vals as an nx3 array

    >>> a = np.random.randint(1,5,size=(2,4))
    >>> a
    array([[4, 1, 4, 3],
           [2, 3, 1, 3]])
    >>> xy_val(a)
    array([(0, 0, 4), (1, 0, 1), (2, 0, 4), (3, 0, 3),
           (0, 1, 2), (1, 1, 3), (2, 1, 1), (3, 1, 3)],
          dtype=[('X', '<i8'), ('Y', '<i8'), ('Val', '<i4')])
    """
    r, c = a.shape
    n = r * c
    x, y = np.meshgrid(np.arange(c), np.arange(r))
    dt = [('X', '<i8'), ('Y', '<i8'), ('Val', a.dtype.str)]
    out = np.zeros((n,), dtype=dt)
    out['X'] = x.ravel()
    out['Y'] = y.ravel()
    out['Val'] = a.ravel()
    return out


# ---- arrays_cols ----
def arrays_struct(arrs):
    """Stack arrays of any dtype to form a structured array, stacked in
    columns format.
    """
    if len(arrs) < 2:
        return arrs
    out_dt = [i.dtype.descr[0] for i in arrs]
    N = arrs[0].shape[0]
    out = np.empty((N,), dtype=out_dt)
    names = np.dtype(out_dt).names
    for i in range(len(names)):
        out[names[i]] = arrs[i]
    return out


# ----------------------------------------------------------------------------
# ---- (3) change/modify arrays ... code section ----
# ---- change_arr, scale, split_array, concat_arrs
#
def change_arr(a, order=None, prn=False):
    """Reorder and/or drop columns in an ndarray or structured array.

    Fields not included will be dropped in the output array.

    Parameters
    ----------
    order : list of fields
        fields in the order that you want them
    prn : boolean
        True, prints additional information prior to returning the array

    Notes:
    ------
    *reorder fields : ['a', 'c', 'b']*
        For a structured/recarray, the desired field order is required.
        An ndarray, not using named fields, will require the numerical
        order of the fields.

    *remove fields : ['a', 'c']*   ...  `b` is dropped
        To remove fields, simply leave them out of the list.  The
        order of the remaining fields will be reflected in the output.
        This is a convenience function.... see the module header for
        one-liner syntax.

    Tip :
        Use... `arraytools._base_functions.arr_info(a, verbose=True)`
        This gives field names which can be copied for use here.

    """
    if order is None or (not isinstance(order, (list, tuple))):
        print("Order not given in a list or tuple")
        return a
    names = a.dtype.names
    if names is None:
        b = a[:, order]
    else:
        out_flds = []
        out_flds = [i for i in order if i in names]
        if prn:
            missing = [i for i in names if i not in order]
            missing.extend([i for i in order if i not in out_flds])
            frmt = """
            : change(a)
            : - field(s) {}
            : - not found, missing or removed.
            """
            print(dedent(frmt).format(missing))
        b = a[out_flds]
    return b


def concat_arrs(arrs, sep=" ", name=None, with_ids=True):
    """Concatenate a sequence of arrays to string format and return a
    structured array or ndarray

    arrs : list
        A list of single arrays of the same length
    sep : string
        The separator between lists
    name : string
        A default name used for constructing the array field names.
    """
    N = len(arrs)
    if N < 2:
        return arrs
    a, b = arrs[0], arrs[1]
    c = ["{}{}{}".format(i, sep, j) for i, j in list(zip(a, b))]
    if N > 2:
        for i in range(2, len(arrs)):
            c = ["{}{}{}".format(i, sep, j) for i, j in list(zip(c, arrs[i]))]
    c = np.asarray(c)
    sze = c.dtype.str
    if name is not None:
        c.dtype = [(name, sze)]
    else:
        name = 'f'
    if with_ids:
        tmp = np.copy(c)
        dt = [('IDs', '<i8'), (name, sze)]
        c = np.empty((tmp.shape[0], ), dtype=dt)
        c['IDs'] = np.arange(1, tmp.shape[0] + 1)
        c[name] = tmp
    return c


# ----------------------------------------------------------------------
# ---- (4) stride, block and pad .... code section
# ----  pad_, stride, sliding_window_view, block
#
def pad_(a, pad_with=None, size=(1, 1)):
    """To use when padding a strided array for window construction.

    Parameters:
    ----------
    pad_with : number
        Options for number types
    - ints : 0, +/-128, +/-32768 `np.iinfo(np.int16).min or max 8, 16, 32`.
    - float : 0., np.nan, np.inf, `-np.inf` or `np.finfo(float64).min or max`
    size : list/tuple
        Size of padding on sides in cells.
    - 2D : 1 cell => (1,1)
    - 3D : 1 cell => (1,1,1)
    """
    if pad_with is None:
        return a
    #
    new_shape = tuple(i+2 for i in a.shape)
    tmp = np.zeros(new_shape, dtype=a.dtype)
    tmp.fill(pad_with)
    if tmp.ndim == 2:
        tmp[1:-1, 1:-1] = a
    elif tmp.ndim == 3:
        tmp[1:-1, 1:-1, 1:-1] = a
    a = np.copy(tmp, order='C')
    del tmp
    return a


def pad_sides(a, TL=(0, 0), RB=(0, 0), value=0):
    """Pad an array's T(op), L(eft), B(ottom) and R(ight) sides with `value`.

    Parameters:
    -----------
    `pad_by` : tuple of integers
        Pad the T, B rows and L, R columns.
    `value` : integer
        Value to use on all axes

    >>> a np.array([[0, 1, 2], [3, 4, 5]])
    >>> pad_sides(a, (0, 1, 0, 1), -1)
    array([[ 0,  1,  2, -1],
           [ 3,  4,  5, -1],
           [-1, -1, -1, -1]])
    >>> pad_sides(a, (1, 0, 0, 2), -1)
    array([[-1, -1, -1, -1, -1],
           [ 0,  1,  2, -1, -1],
           [ 3,  4,  5, -1, -1]])
    """
    L, T = TL
    R, B = RB
    a = np.pad(a, pad_width=((T, B), (L, R)), mode='constant',
               constant_values=value)
    return a


def needed(a, win=(3, 3)):
    """pad size for right bottom padding given array shape and window size
    """
    shp = a.shape
    RB = np.remainder(shp, win)
    return RB


def stride(a, win=(3, 3), stepby=(1, 1)):
    """Provide a 2D sliding/moving view of an array.
    There is no edge correction for outputs. Use the `pad_` function first.

    Requires
    --------
    as_strided : function
        from numpy.lib.stride_tricks import as_strided
    a : array or list
        Usually a 2D array.  Assumes rows >=1, it is corrected as is the
        number of columns.
    win, stepby : array-like
        tuple/list/array of window strides by dimensions
    ::

        - 1D - (3,)       (1,)       3 elements, step by 1
        - 2D - (3, 3)     (1, 1)     3x3 window, step by 1 rows and col.
        - 3D - (1, 3, 3)  (1, 1, 1)  1x3x3, step by 1 row, col, depth

    Examples
    --------
    >>> a = np.arange(10)
    >>> # stride(a, (3,), (1,)) 3 value moving window, step by 1
    >>> stride(a, (3,), (2,))
    array([[0, 1, 2],
           [2, 3, 4],
           [4, 5, 6],
           [6, 7, 8]])
    >>> a = np.arange(6*6).reshape(6, 6)
    #    stride(a, (3, 3), (1, 1))  sliding window
    #    stride(a, (3, 3), (3, 3))  block an array

    Notes:
    -----
    - np.product(a.shape) == a.size   # shape product equals array size
    - To check if the base array and the strided version share memory
    - np.may_share_memory(a, a_s)     # True

    ----------------------------------------------------------
    """
    err = """Array shape, window and/or step size error.
    Use win=(3,) with stepby=(1,) for 1D array
    or win=(3,3) with stepby=(1,1) for 2D array
    or win=(1,3,3) with stepby=(1,1,1) for 3D
    ----    a.ndim != len(win) != len(stepby) ----
    """
    from numpy.lib.stride_tricks import as_strided
    a_ndim = a.ndim
    if isinstance(win, int):
        win = (win,) * a_ndim
    if isinstance(stepby, int):
        stepby = (stepby,) * a_ndim
    assert (a_ndim == len(win)) and (len(win) == len(stepby)), err
    shp = np.array(a.shape)    # array shape (r, c) or (d, r, c)
    win_shp = np.array(win)    # window      (3, 3) or (1, 3, 3)
    ss = np.array(stepby)      # step by     (1, 1) or (1, 1, 1)
    newshape = tuple(((shp - win_shp) // ss) + 1) + tuple(win_shp)
    newstrides = tuple(np.array(a.strides) * ss) + a.strides
    a_s = as_strided(a, shape=newshape, strides=newstrides, subok=True).squeeze()
    return a_s


# ---- sliding_window_view .... new ----
def sliding_window_view(x, shape=None):
    """Create rolling window views of the 2D array with the given shape.
    proposed for upcoming numpy version.
    """
    if shape is None:
        shape = x.shape
    o = np.array(x.shape) - np.array(shape) + 1  # output shape
    strides = x.strides
    view_shape = np.concatenate((o, shape), axis=0)
    view_strides = np.concatenate((strides, strides), axis=0)
    return np.lib.stride_tricks.as_strided(x, view_shape, view_strides)


def block(a, win=(3, 3)):
    """Calls stride with step_by equal to win size.
    No padding of the array, so this works best when win size is divisible
    in both directions

    Note:
        see block_arr if you want padding
    """
    a_b = stride(a, win=win, stepby=win)
    return a_b


def block_arr(a, win=[3, 3], nodata=-1, as_masked=False):
    """Block array into window sized chunks padding to the right and bottom
    to accommodate array and window shape.

    Parameters
    ----------
    `a` : array
        2D array
    `win` : [integer, integer]
        [rows, cols], aka y,x, m,n sized window
    `nodata` : number
        to use for the mask

    Returns
    -------
    The padded array and the masked array blocked.

    Reference
    ---------
    `<http://stackoverflow.com/questions/40275876/how-to-reshape-this-image-
    array-in-python>`_.

    >>> def block_2(a, blocks=2)
            B = blocks # Blocksize
            m, n = a.shape
            out = a.reshape(m//B, B, n//B, B).swapaxes(1, 2).reshape(-1, B, B)
            return out

    """
    s = np.array(a.shape)
    if len(win) != 2:
        print("\n....... Read the docs .....\n{}".format(block_arr.__doc__))
        return None
    win = np.asarray(win)
    m = divmod(s, win)
    s2 = win*m[0] + win*(m[1] != 0)
    ypad, xpad = s2 - a.shape
    pad = ((0, ypad), (0, xpad))
    p_with = ((nodata, nodata), (nodata, nodata))
    b = np.pad(a, pad_width=pad, mode='constant', constant_values=p_with)
    w_y, w_x = win       # Blocksize
    y, x = b.shape       # padded array
    c = b.reshape((y//w_y, w_y, x//w_x, w_x))
    c = c.swapaxes(1, 2).reshape(-1, w_y, w_x)
    if as_masked:
        c = np.ma.masked_equal(c, nodata)
        c.set_fill_value(nodata)
    return c


def rolling_stats(a, no_null=True, prn=True):
    """Statistics on the last two dimensions of an array.

    Requires
    --------
    a : array
        2D array  **Note, use 'stride' above to obtain rolling stats
    no_null : boolean
        Whether to use masked values (nan) or not.
    prn : boolean
        To print the results or return the values.

    Returns
    -------
    The results return an array of 4 dimensions representing the original
    array size and block size.  An original 6x6 array will be broken into
    block 4 3x3 chunks.
    """
    a = np.asarray(a)
    a = np.atleast_2d(a)
    ax = None
    if a.ndim > 1:
        ax = tuple(np.arange(len(a.shape))[-2:])
    if no_null:
        a_min = a.min(axis=ax)
        a_max = a.max(axis=ax)
        a_mean = a.mean(axis=ax)
        a_med = np.median(a, axis=ax)
        a_sum = a.sum(axis=ax)
        a_std = a.std(axis=ax)
        a_var = a.var(axis=ax)
        a_ptp = a_max - a_min
    else:
        a_min = np.nanmin(a, axis=(ax))
        a_max = np.nanmax(a, axis=(ax))
        a_mean = np.nanmean(a, axis=(ax))
        a_med = np.nanmedian(a, axis=(ax))
        a_sum = np.nansum(a, axis=(ax))
        a_std = np.nanstd(a, axis=(ax))
        a_var = np.nanvar(a, axis=(ax))
        a_ptp = a_max - a_min
    if prn:
        s = ['Min', 'Max', 'Mean', 'Med', 'Sum', 'Std', 'Var', 'Range']
        frmt = "...\n{}\n".join([i for i in s])
        v = [a_min, a_max, a_mean, a_med, a_sum, a_std, a_var, a_ptp]
        args = [indent(str(i), '... ') for i in v]
        print(frmt.format(*args))
    else:
        return a_min, a_max, a_mean, a_med, a_sum, a_std, a_var, a_ptp


# ----------------------------------------------------------------------
# ---- (5) querying, working with arrays ----
# ----------------------------------------------------------------------
# ---- _func, find, group_pnts, group_vals, reclass
#
def _func(fn, a, this):
    """Called by 'find' see details there
    (cumsum, eq, neq, ls, lseq, gt, gteq, btwn, btwni, byond)
    """
    #
    fn = fn.lower().strip()
    if fn in ['cumsum', 'csum', 'cu']:
        v = np.where(np.cumsum(a) <= this)[0]
    elif fn in ['eq', 'e', '==']:
        v = np.where(np.in1d(a, this))[0]
    elif fn in ['neq', 'ne', '!=']:
        v = np.where(~np.in1d(a, this))[0]  # (a, this, invert=True)
    elif fn in ['ls', 'les', '<']:
        v = np.where(a < this)[0]
    elif fn in ['lseq', 'lese', '<=']:
        v = np.where(a <= this)[0]
    elif fn in ['gt', 'grt', '>']:
        v = np.where(a > this)[0]
    elif fn in ['gteq', 'gte', '>=']:
        v = np.where(a >= this)[0]
    elif fn in ['btwn', 'btw', '>a<']:
        low, upp = this
        v = np.where((a >= low) & (a < upp))[0]
    elif fn in ['btwni', 'btwi', '=>a<=']:
        low, upp = this
        v = np.where((a >= low) & (a <= upp))[0]
    elif fn in ['byond', 'bey', '<a>']:
        low, upp = this
        v = np.where((a < low) | (a > upp))[0]
    return v


# @time_deco
def find(a, func, this=None, count=0, keep=None, prn=False, r_lim=2):
    """Find the conditions that are met in an array, defined by `func`.
    `this` is the condition being looked for.  The other parameters are defined
    in the Parameters section.

    >>> a = np.arange(10)
    >>> find(a, 'gt', this=5)
    array([6, 7, 8, 9])

    Parameters
    ----------
    `a` :
        Array or array like.
    `func` :
        `(cumsum, eq, neq, ls, lseq, gt, gteq, btwn, btwni, byond)`
        (        ==,  !=,  <,   <=,  >,   >=,  >a<, =>a<=,  <a> )
    `count` :
        only used for recursive functions
    `keep` :
        for future use
    `verbose` :
        True for test printing
    `max_depth` :
        prevent recursive functions running wild, it can be varied

    Recursive functions:
    -------------------
    cumsum :
        An example of using recursion to split a list/array of data
        parsing the results into groups that sum to this.  For example,
        split input into groups where the total population is less than
        a threshold (this).  The default is to use a sequential list,
        however, the inputs could be randomized prior to running.

    Returns
    -------
        A 1D or 2D array meeting the conditions

    """
    a = np.asarray(a)              # ---- ensure array format
    if keep is None:
        keep = []
    this = np.asarray(this)
    # masked = np.ma.is_masked(a)    # ---- check for masked array
    if prn:                        # ---- optional print
        print("({}) Input values....\n  {}".format(count, a))
    ix = _func(func, a, this)      # ---- sub function -----
    if ix is not None:
        keep.append(a[ix])         # ---- slice and save
        if len(ix) > 1:
            a = a[(len(ix)):]      # ---- use remainder
        else:
            a = a[(len(ix)+1):]
    if prn:                        # optional print
        print("  Remaining\n  {}".format(a))
    # ---- recursion functions check and calls ----
    if func in ['cumsum']:  # functions that support recursion
        if (len(a) > 0) and (count < r_lim):  # recursive call
            count += 1
            find(a, func, this, count, keep, prn, r_lim)
        elif count == r_lim:
            frmt = """Recursion check... count {} == {} recursion limit
                   Warning...increase recursion limit, reduce sample size\n
                   or changes conditions"""
            print(dedent(frmt).format(count, r_lim))
    # ---- end recursive functions ----
    # print("keep for {} : {}".format(func,keep))
    #
    if len(keep) == 1:   # for most functions, this is it
        final = keep[0]
    else:                # for recursive functions, there will be more
        temp = []
        incr = 0
        for i in keep:
            temp.append(np.vstack((i, np.array([incr]*len(i)))))
            incr += 1
        temp = (np.hstack(temp)).T
        dt = [('orig', '<i8'), ('class', '<i8')]
        final = np.zeros((temp.shape[0],), dtype=dt)
        final['orig'] = temp[:, 0]
        final['class'] = temp[:, 1]
        # ---- end recursive section
    return final


def find_closest(a, close_to=1):
    """Change values in an ndarray to match the closest in `close_to`.  This
    may be a scalar, or array-like for multiple cases.

    a : array
        an ndarray of integer or floating point values
    close_to : number or array-like
        If a number, it will return a close_to or 0.  If array-like, then the
        closest value in close_to will be returned

    >>> a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> find_closest(a, close_to=[3, 6, 9])
    array([3, 3, 3, 3, 3, 6, 6, 6, 9, 9])
    This behaviour differs from np.digitize

    >>> np.digitize(a, bins=[3, 6, 9])
    array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3], dtype=int64)

    >>> a = np.array([[3, 2, 4, 3, 3],
                      [1, 2, 3, 4, 1],
                      [2, 4, 4, 1, 1]])
    >>> find_closest(a, close_to=[0, 4])
    array([[4, 0, 4, 4, 4],
           [0, 0, 4, 4, 0],
           [0, 4, 4, 0, 0]])

    """
    shp = a.shape
    if np.isscalar(close_to):
        z = a.ravel()
        val = z[np.abs(z - close_to).argmin()]
        r = np.where(a == val, a, 0)
    else:
        close_to = np.atleast_1d(close_to)
        r = close_to[np.argmin(np.abs(a.ravel()[:, np.newaxis] - close_to), axis=1)]
        r = r.reshape(shp)
    return r


def group_pnts(a, key_fld='IDs', shp_flds=['Xs', 'Ys']):
    """Group points for a feature that has been exploded to points by
    `arcpy.da.FeatureClassToNumPyArray`.

    Parameters:
    ---------
    `a` : array
        A structured array, assuming ID, X, Y, {Z} and whatever else
        the array is assumed to be sorted... which will be the case
    `key_fld` : string
        Normally this is the `IDs` or similar
    `shp_flds` : strings
        The fields that are used to produce the geometry.

    Returns:
    -------
    See np.unique descriptions below

    References:
    ----------
    `<https://jakevdp.github.io/blog/2017/03/22/group-by-from-scratch/>`_.
    `<http://esantorella.com/2016/06/16/groupby/>`_.

    Notes:
    -----
    split-apply-combine .... that is the general rule

    """
    returned = np.unique(a[key_fld],           # the unique id field
                         return_index=True,    # first occurrence index
                         return_inverse=True,  # indices needed to remake array
                         return_counts=True)   # number in each group
    uni, idx, inv, cnt = returned
#    from_to = [[idx[i-1], idx[i]] for i in range(1, len(idx))]
    from_to = list(zip(idx, np.cumsum(cnt)))
    subs = [a[shp_flds][i:j] for i, j in from_to]
    groups = [sub.view(dtype='float').reshape(sub.shape[0], -1)
              for sub in subs]
    return groups


# ---- (6) analysis .... code section ----
# ---- uniq, is_in
#
def uniq(ar, return_index=False, return_inverse=False,
         return_counts=False, axis=None):
    """Taken from, but modified for simple axis 0 and 1 and structured
    arrays in (N, m) or (N,) format.

    To enable determination of unique values in uniform arrays with
    uniform dtypes.  np.unique in versions < 1.13 need to use this.

    https://github.com/numpy/numpy/blob/master/numpy/lib/arraysetops.py
    """
    ar = np.asanyarray(ar)
    if np.version.version > '1.13':
        return np.unique(ar, return_index, return_inverse,
                         return_counts, axis=axis)


def is_in(arr, look_for, keep_shape=True, binary=True, not_in=False):
    """Similar to `np.isin` for numpy versions < 1.13, but with additions to
    return the original shaped array with an `int` dtype

    Parameters:
    ----------
    arr : array
        the array to check for the elements
    look_for : number, list or array
        what to use for the check
    keep_shape : boolean
        True, returns the array's original shape.  False, summarizes all axes
    not_in : boolean
        Switch the query look_for True

    Note:
    ----
    >>> from numpy.lib import NumpyVersion
    >>> if NumpyVersion(np.__version__) < '1.13.0'):
        # can add for older versions later
    """
    arr = np.asarray(arr)
    shp = arr.shape
    look_for = np.asarray(look_for)
    uni = False
    inv = False
    if not_in:
        inv = True
    r = np.in1d(arr, look_for, assume_unique=uni, invert=inv)
    if keep_shape:
        r = r.reshape(shp)
    if binary:
        r = r.astype('int')
    return r


def running_count(a, to_label=False):
    """Perform a running count on a 1D array identifying the order number
    of the value in the sequence.

    Parameters
    ----------
    `a` : array
        1D array of values, int, float or string
    `to_label` : boolean
        Return the output as a concatenated string of value-sequence numbers if
        True, or if False, return a structured array with a specified dtype.

    Examples:
    ---------
    >>> a = np.random.randint(1, 10, 10)
    >>> #  [3, 5, 7, 5, 9, 2, 2, 2, 6, 4] #
    >>> running_count(a, False)
    array([(3, 1), (5, 1), (7, 1), (5, 2), (9, 1), (2, 1), (2, 2),
           (2, 3), (6, 1), (4, 1)],
          dtype=[('Value', '<i4'), ('Count', '<i4')])
    >>> running_count(a, True)
    array(['3_001', '5_001', '7_001', '5_002', '9_001', '2_001', '2_002',
           '2_003', '6_001', '4_001'],
          dtype='<U5')

    >>> b = np.array(list("zabcaabbdedbz"))
    >>> #  ['z', 'a', 'b', 'c', 'a', 'a', 'b', 'b', 'd', 'e', 'd','b', 'z'] #
    >>> running_count(b, False)
    array([('z', 1), ('a', 1), ('b', 1), ('c', 1), ('a', 2), ('a', 3),
           ('b', 2), ('b', 3), ('d', 1), ('e', 1), ('d', 2), ('b', 4),
           ('z', 2)], dtype=[('Value', '<U1'), ('Count', '<i4')])
    >>> running_count(b, True)
    array(['z_001', 'a_001', 'b_001', 'c_001', 'a_002', 'a_003', 'b_002',
           'b_003', 'd_001', 'e_001', 'd_002', 'b_004', 'z_002'], dtype='<U5')

    """
    dt = [('Value', a.dtype.str), ('Count', '<i4')]
    z = np.zeros((a.shape[0],), dtype=dt)
    idx = a.argsort(kind='mergesort')
    s_a = a[idx]
    neq = np.where(s_a[1:] != s_a[:-1])[0] + 1
    run = np.ones(a.shape, int)
    run[neq[0]] -= neq[0]
    run[neq[1:]] -= np.diff(neq)
    out = np.empty_like(run)
    out[idx] = run.cumsum()
    z['Value'] = a
    z['Count'] = out
    if to_label:
        z = np.array(["{}_{:0>3}".format(*i) for i in list(zip(a, out))])
    return z


def sequences(data, stepsize=0):
    """Return an array of sequence information denoted by stepsize

    data :
        List/array of values in 1D
    stepsize :
        Separation between the values.  If stepsize=0, sequences of equal
        values will be searched.  If stepsize is 1, then sequences incrementing
        by 1... etcetera.  Stepsize can be both positive or negative

    >>> # check for incrementing sequence by 1's
    >>> d = [1, 2, 3, 4, 4, 5]
    >>> s = sequences(d, 1)
    array([(0, 1, 4, 0, 4), (1, 4, 2, 4, 6)],
          dtype=[('ID', '<i4'), ('Value', '<i4'), ('Count', '<i4'),
                 ('From_', '<i4'), ('To_', '<i4')])
    >>> prn_rec(s)  # prn_rec in frmts.py
     id  ID   Value   Count   From_   To_
    ---------------------------------------
     000    0       1       4       0     4
     001    1       4       2       4     6

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
    https://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-
    sequences-elements-from-an-array-in-numpy
    """
    #
    a = np.array(data)
    a_dt = a.dtype.kind
    dt = [('ID', '<i4'), ('Value', a.dtype.str), ('Count', '<i4'),
          ('From_', '<i4'), ('To_', '<i4')]
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
    out = np.array(list(zip(seq_num, vals, cnts, frum, too)), dtype=dt)
    return out


# ---- (7) sorting,  column and row sorting .... code section ---------------
# ---- sort_rows_by_col, sort_cols_by_row, radial_sort ----
def sort_rows_by_col(a, col=0, descending=False):
    """Sort a 2D array by column.

    >>> sort_rows_by_col(a, 0, True)
    >>> a =array([[0, 1, 2],    array([[6, 7, 8],
                  [3, 4, 5],           [3, 4, 5],
                  [6, 7, 8]])          [0, 1, 2]])
    """
    a = np.asarray(a)
    shp = a.shape[0]
    if not (0 <= abs(col) <= shp):
        raise ValueError("column ({}) in range (0 to {})".format(col, shp))
    a_s = a[a[:, col].argsort()]
    if descending:
        a_s = a_s[::-1]
    return a_s


def sort_cols_by_row(a, col=0, descending=False):
    """Sort the rows of an array in the order of their column values
    :  Uses lexsort """
    ret = a[np.lexsort(np.transpose(a)[::-1])]
    if descending:
        ret = np.flipud(ret)
    return ret


def radial_sort(pnts, cent=None):
    """Sort about the point cloud center or from a given point

    Requires:
    ---------
    pnts : array-like, 2D
        an array of points (x,y) as array or list
    cent : floats
        list, tuple, array of the center's x,y coordinates
        cent = [0, 0] or np.array([0, 0])

    Returns:
    --------
    The angles in the range -180, 180 x-axis oriented, and the sort order.
    """
    pnts = np.asarray(pnts, dtype='float64')
    if cent is None:
        cent = pnts.mean(axis=0)
    ba = pnts - cent
    ang_ab = np.arctan2(ba[:, 1], ba[:, 0])
    ang_ab = np.degrees(ang_ab)
    sort_order = np.argsort(ang_ab)
    return ang_ab, sort_order


def view_sort(a):
    """Sort 2D arrays assumed to be coordinates and other baggage, in the order
    that they appear in the row.  It is best used for sorting x,y coorinate,
    using argsort.

    Returns:
    --------
    The sorted array and the indices of their original positions in the
    input array.
    """
    a_view = a.view(a.dtype.descr * a.shape[1])
    idx = np.argsort(a_view, axis=0, order=(a_view.dtype.names)).ravel()
    a = np.ascontiguousarray(a[idx])
    return a, idx

def xy_sort(a):
    """Formally called `view_sort`.  See the documentation there
    """
    return view_sort(a)


# ---- extras *****

def pack_last_axis(arr, names=None):
    """used in nd2struct
    Then you could do:
    >>> pack_last_axis(uv).tolist()
    to get a list of tuples.
    """
    if arr.dtype.names:
        return arr
    names = names or ['f{}'.format(i) for i in range(arr.shape[-1])]
    return arr.view([(n, arr.dtype) for n in names]).squeeze(-1)


# ----------------------------------------------------------------------
# ----  _help .... code section
def _tools_help_():
    """arraytools.tools help...

    Function list follows:
    """
    _hf = """
    :-------------------------------------------------------------------:
    : ---- arrtools functions  (loaded as 'art') ----
    : ---- from tools.py
    (1)  arr2xyz(a, verbose=False)
         array (col, rows) to (x, y) and array values for z.
    (2)  make_blocks(rows=3, cols=3, r=2, c=2, dt='int')
         make arrays consisting of blocks
    (3)  group_vals(seq, delta=1, oper='!=')
    (4)  reclass(a, bins=[], new_bins=[], mask=False, mask_val=None)
         reclass an array
    (5)  scale(a, x=2, y=2, num_z=None)
         scale an array up in size by repeating values
    (6)  split_array(a, fld='ID')
         split an array using an index field
    (7)  make_flds(n=1, as_type='float', names=None, def_name='col')
         make structured/recarray fields
    (8) nd_rec
    (9) nd_struct
    (10) nd2struct(a)
         convert an ndarray to a structured array with fields
    (11) nd2rec
    (12) rc_vals
    (13) xy_vals
    (14) array_cols
    (15) change_arr(a, order=[], prn=False)
         reorder and/or drop columns
    (16) concat_arrs
    (17) pad__
    (18) stride(a, r_c=(3, 3))
         stride an array for moving window functions
    (19) block
    (20) sliding_window_view
    (21) block_arr(a, win=[3, 3], nodata=-1)
         break an array up into blocks
    (22) rolling_stats((a0, no_null=True, prn=True))
    (23) _func, find(a, func, this=None, count=0, keep=[], prn=False, r_lim=2)
         find elements in an array using...
         func - (cumsum, eq, neq, ls, lseq, gt, gteq, btwn, btwni, byond)
               (      , ==,  !=,  <,   <=,  >,   >=,  >a<, =>a<=,  <a> )
    (24)  group_pnts(a, key_fld='ID', keep_flds=['X', 'Y', 'Z'])
    (25) uniq(ar, return_index=False, return_inverse=False,
              return_counts=False, axis=0)
    (26) is_in
    (27) running_count
    (28) sequences(data, stepsize)
    (29) sort_rows_by_col
    (30) sort_cols_by_row
    (31) radial_sort
    (32) pack_last_axis
    ---  _tools_help_  this function
    :-------------------------------------------------------------------:
    """
    print(dedent(_hf))


# ----------------------------------------------------------------------
# ---- _demo .... code section
# @run_deco
def _demo_tools():
    """
    : - Run examples of the existing functions.
    """
    a = np.arange(3*4).reshape(3, 4).copy()
    b = nd2struct(a)
    c = np.arange(2*3*4).reshape(2, 3, 4)
    d = np.arange(9*6).reshape(9, 6)
    bloc = block_arr(a, win=[2, 2], nodata=-1)  # for block
    chng = change_arr(b, order=['B', 'C', 'A'], prn=False)
    scal = scale(a, 2)
    m_blk = make_blocks(rows=3, cols=3, r=2, c=2, dt='int')
    m_fld = str(make_flds(n=3, as_type='int', names=["A", "B", "C"]))
    spl = split_array(b, fld='A')
    stri = stride(a, (3, 3))
    rsta = rolling_stats(d, no_null=True, prn=False)
#    arr = np.load(data_path + '/sample_20.npy')
#    row = arr['County']
#    col = arr['Town']
#    ctab, a0, result, r0, c0 = crosstab(row, col)
#    arr = arr.reshape(arr.shape[0], -1)
    frmt = """
: ----- _demo {}
:
:Input ndarray, 'a' ...
{!r:}\n
:Input ndarray, 'b' ...
{!r:}\n
:Input ndarray, 'c' ...
{!r:}\n
:Input ndarray, 'd' ...
{!r:}\n
:---- Functions by number  ---------------------------------------------
:(1)  arr2xyz(a, verbose=False)
{}\n
:(2)  block_arr(a, win=[2, 2], nodata=-1)
{}\n
:(3) change_arr(b, order=['B', 'C', 'A'], prn=False
:    Array 'b', reordered with 2 fields dropped...
{!r:}\n
:(5) scale() ... scale an array up by an integer factor...
{}\n
:(7) make_flds() ... create default field names ...
{}\n
:(8) split_array() ... split an array according to an index field
{}\n
:(9) stride() ... stride an array ....
{}\n
:(10) make_blocks(rows=3, cols=3, r=2, c=2, dt='int')
{}\n
:(11) nd_struct() ... make a structured array from another array ...
{!r:}\n
:(12) rolling_stats()... stats for a strided array ...
:    min, max, mean, sum, std, var, ptp
{}\n
"""
    args = ["-"*62, a, b, c, d,
            arr2xyz(a), bloc, chng.reshape(a.shape[0], -1), scal,  # 1 -5
            m_fld, spl, stri, m_blk, nd2struct(a), rsta]  # 6- 12
    print(frmt.format(*args))
    # del args, d, e


def pyramid(core=9, steps=10, incr=(1, 1), posi=True):
    """Create a pyramid see pyramid_demo.py"""
    a = np.array([core])
    a = np.atleast_2d(a)
    for i in range(1, steps):
        val = core - i
        if posi and (val <= 0):
            val = 0
        a = np.lib.pad(a, incr, "constant", constant_values=(val, val))
    return a


# ----------------------------------------------------------------------
# ---- __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))  # script =  sys.argv[0]
#    data_path = script.replace('tools.py', 'Data')
#    def f():
#        pass
#    print(f.__code__.co_filename)
#    _demo_tools()
