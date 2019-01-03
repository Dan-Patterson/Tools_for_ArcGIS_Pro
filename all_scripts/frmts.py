# -*- coding: UTF-8 -*-
"""
frmts
=====

Script:   frmts.py

Author:   Dan_Patterson@carleton.ca

Modified: 2018-11-02

References:
----------
`np.set_printoptions` and `np.printoptions`

`<https://github.com/numpy/numpy/blob/master/numpy/core/arrayprint.py>`_.

>>> set_printoptions(precision=3, threshold=100, edgeitems=3, linewidth=80,
                     suppress=True, nanstr='nan', infstr='inf',
                     formatter=None, sign=None, floatmode=None, **kwarg)

>>> with np.printoptions(precision=deci, linewidth=ln_wdth):
        print(a)  # the original options will be reset after printing

Purpose:
--------

The prn2d function is used to provide a side-by-side view of 2, 3, and 4D
arrays.  Specifically, 3D and 4D arrays are useful and for testing
purposes, seeing the dimensions in a different view can facilitate
understanding.  For the best effect, the array shapes should be carefully
considered. Some guidelines follow.  The middle 'r' part of the shape is
not as affected as the combination of the 'd' and 'c' parts.  The array is
trimmed beyond the 'width' parameter in prn2d.

Sample the 3D array shape so that the format (d, r, c)
is within the 20-21 range for d*c ... for example::
        integers          floats
        2, r, 10  = 20    2, r, 8 = 16
        3, r,  7  = 21    3, 4, 5 = 15
        4, r,  5  = 20    4, r, 4 = 16
        5, r,  4  = 20    5, r, 3 = 15

>>> prn2d(a)  example for a =  np.arange(3*4*5).reshape(3, 4, 5)
---------------------------------------------------
Array...
-shape (3, 4, 5), ndim 3
  .  0  1  2  3  4    20 21 22 23 24    40 41 42 43 44
  .  5  6  7  8  9    25 26 27 28 29    45 46 47 48 49
  . 10 11 12 13 14    30 31 32 33 34    50 51 52 53 54
  . 15 16 17 18 19    35 36 37 38 39    55 56 57 58 59
  .   sub (0 )        : sub (1 )        : sub (2 )

The middle part of the shape should also be reasonable should you want
to print the results:

How it works

>>> a[...,0,:].flatten()
array([ 0,  1,  2,  3,  4, 20, 21, 22, 23, 24, 40, 41, 42, 43, 44])

>>> a[...,0,(0, 1, -2, -1)].flatten()
array([ 0,  1,  3,  3, 20, 21, 23, 23, 40, 41, 43, 43])


Functions:
=========
help(<function name>) for help

::

    public  -  private...
    deline  -  _pre
    prn2d   - _check, _concat, _row_format
    prn_ma - _fix
    in_by   - _pre_num

 ... see __all__ for a complete listing

1(a) col_hdr() :

produce column headers to align output for formatting purposes

``.........1.........2.........3.........4.........5.........6.........
123456789012345678901234567890123456789012345678901234567890123456789``

----------------------------------------------------------------------


1(b)  deline(a)::

     shp = (2,3,4)
     a = np.arange(np.prod(shp)).reshape(shp)
     deline(a)

     Main array...
     ndim: 3 size: 24
     shape: (2, 3, 4)
     [[[ 0  1  2  3]
       [ 4  5  6  7]
       [ 8  9 10 11]]
     a[1]....
      [[12 13 14 15]
       [16 17 18 19]
       [20 21 22 23]]]

(1c) in_by

indent objects, added automatic support for arrays and optional line numbers
::
     a = np.arange(2*3*4).reshape(2,3,4)
     print(art.in_by(a, hdr='---- header ----', nums=True, prefix =".."))
     ---- header ----
     00..[[[ 0  1  2  3]
     01..  [ 4  5  6  7]
     02..  [ 8  9 10 11]]
     03..
     04.. [[12 13 14 15]
     05..  [16 17 18 19]
     06..  [20 21 22 23]]]

(1d)  redent(lines, spaces=4)
::
     a = np.arange(3*5).reshape(3,5)
     >>> print(redent(a))
     |    [[ 0  1  2  3  4]
     |     [ 5  6  7  8  9]
     |     [10 11 12 13 14]]

(2) prn2d(a)
::
   a = np.arange(2*3*3).reshape(2,3,3)
   array([[[ 0,  1,  2],
           [ 3,  4,  5],
           [ 6,  7,  8]],

          [[ 9, 10, 11],
           [12, 13, 14],
           [15, 16, 17]]])
   prn2d(a)
   Array... shape (2, 3, 3), ndim 3, not masked
    0,  1,  2     9, 10, 11
    3,  4,  5    12, 13, 14
    6,  7,  8    15, 16, 17
   sub (0)       sub (1)

(3) prn_ma
::
    :--------------------
    :Masked array........
    :  ndim: 2 size: 20
    :  shape: (5, 4)
    :
    :... a[:5, :4] ...
      -  1  2  3
      4  5  6  7
      8  -  -  -
     12 13 14 15
     16 17 18  -

(4) pd and quick_prn
    see code

(5) prn_struct and prn_rec : main functions
        _col_kind_width, _col_format,subsample

prn_struct(b, edges=3, max_lines=10, width=100, deci=2)
::
    OBJECTID   f0   County  Town  Facility  Time
    ----------------------------------------------
             1    0 B       A_    Hall          26
             2    1 C       C_    Hall          60
             3    2 D       A_    Hall          42
           ...  ...     ...   ...       ...
            18   17 A       C_    Hall          59
            19   18 C       C_    Hosp          37
            20   19 B       B_    Hall          52

    Array... shape: (20,)

prn_rec(a, edges=5, max_rows=25, deci=2)
::
    Format ... C:/Git_Dan/arraytools/Data/sample_20.npy
    record/structured array, with and without field names.
    --n-- OBJECTID   f0  County  Town  Facility  Time``
    -------------------------------------------------
    000         1    0       B    A_      Hall    26
    001         2    1       C    C_      Hall    60
    002         3    2       D    A_      Hall    42


(6) make_row_format
::
    make_row_format(dim=2, cols=3, a_kind='f', deci=1,
                    a_max=10, a_min=-10, prn=False)
    '{:6.1f}{:6.1f}{:6.1f}  {:6.1f}{:6.1f}{:6.1f}'

prn_
::
  prn_(a, deci=2, width=100, title="Array", prefix=". . ", prnt=True)

  Array... ndim: 3  shape: (2, 3, 3)
  . .   0  1  2    9 10 11
  . .   3  4  5   12 13 14
  . .   6  7  8   15 16 17

(7)  prn_3d4d(a, deci=2, edgeitems=3, width=100, prnt=True)
::
    prn_3d4d(z)
    Array... ndim 4  shape(1, 2, 3, 4)
    |  0  1  2  3   12 13 14 15 |
    |  4  5  6  7   16 17 18 19 |
    |  8  9 10 11   20 21 22 23 |
    |=> (0 2 3 4)

Notes:
=====

**column numbering**

>>> d = (('{:<10}')*7).format(*'0123456789'), '0123456789'*7, '-'*70
>>> s = '\n{}\n{}\n{}'.format(args[0][1:], args[1][1:], args[2]) #*args)
>>> print(s)
             1         2         3         4         5         6
    123456789012345678901234567890123456789012345678901234567890123456789


**Getting default print options, then setting them back **

>>> pr_opt = np.get_printoptions()
>>> df_opt = ", ".join(["{}={}".format(i, pr_opt[i]) for i in pr_opt])


** Rearranging blocks into columns using np.c_[...] **

>>>  a = np.arange(3*2*3).reshape(3, 2, 3)
>>>  a_max = a.max()
>>>  a_min = a.min()
>>>  aa = np.c_[(a[0], a[1], a[2])]
>>>  d, r, c = a.shape
>>>  deci = 1
>>>  a_kind = a.dtype.kind
>>>  f = _format_row_test(d, r, c, a_kind, deci, a_max, a_min)
::
Row format given
d 3, r 2, c 3
kind i decimals 1
{:3.0f}{:3.0f}{:3.0f}  {:3.0f}{:3.0f}{:3.0f}  {:3.0f}{:3.0f}{:3.0f}
0123456789012345678901234567890123456789012345678901234567890123456789
0         1         2         3         4         5         6

>>>  r = `\\n`.join([f.format(*i) for i in aa])
>>>  print(r)
  0  1  2    6  7  8   12 13 14
  3  4  5    9 10 11   15 16 17

>>> # Now change dtype and decimals
>>>  a_kind = 'f'
>>>  deci = 2
>>>  f = _format_row_test(d, r, c, a_kind, deci, a_max, a_min)
 .... snip ....
>>>  print(r)
  0.00  1.00  2.00    6.00  7.00  8.00   12.00 13.00 14.00
  3.00  4.00  5.00    9.00 10.00 11.00   15.00 16.00 17.00


**all at once**

>>> a
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],
       [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]]])

>>>  s0, s1, s2 = a.shape
>>>  b = a.swapaxes(2, 1).reshape(s0*s2, s1).T
>>>  b
  array([[ 0,  1,  2,  3, 12, 13, 14, 15],
         [ 4,  5,  6,  7, 16, 17, 18, 19],
         [ 8,  9, 10, 11, 20, 21, 22, 23]])


Masked array info:
------------------

>>>  a.get_fill_value() # see default_filler dictionary
>>>  a.set_fill_value(np.NaN)
>>>  np.ma.maximum_fill_value(a)   -inf
>>>  np.ma.minimum_fill_value(a)    inf
>>>  default_filler =
     {'b': True, 'c': 1.e20 + 0.0j, 'f': 1.e20, 'i': 999999,'O': '?',
      'S': b'N/A', 'u': 999999,'V': '???','U': sixu('N/A')}


Others:
------

>>> b.transpose(1, 2, 0)[:,:,::-1]
>>> # ** tip *** reorder from after transpose or even a swapaxes
>>> # the ::-1 does the reversing... same as [...,::-1]


----------
"""

# pylint: disable=C0103
# pylint: disable=R1710
# pylint: disable=R0914

# ---- imports, formats, constants ----

import sys
from textwrap import dedent, indent
import numpy as np

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{!r: 0.3f}'.format}
edge = 3
ln_wdth = 100
np.set_printoptions(edgeitems=edge, linewidth=ln_wdth, precision=3,
                    suppress=True, nanstr='-n-', threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

pr_opt = np.get_printoptions()
df_opt = ", ".join(["{}={}".format(i, pr_opt[i]) for i in pr_opt])

script = sys.argv[0]

__all__ = ['col_hdr',            # column headers
           'xy_dist_headers',    # table dtype format
           'deline',             # remove excessive blank lines
           'in_by',              # an indent variant with options
           'redent',             # indent
           '_chunks',            # take chunks of stuff
           'head_tail',          # return the head/tail of a 1d array
           '_check',             # helper functions
           '_slice_rows',        #
           '_slice_cols',        #
           '_slice_head_tail',   #
           '_col_format',        # printing section
           'prn_nd',             # for ndarray
           'prn_ma',             # for masked arrays
           'prn_rec', 'pd_',     # record/structured arrays
           'prn_struct',         #
           'make_row_format',    # a big helper function
           'prn_',               # ndarray variant
           'prn'        # ---- this def is used to call all the others ----
           ]


# ----------------------------------------------------------------------
# (1) Short, or reused code section
#
# (1a) col_hdr ... code section .....
def col_hdr(num=8):
    """Print numbers from 1 to 10*num to show column positions"""
    args = [(('{:<10}')*num).format(*'0123456789'),
            '0123456789'*num, '-'*10*num]
    s = "\n{}\n{}\n{}".format(args[0][1:], args[1][1:], args[2])  # *args)
    print(s)


def xy_dist_headers(N):
    """Construct headers for the optional table output"""
    vals = np.repeat(np.arange(N), 2)
    names = ['X_{}', 'Y_{}']*N + ['d_{}']*(N-1)
    vals = (np.repeat(np.arange(N), 2)).tolist() + [i for i in range(1, N)]
    n = [names[i].format(vals[i]) for i in range(len(vals))]
    f = ['<f8']*N*2 + ['<f8']*(N-1)
    return list(zip(n, f))
# ----------------------------------------------------------------------
# (1b) deline ... code section .....
def deline(a, width=100, header="Array...", prefix="  ."):
    """Remove extraneous lines from array output.
    More useful for long arrays with ndim >= 3

    Requires:
    --------
    `a` : anything
        anything that can be put into array form
    `header` :
        an optional header
    `prefix` : text
        could be just spaces or something like shown
    """
    def _pre(obj):
        for line in obj.splitlines(False):
            frmt = "{}{}".format(prefix, line)
            yield frmt
    # ----
    if not isinstance(a, (list, tuple, np.ndarray)):
        return a
    a = np.asanyarray(a)
    if a.dtype.kind not in ('i', 'u', 'f', 'c'):
        return a
    header += " shape: {} ndim: {}".format(a.shape, a.ndim)
    f1 = (":arr[{}" + ", :{}"*len(a.shape[1:]) + "]")
    out = [header]
    c = 0
    for i in a:
        a_s = f1.format(c, *i.shape)  # ---- uses f1 format above
        out.append(a_s)
        out.extend(_pre(str(i)))
        c += 1
    f = "\n".join([i for i in out if i != prefix])
    with np.printoptions(edgeitems=edge, linewidth=width):
        print(f)
    # ----


# ---------------------------------------------------------------------------
# (1c) in_by .... code section
def in_by(obj, hdr="", line_nums=False, prefix="   ."):
    """A `textwrap.indent` variant for python 2.7 or a substitute for
    any version of python.  The function stands for `indent by`.

    Requires:
    --------
    `obj` : object that can be cast as string
        obj to indent, List, tuple, ndarray converted to strings
        first. You can use repr representation before using if needed.
    `hdr` : text
        optional header
    `line_nums` : boolean
        True to add line numbers
    `prefix` : test
        Text to use for indent ie '  ' for 2 spaces or '....'

    Reference:
    ---------
    [1] https://docs.python.org/3.7/library/textwrap.html for python > 3.3

    Notes:
    -----
        Header and line numbers options added.
    """
    def _pre_num():
        c = 0
        for line in obj.splitlines(True):
            if line_nums:
                frmt = "{:>02}{}{}".format(c, prefix, line)
            else:
                frmt = "{}{}".format(prefix, line)
            yield frmt
            c += 1
    #
    if hdr != "":
        hdr = "\n{}\n".format(hdr)
    if isinstance(obj, (list, tuple, np.ndarray)):
        obj = str(obj)
    out = hdr + "".join(_pre_num())
    print(out)
    # ----


# ----------------------------------------------------------------------
# (1d) redent .... code section
def redent(lines, spaces=4):
    """Strip and reindent by num_spaces, a sequence of lines
    `lines` : text
        Text or what can be made text
        Use str() or repr() on the inputs if you want control on form

    See also:
    --------
        See `in_by` for more options
    """
    lines = str(lines).splitlines()
    sp = [len(ln) - len(ln.lstrip()) for ln in lines]
    spn = " "*spaces
    out = list(zip(lines, sp))
    ret = "\n".join(["{0}{1!s:>{2}}".format(spn, *ln) for ln in out])
    return ret


# ----------------------------------------------------------------------
# (1e) _chunks .... code section
def _chunks(s, n):
    """Produce n-sized chunks from s."""
    for start in range(0, len(s), n):
        yield s[start:start+n]

# ----------------------------------------------------------------------
# (1f) head_tail  .... code section
#
def head_tail(size=10, head=3, tail=None, fill=None):
    """Slice `head` and `tail` elements of a 1D array of a given `size`.
    Optionally, insert a middle `fill` element.

    >>> head_tail(size=10, head=3, tail=None, fill=None)
    [0, 1, 2, 7, 8, 9]
    >>> head_tail(size=10, head=3, tail=None, fill="...")
    [0, 1, 2, '...', 7, 8, 9]
    """
    if head is None:
        head = 0
    if tail is None:
        tail = head
    head, tail = [int(abs(i)) for i in [head, tail]]
    r = np.arange(size).tolist()
    if fill is None:
        return r[:head] + r[-tail:]
    return r[:head] + [fill] + r[-tail:]


# ----------------------------------------------------------------------
# ---- Format section ---- functions and constants must be used ----

# (2) ---- constants and common functions----

floats = np.typecodes['AllFloat']
ints = np.typecodes['AllInteger']
nums = floats + ints


def _check(a):
    """Check dtype and max value for formatting information"""
    return a.shape, a.ndim, a.dtype.kind, np.nanmax(a), np.nanmin(a)

def _slice_rows(a, edge_rows=3):
    """Split an array keeping `edge_rows` from the start and end of the array.
    Used by `prn_struct` and `prn_rec`.
    """
    if a.shape[0] <= (edge_rows * 2):
        return a
    return np.hstack((a[:edge_rows], a[-edge_rows:]))  # top, bott


def _slice_cols(a, edge_cols=3):
    """Split a structured array keeping `edge_cols from the start of an array.
    Used by `prn_struct` and `prn_rec`.
    """
    names = a.dtype.names
    return a[list(names[:edge_cols])]


def _slice_head_tail(a, edge_cols=3):
    """Split a structuredarray keeping `edge_cols` from the start and end.
    Used by `prn_struct` and `prn_rec`.
    """
    names = list(a.dtype.names)
    shp = a.shape[0]
    left = a[names[:edge_cols]]
    right = a[names[-edge_cols:]]
    dt_new = list(left.dtype.descr) + [('...', '3U')] + list(right.dtype.descr)
    z = np.zeros((shp,), dtype=dt_new)
    for i in left.dtype.names:
        z[i] = left[i]
    z['...'] = ['...'] * shp
    for i in right.dtype.names:
        z[i] = right[i]
    del left, right, a
    return z


def _row_format(a, sep='', deci=0):
    """Create format string from array parameters.

    Parameters:
    -----------
    array parameters: _check function
        a.shape, a.ndim, a.dtype.kind, a.max(), a.min()
    sep : string
        A separator for parts of the array
    deci : integer
        Number of decimal places to use for all floats

    Requires:
    ---------
    constants : numbers
        float and int as defined above
    _check : function
        Returns base array information
    """
    shp, ndim, kind, a_max, a_min = _check(a)  # get base array information
    cols = shp[-1]
    if kind in floats:
        w_, m_ = [':{}.{}f', '{:0.{}f}']
    elif kind in ints:
        w_, m_ = [':{}.0f', '{:0.0f}']
    else:
        w_, m_ = ['!s:>{}', '{}']
        deci = 0
    m = max(len(m_.format(a_max, deci)), len(m_.format(a_min, deci))) + 1
    w_fmt = w_.format(m, deci)
    r_fmt = sep.join(['{' + w_fmt + '}' for i in range(cols)])
    return r_fmt


def _col_kind_width(a, deci=0):
    """Column properties for ndarray and recarray/structured array types.

    Notes:
        Used by _col_format, prn_rec and prn_struct.  It check the
    length of the values in the field, rounded to `deci`mal places if needed,
    That value to the field name length, if found, and returns the `max`.

    sample output for a structured array :
        [('i', 5), ('U', 25), ('i', 5), ('i', 5)]

    Requires:
    ---------
    constants : numbers
        float and int as defined above
    """
    def _ckw_(a, name, deci):
        """process for arrays arrays with named fields"""
        c_kind = a.dtype.kind
        if (c_kind in floats) and (deci != 0):  # float with decimals
            c_max, c_min = np.round([np.nanmin(a), np.nanmax(a)], deci)
            c_width = len(max(str(c_min), str(c_max), key=len))
        elif c_kind in nums:      # int, unsigned int, float wih no decimals
            c_width = len(max(str(np.nanmin(a)), str(np.nanmax(a)), key=len))
        elif c_kind in ('U', 'S', 's'):
            c_width = len(max(a, key=len))
        else:
            c_width = len(str(a))
        c_width = max(len(name), c_width)
        return [c_kind, c_width]
    # ---- constants
    # ---- call to _ckw_ ----
    dtn = a.dtype.names
    if dtn is None:  # ---- uniform dtype
        return [_ckw_(a, name="", deci=deci)]
    return [_ckw_(a[name], name, deci=deci) for name in dtn]


def _col_format(c, c_name="c00", deci=0):
    """Determine column format for an ndarray or structured array.  The
    number of decimal places for float fields can be specified.
    Used by prn_rec.

    `c` : column
        A column in an array.
    `c_name` : text
        column name for ndarrays of uniform dtype.  Ignored otherwise
    `deci` : int
        Desired number of decimal points if the data are numeric

    Requires:
    ---------
    _col_kind_width : function
        This function does the determination of column kind and width

    Notes:
    -----
    To do all field `names`

    >>> [_col_format(j) for j in [a[i] for i in names]]
    [(':> 6.0f', 5), ('!s:<26', 25), (':> 6.0f', 5),
     (':> 6.0f', 5), (':> 4.0f', 2)]
    """
    pairs = _col_kind_width(c, deci=deci)
    form_width = []
    for c_kind, c_width in pairs:
        if c_kind in ints:  # ---- integer type
            w_ = ':> {}.0f'
            c_width = max(len(c_name), c_width) + deci
            c_format = w_.format(c_width, 0)
        elif (c_kind in floats) and np.isscalar(c[0]):  # ---- float rounded
            w_ = ':> {}.{}f'
            c_width = max(len(c_name), c_width) + deci
            c_format = w_.format(c_width, deci)
        else:
            c_format = "!s:<{}".format(c_width)

        form_width.append([c_format, c_width])
    return form_width


# ----------------------------------------------------------------------
# (3) prn_nd .... code section
def prn_nd(a, deci=2, width=100, title="Array", prefix="  .", prnt=True):
    """Format number arrays by row, and print

    Parameters:
    -----------
    `a` : array
        An array of int or float dtypes, 1, 2, 3 and 4D arrays tested.
    `deci` - int
        Decimal places for floating point numbers
    `width` : int
        Default width for onscreen and printing, output beyond this
        length will be truncated with a warning.  Reshape to overcome.
    `title` : string
        The default title, change to provide more information.

    Returns:
    --------
    Prints the array with the 1st dimension flattened-like by row

    Notes:
    -----
    - `w_frmt` :  width formatter
    - `m_frmt` :  max number formatter to get max. number of characters
    """

    def _concat(rows, r_fmt, width, prefix):
        """print the subset to maximimum width"""
        end = ["", "...."][len(r_fmt.format(*rows[0])) > width]
        txt = prefix
        rw = [r_fmt.format(*v)[:width] + end for v in rows]
        txt += ("\n" + prefix).join(rw)  # + "\n"
        return txt

    def d4_frmt(a_shp, a, txt, a_dim):
        """Dealing with 4, 5 ?D arrays"""
        d4, d, r, c = a_shp
        hdr = "\n" + "-"*25
        fm = hdr + "\n-({}, + ({}, {}, {})"
        if a_dim == 5:
            fm = "\n--(.., {}, + ({}, {}, {})"
        t = ""
        for d3 in range(d4):
            t += fm.format(d3, d, r, c) + "\n"
            a_s = a[d3]
            rows = [a_s[..., i, :].flatten() for i in range(r)]
            t += _concat(rows, row_frmt, width, prefix)
        return t
    #
    # ---- begin constructing the array format ----
    txt = ""
    a = np.asanyarray(a)
    # ---- run _check ----
    if a.ndim < 3:
        if a.ndim == 2:
            a = a.reshape((1,) + a.shape)
        else:
            return "Array is not >= 2D"
    #
    a_shp, a_dim, a_kind, a_min, a_max = _check(a)  # get base array info
    #
    fv = ""
    if np.ma.isMaskedArray(a):  # ----
        a = np.ma.round(a, decimals=deci)
        if a.dtype.kind in floats:
            default_fill = np.ma.default_fill_value(a)
            a.set_fill_value(default_fill)
        else:
            a.set_fill_value(np.iinfo(a.dtype).max)
        fv = ", masked array, fill value {}".format(a.get_fill_value())
        #a = a.data
    # ---- correct dtype, get formats ----
    if (a_kind in nums) and (a_dim >= 3):
        args = title, a_shp, a_dim, fv
        txt = "{}...\n-shape {}, ndim {}{}".format(*args)
        d, r, c = a_shp[-3:]
        row_frmt = _row_format(a, sep='', deci=deci)
        row_frmt = (row_frmt + "  ") * d
        if a_dim == 3:
            rows = [a[..., i, :].flatten() for i in range(r)]
            txt += "\n" + _concat(rows, row_frmt, width, prefix)
        elif a_dim == 4:
            d4, d, r, c = a_shp
            t = d4_frmt(a_shp, a, txt, a_dim)
            txt += t
        elif a_dim == 5:
            d5, d4, d, r, c = a_shp
            hdr = "\n" + "-"*25
            for i in range(d5):
                txt += hdr + '\n--({}, ..'.format(i)
                t = d4_frmt(a_shp[1:], a[i], txt, a_dim)
                txt += t
    else:
        txt = "Only integer and float arrays with ndim >= 2 supported"
    if prnt:
        with np.printoptions(precision=deci, linewidth=ln_wdth):
            print(txt)
    else:
        return txt


# ----------------------------------------------------------------------
# (4) prn_ma .... code section
def prn_ma(a, edge=5, deci=2, width=100, prnt=True, prefix="  ."):
    """Format a masked array to preserve columns widths and style.

    Parameters
    ----------
    `a` : masked array
        A masked array
    `prn` : Boolean
        True to print
    `prefix` : text
        Can be "" for no indentation or "   " or the default

    Returns
    -------
    Returns a print version of a masked array formatted with masked
    values and appropriate spacing.
    b = a.reshape(2,4,5) for 3d

    Notes
    -----
    Get a string representation of the array.  Determine the maximum value
    length of a string of the values in the array  and format each column
    using that value.  Pad the result with a leader or replace the prefix
    with ''
    """
    def _fix(v, tmp, prefix):
        """ sub array adjust"""
        r = [['[[', " "], ['[', ""], [']', ""], [']]', ""]]
        for i in r:
            tmp = tmp.replace(i[0], i[1])
        tmp0 = [i.strip().split(' ') for i in tmp.split('\n')]
        N = len(tmp0[0])
        out = [""]
        for i in range(len(tmp0)):
            N = len(tmp0[i])
            out.append((frmt*N).format(*tmp0[i]))
        jn = "\n" + prefix
        v += jn.join([i for i in out])
        v += '\n'
        return v
    # ---- main section ----
#    np.set_printoptions(edgeitems=edge, linewidth=width, precision=3,
#                    suppress=True, nanstr='-n-', threshold=1000)
    dim = a.ndim
    shp = a.shape
    a = np.ma.round(a, decimals=deci)
    a_max = max(len(str(np.ma.max(a))), len(str(np.ma.min(a))))  # largest str
    frmt = '{:>' + str(a_max + 1) + '} '
    v = "\n:Masked array...\n:shape: {}  ndim: {}\n".format(shp, dim)
    if dim == 2:
        v += "\n:.. a[:{}, :{}] ...".format(*shp)
        v = _fix(v, str(a), prefix)
    elif dim == 3:
        for d0 in range(shp[0]):  # dimension blocks
            v += "\n:.. a[{}, :{}, :{}] ...".format(d0, *a[d0].shape)
            v = _fix(v, str(a[d0]), prefix)
    if prnt:
        print(v)
    else:
        return v

# ----------------------------------------------------------------------
# (5) pd and quick_prn
def pd_(a, deci=2, use_names=True, prnt=True):
    """see help for `prn_rec`..."""
    ret = prn_rec(a, deci=deci, prnt=prnt)
    return ret

def quick_prn(a, edges=3, max_lines=25, width=100, decimals=2):
    """Format a structured array by setting the width so it hopefully wraps.
    """
    width = min(len(str(a[0])), width)
    with np.printoptions(edgeitems=edges, threshold=max_lines, linewidth=width,
                         precision=decimals, suppress=True, nanstr='-n-'):
        print("\nArray fields/values...:\n{}\n{}".format(a.dtype.names, a))

def prn_q(a, rows=None, deci=2):
    """Quick print a structured array.

    rows : None or integer
        None, prints all the rows. If an integer, prints [:rows] of the array.
    deci : integer
        Number of decimal places to use for float values.
    """
    cf = _col_format(a, deci=deci)  # ---- the big work done here
    frmt = " ".join(['{' + i[0] + '}' for i in cf])
    hdr = " ".join(['{!s:<' + str(i[1]) + '}' for i in cf])
    if rows is None:
        rows = a.shape[0]
    print(hdr.format(*a.dtype.names))
    for row in range(rows):
        print(frmt.format(*a[row]))

    # ----
# ----------------------------------------------------------------------
# (6) prn_rec and prn_struct .... code section
#  both requires _c_kind_width, _col_format and subsample

def prn_struct(a, rows_m=25, cols_m=None, deci=2, width=100, prnt=True):
    """Format a structured or recarray array.  See prn_rec for more details.
    This variant adds row and column slicing with the `width` and
    `max_lines` parameters.  Requires `subsample` and `_col_format`.
    """
    info = "Array... shape: {}".format(a.shape)
    names = list(a.dtype.names)
    dtn = list(a.dtype.names)
    if cols_m is None:
        cols_m = len(names)//2
    rows_m = min(a.shape[0]//2, rows_m)
    # ---- slice the rows
    a = _slice_rows(a, edge_rows=rows_m)
    form_width = _col_format(a, deci=deci)
    dts = [i[0] for i in form_width]
    wdths = [i[1] for i in form_width]
    tot_width = sum(wdths)
    if tot_width > width:  # ---- split wide arrays
        cs = np.cumsum(wdths)
        cols_m = min(np.sum(cs < width)//2, cols_m)
        a = _slice_head_tail(a, cols_m)
        dtn = list(a.dtype.names)
        form_width = _col_format(a, deci=deci)  # _col_format again
        dts = [i[0] for i in form_width]
        wdths = [i[1] for i in form_width]
    header = " ".join(['{'+"!s:<{}".format(i)+'}' for i in wdths])
    h = header.format(*dtn)
    print("\n{}\n{}".format(h, "-"*len(h)))
    dtf = " ".join(['{' + i + '}' for i in dts])  #z[:, 0]])
    if prnt:
        for i in range(a.shape[0]): # <= rows_m:
            print(dtf.format(*a[i]))
        print("\n{}".format(info))
        print("Head/tail rows: {}, columns: {}".format(rows_m, cols_m))
        return None
    return a
    # ---- done ----


def prn_rec(a, rows_m=25, cols_m=None, deci=2, width=100, prnt=True):
    """Format a structured array with a mixed dtype.

    NOTE : Can be called as `pd_(a, ... )` to emulate pandas dataframes
        You should limit large arrays to a slice ie. a[:50]

    Requires:
    -------
    `a` : array
        A structured/recarray
    `edges` : integer
        Rows to keep from the start and end if max_rows is exceeded.
    `deci` : int
        To facilitate printing, this value is the number of decimal
        points to use for all floating point fields.
    `max_rows : integer
        The number of rows to print before truncating to the `edges` option.
        Change this to a larger value should you need to print whole arrays.
    `subsample` : function
        Requires this fucntion for sampling an array that exceeds max_rows.

    Notes:
    -----
    `_col_format` : does the actual work of obtaining a representation of
    the column format.

    It is not really possible to deconstruct the exact number of decimals
    to use for float values, so a decision had to be made to simplify.
    """
    names = a.dtype.names
    if cols_m is None:
        cols_m = len(names)
    # slice off excess rows
    if a.shape[0] > rows_m:
        a = _slice_rows(a, edge_rows=rows_m)
    # ---- get the column formats from ... _col_format ----
    form_width = _col_format(a, deci=deci)
    dts = [i[0] for i in form_width]
    wdths = [i[1] for i in form_width]
    # slice off excess columns
    c_sum = np.cumsum(wdths)
    N = len(np.where(c_sum < width)[0])
    a = _slice_cols(a, edge_cols=N)
    # slice the formats
    tail = ''
    if N < len(names):
        tail = ' ...'
    row_frmt = " ".join([('{' + i + '}') for i in dts[:N]])
    hdr = ["!s:<" + "{}".format(wdths[i]) for i in range(N)]
    hdr2 = " ".join(["{" + hdr[i] + "}" for i in range(N)])
    header = " id  " + hdr2.format(*names[:N]) + tail
    header = "\n{}\n{}".format(header, "-"*len(header))
    # ---- assemble the print string ----
    txt = [header]
    idx = 0
    for i in range(a.shape[0]):
        txt.append(" {:>03.0f} ".format(idx) + row_frmt.format(*a[i]) + tail)
        idx += 1
    msg = "\n".join([i for i in txt])
    if prnt:
        print(msg)
        return None
    return msg

# ----------------------------------------------------------------------
# (7) prn_ ... code section .....
#  prn_ requires make_row_format
def make_row_format(dim=3, cols=5, a_kind='f', deci=1,
                    a_max=10, a_min=-10, width=100, prnt=False):
    """Format the row based on input parameters

    `dim` - int
        Number of dimensions
    `cols` : int
        Columns per dimension

    `a_kind`, `deci`, `a_max` and `a_min` allow you to specify a data type,
    number of decimals and maximum and minimum values to test formatting.
    """
    if a_kind not in nums:
        a_kind = 'f'
    w_, m_ = [[':{}.0f', '{:0.0f}'], [':{}.{}f', '{:0.{}f}']][a_kind == 'f']
    m_fmt = max(len(m_.format(a_max, deci)), len(m_.format(a_min, deci))) + 1
    w_fmt = w_.format(m_fmt, deci)
    suffix = '  '
    while m_fmt*cols*dim > width:
        cols -= 1
        suffix = '.. '
    row_sub = (('{' + w_fmt + '}')*cols + suffix)
    row_frmt = (row_sub*dim).strip()
    if prnt:
        frmt = "Row format: dim cols: ({}, {})  kind: {} decimals: {}\n\n{}"
        print(dedent(frmt).format(dim, cols, a_kind, deci, row_frmt))
        a = np.random.randint(a_min, a_max+1, dim*cols)
        col_hdr(width//10)  # run col_hdr to produce the column headers
        print(row_frmt.format(*a))
    else:
        return row_frmt


def prn_(a, deci=2, width=100, title="Array", prefix=". . ", prnt=True):
    """Alternate format to prn_nd function.
    Inputs are largely the same.
    """
    def _piece(sub, i, frmt, linewidth):
        """piece together 3D chunks by row"""
        s0 = sub.shape[0]
        block = np.hstack([sub[j] for j in range(s0)])
        txt = ""
        if i is not None:
            fr = (":arr[{}" + ", :{}"*len(a.shape[1:]) + "]\n")
            txt = fr.format(i, *sub.shape)
        for line in block:
            ln = frmt.format(*line)[:linewidth]
            end = ["\n", "...\n"][len(ln) >= linewidth]
            txt += indent(ln + end, ". . ")
        return txt
    # ---- main section ----
    out = "\n{}... ndim: {}  shape: {}\n".format(title, a.ndim, a.shape)
    linewidth = width
    if a.ndim <= 1:
        return a
    if a.ndim == 2:
        a = a.reshape((1,) + a.shape)
    # ---- pull the 1st and 3rd dimension for 3D and 4D arrays
    frmt = make_row_format(dim=a.shape[-3],
                           cols=a.shape[-1],
                           a_kind=a.dtype.kind,
                           deci=deci,
                           a_max=a.max(),
                           a_min=a.min(),
                           width=width,
                           prnt=False)
    if a.ndim == 3:
        s0, _, _ = a.shape
        out += _piece(a, None, frmt, linewidth)  # ---- _piece ----
    elif a.ndim == 4:
        s0, _, _, _ = a.shape
        for i in range(s0):
            out = out + "\n" + _piece(a[i], i, frmt, linewidth)  # ---- _piece
    if prnt:
        with np.printoptions(precision=deci, linewidth=width):
            print(out)
    else:
        return out


# ----------------------------------------------------------------------
# (8)  ---- form prn_3d4d ----
def prn_3d4d(a, deci=2, edgeitems=3, width=100, prnt=True):
    """Another variant for formatting arrays geared towards 3d and 4d
    numeric and text arrays.  For object, structured arrays, see prn_rec
    in arraytools.frmts
    """
    def _row_format(d, r, c, k, deci, a_min, a_max):
        """abbreviated row format, see frmts.py in arraytools
        """
        if k in nums:
            w_, m_ = [[':{}.0f', '{:0.0f}'], [':{}.{}f', '{:0.{}f}']][k == 'f']
        else:
            w_, m_ = ['!s:>{}', '{}']
            deci = 0
        m = max(len(m_.format(a_max, deci)), len(m_.format(a_min, deci))) + 1
        w_fmt = w_.format(m, deci)
        r_fmt = (('{' + w_fmt + '}') * c + '') * 1  # d
        return r_fmt
    #
    def head_tail(s, e):
        """Keep head and tail of array row/column indices [:e], [-e:]
        if s = 10 and e = 3, then
        h_t => array([ 0,  1,  2, -1,  5,  6,  7]) where -1 is a marker
        """
        h_t = np.arange(s)
        if s > e*2:
            h_t = np.concatenate([h_t[:e], [-1], h_t[-e:]])
        return h_t
    # ---- main section
    # bail?
    if (a.ndim not in (3, 4)) or (a.dtype.kind not in nums):
        msg = "Requires a 3D/4D numeric or text array. Kind in (i, f, U)))"
        print(msg)
        return msg
    # (1) base information and reshape 3D to 4D array
    if a.ndim == 3:
        a = a.reshape((1,) + a.shape)
    s4, s3, s2, s1 = a_shp = a.shape
    # ---- start the format process ----
    # (1) split the indices keeping the row, column edgeitems
    e = edgeitems
    s3_ = head_tail(s3, e)
    s2_ = head_tail(s2, e)
    # (2) assemble information for _row_format
    d_, r_, c_ = a_shp[-3:]
    if a.dtype.kind in ('U', 'S', 'b', 'O', 'V'):
        n = int(a.dtype.str.lstrip('<^>|bOUSV')) #+ 1
        a_min = a_max = int('1'*n)  # cheat to get a number len of string
    elif a.dtype.kind in ('i', 'u', 'f'):
        a_min = a.min()
        a_max = a.max()
    fm0 = _row_format(d_, r_, c_, a.dtype.kind, 2, a_min, a_max)  # d=1
    split_0 = c_ > e*2  # boolean check
    if split_0:  # e in place of c_
        fm1 = _row_format(d_, r_, e, a.dtype.kind, 2, a_min, a_max)
    # (3) process
    t = "Array... ndim {}  shape{}".format(a.ndim, a.shape)
    for k in range(s4):
        for j in s2_:
            row = []
            for i in s3_:
                r_ = a[k][i][j]
                if split_0:
                    sub = fm1.format(*r_[:e]) + " ..." + fm1.format(*r_[-e:])
                else:
                    sub = fm0.format(*r_)
                if j == -1:
                    row.append("{!s:^{}}".format(" . .", len(sub)))
                else:
                    row.append(sub)
            s = "  ".join(row)
            if len(s) > width:
                s = s[:width] + "..."
            t += "\n|" + s + " |"
        t += "\n|=> ({} {} {} {})\n".format(k, s3, s2, s1)
    if prnt:
        print(t)
        return None
    return t


# ----------------------------------------------------------------------
# (9)  ---- prn .... calls the other print methods
#
def prn(a, rows=20, cols=3, deci=2, width=100, title="", prnt=True):
    """Calling function for all the array print options

    References:
    -----------
    `<https://github.com/numpy/numpy/blob/
    f85c71a6d16ab64695f07fde23e2c19104d36208/numpy/core/numerictypes.py>`_.
    ::
        prn_nd(a, deci=2, width=100, title="Array", prefix="  .", prnt=True)
        prn_(a, deci=2, width=100, title="Array", prefix=". . ", prnt=True)
        prn_ma(a, deci=2, prnt=True, prefix="  .")
        prn_struct(a, rows_m=25, cols_m=None, deci=2, width=100, prnt=True)
        prn_rec(a, rows_m=25, cols_m=None, deci=2, width=100, prnt=True)
        prn_3d4d(a, deci=2, edgeitems=3, width=100, prnt=True)
    """
    #_kind_list = ['b', 'u', 'i', 'f', 'c', 'S', 'U', 'V', 'O', 'M', 'm']
    kind = a.dtype.kind
    dt = a.dtype
    ndim = a.ndim
    if title != "":
        msg = "\n{}\n{}".format('-'*30, title)
        print(msg)
    if np.ma.isMaskedArray(a):          # ---- use this for masked arrays
        prn_ma(a, deci=deci, prnt=True, prefix="  .")
    elif kind in ('i', 'u', 'f', 'c'):  # ---- float-kind
        v = "{!s:<3} : {}".format(kind, dt)
        if ndim < 2:
            print(a)
        elif ndim == 2:
            prn_(a)
        elif ndim > 2:
            if ndim <= 4:
                prn_3d4d(a, deci=deci, edgeitems=cols, width=width, prnt=True)
            else:
                prn_nd(a, deci=deci, width=width, title="Array",
                       prefix="  .", prnt=True)
            #prn_(a)
            #prn_3d4d(a, deci=deci, edgeitems=cols, width=width, prnt=True)
    elif kind in ('S', 'U'):            # ---- unicode, string
        v = "{!s:<3} : {}".format(kind, dt)
    elif kind in ('V', 'O'):            # ---- void, object arrays
        #v = "{!s:<3} : {}".format(kind, dt)
        #prn_struct(a, rows_m=5, cols_m=None, deci=2, width=100, prnt=True)
        prn_rec(a, rows_m=rows, cols_m=cols, deci=deci, width=width, prnt=prnt)
    else:
        v = "{!s:<3} : {}".format(kind, dt)
        print(v)


# ----------------------------------------------------------------------
# (8)  ---- sample data ----

def _data():
    """base file"""
    pth = _data.__code__.co_filename
    pth = pth.replace("frmts.py", "")
    a = np.load(pth + "Data/points_2000.npy")
    b = np.load(pth + "Data/sample_20.npy")
    c_d = np.load(pth + "Data/masked_data.npy")
    c_m = np.load(pth + "Data/masked_mask.npy")
    c = np.ma.MaskedArray(c_d, c_m)
    d = np.load(pth + "Data/ndim4.npy")
    return a, b, c, d

# -------------------------
if __name__ == "__main__":
    """Main section...   """
#    print("Script... {}".format(script))
#    row_frmt = make_row_format()
#    a, b, c, d = _data()
