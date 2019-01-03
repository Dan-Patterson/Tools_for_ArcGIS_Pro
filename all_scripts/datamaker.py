# -*- coding: UTF-8 -*-
"""
datamaker
=========

Script:   datamaker.py

Author:   Dan_Patterson@carleton.ca

Modified: 2018-11-03

Purpose:  tools for working with numpy arrays

Useage:

References:

`blog post_. https://community.esri.com/blogs/dan_patterson/2016/04/04/
numpy-lessons-6-creating-data-for-testing-purposes`

---------------------------------------------------------------------
"""
# ---- imports, formats, constants ----
import sys
from functools import wraps
import numpy as np
import numpy.lib.recfunctions as rfn
# Required imports

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


# ----------------------------------------------------------------------------
# ---- Required constants  ... see string module for others
str_opt = ['0123456789',
           '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~',
           'abcdefghijklmnopqrstuvwxyz',
           'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
           'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
           ]


# ----------------------------------------------------------------------------
# ---- decorators and helpers ----
def func_run(func):
    """Prints basic function information and the results of a run.
    :Required:  from functools import wraps
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("\nFunction... {}".format(func.__name__))
        print("  args.... {}\n  kwargs.. {}".format(args, kwargs))
        print("  docs.... \n{}".format(func.__doc__))
        result = func(*args, **kwargs)
        print("{!r:}\n".format(result))  # comment out if results not needed
        return result                    # for optional use outside.
    return wrapper


def time_deco(func):  # timing originally
    """timing decorator function
    print("\n  print results inside wrapper or use <return> ... ")
    """
    import time
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()        # start time
        result = func(*args, **kwargs)  # ... run the function ...
        t1 = time.perf_counter()        # end time
        dt = t1 - t0
        print("\nTiming function for... {}".format(func.__name__))
        if result is None:
            result = 0
        print("  Time: {: <8.2e}s for {:,} objects".format(dt, result))
        return result                   # return the result of the function
        return dt                       # return delta time
    return wrapper


def strip_concatenate(in_flds, strip_list=[" ", ",", None]):
    """Provide a list of fields ie [a, b, c] to strip spaces and remove nulls

    - use: python parser
    - syntax: strip_stuff('!a!, !b!, !c!]) assumed field names

    """
    fixed = []
    fmt = []
    for i in in_flds:
        if i not in strip_list:
            fixed.append(i)
            fmt.append("{}")
    frmt = " ".join([f for f in fmt])
    frmt.strip()
    fixed = [str(i).strip() for i in fixed]
    result = frmt.format(*fixed)
    return result


# ----------------------------------------------------------------------------
# ---- functions
def concat_flds(a, flds=None, out_name="Concat", sep=" ", with_ids=True):
    """Concatenate a sequence of fields to string format and return a
    structured array or ndarray

    Requires
    --------

    - arrs : a list single arrays of the same length
    -  sep : the separator between lists
    -  name : used for structured array
    """
    strip_list = [" ", ",", None]
    if (flds is None) or (a.dtype.names is None):
        msg = "Field/column names are required or need to exist in the array."
        print(msg)
        return a
    N = min(len(flds), len(a.dtype.names))
    if N < 2:
        print("Two fields are required for concatenation")
        return a
    s0 = [str(i) if i not in strip_list else '' for i in a[flds[0]]]
    s1 = [str(i) if i not in strip_list else '' for i in a[flds[1]]]
    c = [("{}{}{}".format(i, sep, j)).strip() for i, j in list(zip(s0, s1))]
    if N > 2:
        for i in range(2, len(flds)):
            f = flds[i]
            f = [str(i) if i not in strip_list else '' for i in a[flds[i]]]
            c = ["{}{}{}".format(i, sep, j) for i, j in list(zip(c, f))]
    c = np.asarray(c)
    sze = c.dtype.str
    if out_name is not None:
        c.dtype = [(out_name, sze)]
    else:
        out_name = 'f'
    if with_ids:
        tmp = np.copy(c)
        dt = [('IDs', '<i8'), (out_name, sze)]
        c = np.empty((tmp.shape[0], ), dtype=dt)
        c['IDs'] = np.arange(1, tmp.shape[0] + 1)
        c[out_name] = tmp
    return c


def colrow_txt(N=10, cols=2, rows=2, zero_based=True):
    """  Produce spreadsheet like labels either 0- or 1-based.

    Requires
    --------
    N : number
        Number of records/rows to produce.
    cols/rows : numbers
        This combination will control the output of the values
        cols=2, rows=2 - yields (A0, A1, B0, B1)
        as optional classes regardless of the number of records being produced
    zero-based : boolean
        True for conventional array structure,
        False for spreadsheed-style classes
    """
    if zero_based:
        start = 0
    else:
        start = 1
        rows = rows + 1
    UC = (list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))[:cols]  # see constants
    dig = (list('0123456789'))[start:rows]
    cr_vals = [c + r for r in dig for c in UC]
    colrow = np.random.choice(cr_vals, N)
    return colrow


def rowcol_txt(N=10, rows=2, cols=2):
    """  Produce array-like labels in a tuple format.
    """
    rc_vals = ["({},{})".format(r, c)
               for c in range(cols)
               for r in range(rows)]
    rowcol = np.random.choice(rc_vals, N)
    return rowcol


def pnts_IdShape(N=10, x_min=0, x_max=10, y_min=0, y_max=10, simple=True):
    """Create an array with a nested dtype which emulates a shapefile's
    data structure.  This array is used to append other arrays to enable
    import of the resultant into ArcMap.  Array construction, after hpaulj

    http://stackoverflow.com/questions/32224220/
        methods-of-creating-a-structured-array
    """
    Xs = np.random.randint(x_min, x_max, size=N)
    Ys = np.random.randint(y_min, y_max, size=N)
    IDs = np.arange(0, N)
    c_stack = np.column_stack((IDs, Xs, Ys))
    if simple:     # version 1  short version, optional form
        dt = [('ID', '<i4'), ('X', '<f8'), ('Y', '<f8')]
        a = np.ones(N, dtype=dt)
        a['ID'] = c_stack[:, 0]
        a['X'] = c_stack[:, 1]         # this line too
        a['Y'] = c_stack[:, 2]
    else:          # version 2
        dt = [('ID', '<i4'), ('Shape', ([('X', '<f8'), ('Y', '<f8')]))]
        a = np.ones(N, dtype=dt)
        a['ID'] = c_stack[:, 0]
        a['Shape']['X'] = c_stack[:, 1]
        a['Shape']['Y'] = c_stack[:, 2]
    return a


def rand_text(N=10, cases=3, vals=str_opt[3]):
    """Generate N samples from the letters of the alphabet denoted by the
    number of cases.  If you want greater control on the text and
    probability, see rand_case or rand_str.

    vals:  see str_opt in required constants section
    """
    vals = list(vals)
    txt_vals = np.random.choice(vals[:cases], N)
    return txt_vals


def rand_str(N=10, low=1, high=10, vals=str_opt[3]):
    """Returns N strings constructed from 'size' random letters to form a
    string

    - create the cases as a list:  string.ascii_lowercase or ascii_uppercase
    - determine how many letters. Ensure min <= max. Add 1 to max alleviate
      low==high
    - shuffle the case list each time through loop
    """
    vals = list(vals)
    letts = np.arange(min([low, high]), max([low, high])+1)  # num letters
    result = []
    for i in range(N):
        np.random.shuffle(vals)
        size = np.random.choice(letts, 1)
        result.append("".join(vals[:size]))
    result = np.array(result)
    return result


def rand_case(N=10, cases=["Aa", "Bb"], p_vals=[0.8, 0.2]):
    """Generate N samples from a list of classes with an associated probability

    - ensure: len(cases)==len(p_vals) and  sum(p_values) == 1
    - small sample sizes will probably not yield the desired p-values
    """
    p = (np.array(p_vals))*N   # convert to integer
    kludge = [np.repeat(cases[i], p[i]).tolist() for i in range(len(cases))]
    case_vals = np.array([val for i in range(len(kludge))
                          for val in kludge[i]])
    np.random.shuffle(case_vals)
    return case_vals


def rand_int(N=10, begin=0, end=10):
    """Generate N random integers within the range begin - end
    """
    int_vals = np.random.randint(begin, end, size=(N))
    return int_vals


def rand_float(N=10, begin=0, end=10):
    """Generate N random floats within the range begin - end.

    Technically, N random integers are produced then a random
    amount within 0-1 is added to the value
    """
    float_vals = np.random.randint(begin, end-1, size=(N))
    float_vals = float_vals + np.random.rand(N)
    return float_vals


def blog_post():
    """sample run"""
    N = 10000
    id_shape = pnts_IdShape(N,
                            x_min=300000,
                            x_max=305000,
                            y_min=5000000,
                            y_max=5005000)
    case1_fld = rand_case(N,
                          cases=['A', 'B', 'C', 'D'],
                          p_vals=[0.4, 0.3, 0.2, 0.1])
    int_fld = rand_int(N, begin=0, end=10)
    float_0 = rand_float(N, 5, 15)
    float_1 = rand_float(N, 5, 20)
    fld_names = ['Case', 'Observed', 'Size', 'Mass']
    fld_data = [case1_fld, int_fld, float_0, float_1]
    arr = rfn.append_fields(id_shape, fld_names, fld_data, usemask=False)
    return arr


def blog_post2(N=20):
    """sample run
    : import arcpy
    : out_fc = r'C:\GIS\A_Tools_scripts\Graphing\Graphing_tools\
    :            Graphing_tools.gdb\data_01'
    : arcpy.da.NumPyArrayToFeatureClass(a, out_fc, ['X', 'Y'])
    """
    ids = np.arange(1, N + 1)  # construct the base array of IDs to append to
    ids = np.asarray(ids, dtype=[('Ids', '<i4')])
    int_fld = rand_int(N, begin=10, end=1000)
    case1 = rand_case(N,
                      cases=['N', 'S', 'E', 'W', ''],
                      p_vals=[0.1, 0.1, 0.2, 0.2, 0.4])
    case2 = rand_case(N,
                      cases=['Maple', 'Oak', 'Elm', 'Pine', 'Spruce'],
                      p_vals=[0.3, 0.15, 0.2, 0.25, 0.1])
    case3 = rand_case(N,
                      cases=['Ave', 'St', 'Crt'],
                      p_vals=[0.3, 0.6, 0.1])
    case4 = rand_case(N,
                      cases=['Carp', 'Almonte', 'Arnprior', 'Carleton Place'],
                      p_vals=[0.3, 0.3, 0.2, 0.2])
    fld_names = ['Str_Number', 'Prefix', 'Str_Name', 'Str_Type', 'Town']
    fld_data = [int_fld, case1, case2, case3, case4]
    arr = rfn.append_fields(ids, fld_names, fld_data, usemask=False)
    return arr


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """create ID,Shape,{txt_fld,int_fld...of any number}
    """
    a = blog_post2(N=20)
#    a = np.array(['a11', 'b11', 'c12', 'a13', 'b15', 'c15'])
#
#    check = np.array(['11', '12', '13'])
#
#    is_there = np.asarray([[np.char.rfind(i, val) for val in check]
#                          for i in a]).max(axis=1)
