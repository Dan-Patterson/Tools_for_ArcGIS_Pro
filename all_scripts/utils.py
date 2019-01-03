# -*- coding: utf-8 -*-
"""
utils
=====

Script :   utils.py

Author :   Dan_Patterson@carleton.ca

Modified : 2018-11-22

Purpose:  tools for working with numpy arrays

Useage:
-------

**doc_func(func=None)** : see get_func and get_modu

**get_func** : retrieve function information
::
    get_func(func, line_nums=True, verbose=True)
    print(art.get_func(art.main))

    Function: .... main ....
    Line number... 1334
    Docs:
    Do nothing
    Defaults: None
    Keyword Defaults: None
    Variable names:
    Source code:
       0  def main():
       1   '''Do nothing'''
       2      pass

**get_modu** : retrieve module info

    get_modu(obj, code=False, verbose=True)

**info(a, prn=True)** : retrieve array information
::
    - array([(0, 1, 2, 3, 4), (5, 6, 7, 8, 9),
             (10, 11, 12, 13, 14), (15, 16, 17, 18, 19)],
      dtype=[('A', '<i8'), ('B', '<i8')... snip ..., ('E', '<i8')])
    ---------------------
    Array information....
    array
      |__shape (4,)
      |__ndim  1
      |__size  4
      |__type  <class 'numpy.ndarray'>
    dtype      [('A', '<i8'), ('B', '<i8') ... , ('E', '<i8')]
      |__kind  V
      |__char  V
      |__num   20
      |__type  <class 'numpy.void'>
      |__name  void320
      |__shape ()
      |__description
         |__name, itemsize
         |__['A', '<i8']
         |__['B', '<i8']
         |__['C', '<i8']
         |__['D', '<i8']
         |__['E', '<i8']

References
----------
`<http://pro.arcgis.com/en/pro-app/arcpy/data-access/numpyarraytotable.htm>`_.

`<http://pro.arcgis.com/en/pro-app/arcpy/data-access/tabletonumpyarray.htm>`_.

---------------------------------------------------------------------
"""
# pylint: disable=C0103
# pylint: disable=R1710
# pylint: disable=R0914

import sys
from textwrap import dedent, indent, wrap
import warnings
import numpy as np

warnings.simplefilter('ignore', FutureWarning)

#from arcpytools import fc_info, tweet  #, frmt_rec, _col_format
#import arcpy

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


__all__ = ['time_deco',
           'run_deco',
           'doc_func',
           'get_func',
           'get_modu',
           'dirr',
           'wrapper',
           '_utils_help_'
           ]

# ---- decorators and helpers ------------------------------------------------
#
def time_deco(func):  # timing originally
    """Timing decorator function

    Requires:
    ---------
    The following import.  Uncomment the import or move it inside the script.

    >>> from functools import wraps

    Useage::

        @time_deco  # on the line above the function
        def some_func():
            '''do stuff'''
            return None

    """
    import time
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        t_0 = time.perf_counter()        # start time
        result = func(*args, **kwargs)   # ... run the function ...
        t_1 = time.perf_counter()        # end time
        dt = t_1 - t_0
        print("\nTiming function for... {}".format(func.__name__))
        if result is None:
            result = 0
        print("  Time: {: <8.2e}s for {:,} objects".format(dt, result))
        # return result                   # return the result of the function
        return dt                       # return delta time
    return wrapper


def run_deco(func):
    """Prints basic function information and the results of a run.

    Requires:
    ---------
    The following import.  Uncomment the import or move it inside the script.

    >>> from functools import wraps

    Useage::

        @run_deco  # on the line above the function
        def some_func():
            '''do stuff'''
            return None

    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        """wrapper function"""
        frmt = "\n".join(["Function... {}", "  args.... {}",
                          "  kwargs.. {}", "  docs.... {}"])
        ar = [func.__name__, args, kwargs, func.__doc__]
        print(dedent(frmt).format(*ar))
        result = func(*args, **kwargs)
        print("{!r:}\n".format(result))  # comment out if results not needed
        return result                    # for optional use outside.
    return wrapper


# ----------------------------------------------------------------------------
# ---- (1) doc_func ... code section ... ----
def doc_func(func=None, verbose=True):
    """(doc_func)...Documenting code using inspect

    Requires:
    ---------
    >>> import inspect  # module

    Returns
    -------
    A listing of the source code with line numbers

    Parameters
    ----------
    func : function
        Function name to document, without quotes
    verbose : Boolean
        True prints the result, False returns a string of the result.

    Notes
    -----

    Source code for...
    ::

        module level
        - inspect.getsourcelines(sys.modules[__name__])[0]

        function level
        - as a list => inspect.getsourcelines(num_41)[0]
        - as a string => inspect.getsource(num_41)

        file level
        - script = sys.argv[0]

    """
    def demo_func():
        """dummy...
        : Demonstrates retrieving and documenting module and function info.
        """
        def sub():
            """sub in dummy"""
            pass
        return None
    #
    import inspect
    if func is None:
        func = demo_func
    if not inspect.isfunction(func):
        out = "\nError... `{}` is not a function, but is of type... {}\n"
        print(out.format(func.__name__, type(func)))
        return None
    script = sys.argv[0]  # a useful way to get a file's name
    lines, line_num = inspect.getsourcelines(func)
    code = "".join(["{:4d}  {}".format(idx+line_num, line)
                    for idx, line in enumerate(lines)])
    nmes = ['args', 'varargs', 'varkw', 'defaults', 'kwonlyargs',
            'kwonlydefaults', 'annotations']
    f = inspect.getfullargspec(func)
    f_args = "\n".join([str(i) for i in list(zip(nmes, list(f)))])
    args = [line_num, code,
            inspect.getcomments(func),
            inspect.isfunction(func),
            inspect.ismethod(func),
            inspect.getmodulename(script),
            f_args]
    frmt = """
    :----------------------------------------------------------------------
    :---- doc_func(func) ----
    :Code for a function on line...{}...
    :
    {}
    Comments preceeding function
    {}
    function?... {} ... or method? {}
    Module name... {}
    Full specs....
    {}
    ----------------------------------------------------------------------
    """
    out = (dedent(frmt)).format(*args)
    if verbose:
        print(out)
    else:
        return out


# ----------------------------------------------------------------------
# ---- (2) get_func .... code section ----
def get_func(func, line_nums=True, verbose=True):
    """Get function information (ie. for a def)

    Requires
    --------
    >>> from textwrap import dedent, indent, wrap
    >>> import inspect

    Returns
    -------
    The function information includes arguments and source code.
    A string is returned for printing.

    Notes
    -----
    Import the module containing the function and put the object name in
    without quotes...

    >>> from arraytools.utils import get_func
    >>> get_func(get_func)  # returns this source code etc.
    """
    frmt = """
    :-----------------------------------------------------------------
    :Function: .... {} ....
    :Line number... {}
    :Docs:
    {}
    :Defaults: {}
    :Keyword Defaults: {}
    :Variable names:
    {}\n
    :Source code:
    {}
    :
    :-----------------------------------------------------------------
    """
    import inspect
    from textwrap import dedent, wrap

    if not inspect.isfunction(func):
        out = "\nError... `{}` is not a function, but is of type... {}\n"
        print(out.format(func.__name__, type(func)))
        return None

    lines, ln_num = inspect.getsourcelines(func)
    if line_nums:
        code = "".join(["{:4d}  {}".format(idx + ln_num, line)
                        for idx, line in enumerate(lines)])
    else:
        code = "".join(["{}".format(line) for line in lines])

    vars_ = ", ".join([i for i in func.__code__.co_varnames])
    vars_ = wrap(vars_, 50)
    vars_ = "\n".join([i for i in vars_])
    args = [func.__name__, ln_num, dedent(func.__doc__), func.__defaults__,
            func.__kwdefaults__, indent(vars_, "    "), code]
    code_mem = dedent(frmt).format(*args)
    if verbose:
        print(code_mem)
    else:
        return code_mem


# ----------------------------------------------------------------------
# ---- (3) get_modu .... code section ----
def get_modu(obj, code=False, verbose=True):
    """Get module (script) information, including source code for
    documentation purposes.

    Requires
    --------
    >>> from textwrap import dedent, indent
    >>> import inspect

    Returns
    -------
    A string is returned for printing.  It will be the whole module
    so use with caution.

    Notes
    -----
    Useage::

    >>> from arraytools.utils import get_modu
    >>> get_modu(tools, code=False, verbose=True)
    >>> # No quotes around module name, code=True for module code

   """
    frmt = """
    :-----------------------------------------------------------------
    :Module: .... {} ....
    :------
    :File: ......
    {}\n
    :Docs: ......
    {}\n
    :Members: .....
    {}
    """
    frmt0 = """
    :{}
    :-----------------------------------------------------------------
    """
    frmt1 = """
    :Source code: .....
    {}
    :
    :-----------------------------------------------------------------
    """
    import inspect
    from textwrap import dedent

    if not inspect.ismodule(obj):
        out = "\nError... `{}` is not a module, but is of type... {}\n"
        print(out.format(obj.__name__, type(obj)))
        return None
    if code:
        lines, _ = inspect.getsourcelines(obj)
        frmt = frmt + frmt1
        code = "".join(["{:4d}  {}".format(idx, line)
                        for idx, line in enumerate(lines)])
    else:
        lines = code = ""
        frmt = frmt + frmt0
    memb = [i[0] for i in inspect.getmembers(obj)]
    args = [obj.__name__, obj.__file__, obj.__doc__, memb, code]
    mod_mem = dedent(frmt).format(*args)
    if verbose:
        print(mod_mem)
    else:
        return mod_mem

# ----------------------------------------------------------------------
# ---- (4) dirr .... code section ----
def dirr(obj, colwise=False, cols=4, sub=None, prn=True):
    """A formatted `dir` listing of an object, module, function... anything you
    can get a listing for.

    Source : arraytools.py_tools has a pure python equivalent

    Other : arraytools `__init__._info()` has an abbreviated version

    Parameters
    ----------
    colwise : boolean
        `True` or `1`, otherwise, `False` or `0`
    cols : number
      pick a size to suit
    sub : text
      sub array with wildcards

    - `arr*` : begin with `arr`
    - `*arr` : endswith `arr` or
    - `*arr*`: contains `arr`
    prn : boolean
      `True` for print or `False` to return output as string

    Return:
    -------
    A directory listing of a module's namespace or a part of it if the
    `sub` option is specified.

    Notes
    -----
    See the `inspect` module for possible additions like `isfunction`,
    `ismethod`, `ismodule`

    **Examples**::

        dirr(art, colwise=True, cols=3, sub=None, prn=True)  # all columnwise
        dirr(art, colwise=True, cols=3, sub='arr', prn=True) # just the `arr`'s

          (001)    _arr_common     arr2xyz         arr_json
          (002)    arr_pnts        arr_polygon_fc  arr_polyline_fc
          (003)    array2raster    array_fc
          (004)    array_struct    arrays_cols
    """
    err = """
    ...No matches found using substring .  `{0}`
    ...check with wildcards, *, ... `*abc*`, `*abc`, `abc*`
    """
    d_arr = dir(obj)
    a = np.array(d_arr)
    dt = a.dtype.descr[0][1]
    if sub not in (None, '', ' '):
        start = [0, 1][sub[0] == "*"]
        end = [0, -1][sub[-1] == "*"]
        if not start and abs(end):
            a = [i for i in d_arr
                 if i.startswith(sub[start:end], start, len(i))]
        elif start and abs(end):
            a = [i for i in d_arr
                 if sub[1:4] in i[:len(i)]]
        elif abs(end):
            sub = sub.replace("*", "")
            a = [i for i in d_arr
                 if i.endswith(sub, start, len(i))]
        else:
            a = []
        if len(a) == 0:
            print(dedent(err).format(sub))
            return None
        num = max([len(i) for i in a])
    else:
        num = int("".join([i for i in dt if i.isdigit()]))
    frmt = ("{{!s:<{}}} ".format(num)) * cols
    if colwise:
        z = np.array_split(a, cols)
        zl = [len(i) for i in z]
        N = max(zl)
        e = np.empty((N, cols), dtype=z[0].dtype)
        for i in range(cols):
            n = min(N, zl[i])
            e[:n, i] = z[i]
    else:
        csze = len(a) / cols
        rows = int(csze) + (csze % 1 > 0)
        z = np.array_split(a, rows)
        e = np.empty((len(z), cols), dtype=z[0].dtype)
        N = len(z)
        for i in range(N):
            n = min(cols, len(z[i]))
            e[i, :n] = z[i][:n]
    if hasattr(obj, '__name__'):
        args = ["-"*70, obj.__name__, obj]
    else:
        args = ["-"*70, type(obj), "np version"]
    txt_out = "\n{}\n| dir({}) ...\n|    {}\n-------".format(*args)
    cnt = 1
    for i in e:
        txt_out += "\n  ({:>03.0f})    {}".format(cnt, frmt.format(*i))
        cnt += cols
    if prn:
        print(txt_out)
    else:
        return txt_out


# ----------------------------------------------------------------------
# ---- (5) wrapper .... code section ----
def wrapper(a, wdth=70):
    """Wrap stuff using textwrap.wrap

    Notes:
    -----
    TextWrapper class
    __init__(self, width=70, initial_indent='', subsequent_indent='',
             expand_tabs=True, replace_whitespace=True,
             fix_sentence_endings=False, break_long_words=True,
             drop_whitespace=True, break_on_hyphens=True, tabsize=8,
             *, max_lines=None, placeholder=' [...]')
    """
    if isinstance(a, np.ndarray):
        txt = [str(i) for i in a.tolist()]
        txt = ", ".join(txt)
    elif isinstance(a, (list, tuple)):
        txt = ", ".join([str(i) for i in a])
    txt = "\n".join(wrap(txt, width=wdth))
    return txt


def _utils_help_():
    """arraytools.utils help...

    Function list follows:
    """
    _hf = """
    :-------------------------------------------------------------------:
    : ---- arrtools functions  (loaded as 'art') ----
    : ---- from utils.py
    (1)  doc_func(func=None)
         documenting code using inspect
    (2)  get_func(obj, line_nums=True, verbose=True)
         pull in function code
    (3)  get_modu(obj)
         pull in module code
    (4)  dirr(a)  object info
    (5)  wrapper(a)  format objects as a string
    :-------------------------------------------------------------------:
    """
    print(dedent(_hf))

# ----------------------------------------------------------------------
# .... final code section producing the featureclass and extendtable

#

# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
    testing = True
    print('\n{} in source script... {}'.format(__name__, script))
        # parameters here
else:
    testing = False
    # parameters here
