# -*- coding: UTF-8 -*-
"""
py_tools
========

Script :   py_tools.py

Author :   Dan_Patterson@carleton.ca

Modified: 2018-10-15

-------

Purpose : tools for working with python, numpy and other python packages

- iterables :
    _flatten, flatten_shape, pack, unpack
- folders :
    get_dir, folders, sub-folders, dir_py
Useage:

References:

------------------------------------------------------------------------------
"""
# pylint: disable=C0103
# pylint: disable=R1710
# pylint: disable=R0914

# ---- imports, formats, constants ---------------------------------------
import sys
import os
from textwrap import dedent
import numpy as np


ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


__all__ = ['comp_info',
           'get_dir', 'folders', 'sub_folders',  # basic folder functions
           'dir_py',    # object and directory functions
           '_flatten', 'flatten_shape',  # iterables
           'pack', 'unpack',
           'combine_dicts']


# ---- (1) computer, python stuff ... code section ---------------------------
#
def comp_info():
    """Return information for the computer and python version
    """
    import platform
    winv = platform.platform()
    py_ver = platform.python_version()
    plat = platform.architecture()
    proc = platform.processor()
    p_node = platform._node()
    u_name = platform.uname()
    ud = u_name._asdict()
    udl = list(zip(ud.keys(), ud.values()))
    frmt = """
    ---------------------------
    Computer/python information

    Platform:        {}
    python version:  {}
    windows version: {}
    processor:       {}
    node:            {}
    user/machine:    {}\n
    Alternate form...."""
    args = [winv, py_ver, plat, proc, p_node, u_name]
    print(dedent(frmt).format(*args))
    print("\n".join(["{:<10}: {}".format(*i) for i in udl]))


# ---- (2) general file functions ... code section ---------------------------
#
def get_dir(path):
    """Get the directory list from a path, excluding geodatabase folders.
    Used by.. folders

    >>> get_dir('C:/Git_Dan/arraytools')
    ['C:/Git_Dan/arraytools/.spyproject',
     'C:/Git_Dan/arraytools/analysis',
     ... snip ...
     'C:/Git_Dan/arraytools/__pycache__']
    >>> # ---- common path prefix
    >>> os.path.commonprefix(get_dir('C:/Git_Dan/arraytools'))
    'C:/Git_Dan/arraytools/'
    """
    if os.path.isfile(path):
        path = os.path.dirname(path)
    p = os.path.normpath(path)
    full = [os.path.join(p, v) for v in os.listdir(p)]
    dirlist = [val for val in full if os.path.isdir(val)]
    return dirlist


def folders(path, first=True, prefix=""):
    """ Print recursive listing of folders in a path.  Make sure you `raw`
    format the path...
    ::
        r'c:\Temp'  or 'c:/Temp' or 'c:\\Temp'

    - Requires : _get_dir .... also, an example of path common prefix
    """
    if first:  # Detect outermost call, print a heading
        print("-"*30 + "\n|.... Folder listing for ....|\n|--{}".format(path))
        prefix = "|-"
        first = False
        cprev = path
    dirlist = get_dir(path)
    for d in dirlist:
        fullname = os.path.join(path, d)  # Turn name into full pathname
        if os.path.isdir(fullname):       # If a directory, recurse.
            cprev = path
            pad = ' ' * len(cprev)
            n = d.replace(cprev, pad)
            print(prefix + "-" + n)  # fullname) # os.path.relpath(fullname))
            p = "  "
            folders(fullname, first=False, prefix=p)
    # ----


def sub_folders(path, combine=False):
    """Print the folders in a path, excluding '.' folders
    This is the best one.
    """
    import pathlib
    print("Path...\n{}".format(path))
    if combine:
        r = " "*len(path)
    else:
        r = ""
    f = "\n".join([(p._str).replace(path, r)
                   for p in pathlib.Path(path).iterdir()
                   if p.is_dir() and "." not in p._str])
    print("{}".format(f))


def env_list(pth, ordered=False):
    """List folders and files in a path
    """
    import os
    d = []
    for item in os.listdir(pth):
        check = os.path.join(pth, item)
        check = check.replace("\\", "/")
        if os.path.isdir(check) and ("." not in check):
            d.append(check)
    d = np.array(d)
    if ordered:
        d = d[np.argsort(d)]
    return d


# ---- (3) dirr ... code section ... -----------------------------------------
#
def dir_py(obj, colwise=False, cols=4, prn=True):
    """The non-numpy version of dirr
    """
    from itertools import zip_longest as zl
    a = dir(obj)
    w = max([len(i) for i in a])
    frmt = (("{{!s:<{}}} ".format(w)))*cols
    csze = len(a) / cols  # split it
    csze = int(csze) + (csze % 1 > 0)
    if colwise:
        a_0 = [a[i: i+csze] for i in range(0, len(a), csze)]
        a_0 = list(zl(*a_0, fillvalue=""))
    else:
        a_0 = [a[i: i+cols] for i in range(0, len(a), cols)]
    if hasattr(obj, '__name__'):
        args = ["-"*70, obj.__name__, obj]
    else:
        args = ["-"*70, type(obj), "py version"]
    txt_out = "\n{}\n| dir({}) ...\n|    {}\n-------".format(*args)
    cnt = 0
    for i in a_0:
        cnt += 1
        txt = "\n  ({:>03.0f})  ".format(cnt)
        frmt = (("{{!s:<{}}} ".format(w)))*len(i)
        txt += frmt.format(*i)
        txt_out += txt
    if prn:
        print(txt_out)
    else:
        return txt_out


# ---- (4) iterables ---------------------------------------------------------
#
def _flatten(a_list, flat_list=None):
    """Change the isinstance as appropriate.

    Flatten an object using recursion

    see: itertools.chain() for an alternate method of flattening.
    """
    if flat_list is None:
        flat_list = []
    for item in a_list:
        if hasattr(item, '__iter__'):
            _flatten(item, flat_list)
        else:
            flat_list.append(item)
    return flat_list


def flatten_shape(shp, completely=False):
    """Flatten a array or geometry shape object using itertools.

    Parameters:
    -----------

    shp :
       an array or an array representing polygon, polyline, or point shapes
    completely :
       True returns points for all objects
       False, returns Array for polygon or polyline objects

    Notes:
    ------
    - for conventional array-like objects use `completely = False` to flatten
      the object completely.
    - for geometry objects, use `True` for polygon and polylines to retain their
      parts, but for points, use `False` since you need to retain the x,y pair
    - `__iter__` property: Polygon, Polyline, Array all have this property...
      Points do not.
    """
    import itertools
    if completely:
        vals = [i for i in itertools.chain(shp)]
    else:
        vals = [i for i in itertools.chain.from_iterable(shp)]
    return vals


def pack(a, param='__iter__'):
    """Pack an iterable into an ndarray or object array
    """
    if not hasattr(a, param):
        return a
    return np.asarray([np.asarray(i) for i in a])


def unpack(iterable, param='__iter__'):
    """Unpack an iterable based on the param(eter) condition using recursion.

    Notes:
    ------
    - Use `flatten` for recarrays or structured arrays.
    - See main docs for more information and options.
    - To produce uniform array from this, use the following after this is done.
    >>> out = np.array(xy).reshape(len(xy)//2, 2)

    - To check whether unpack can be used.
    >>> isinstance(x, (list, tuple, np.ndarray, np.void)) like in flatten above
    """
    xy = []
    for x in iterable:
        if hasattr(x, param):
            xy.extend(unpack(x))
        else:
            xy.append(x)
    return xy

def combine_dicts(ds):
    """Combine dictionary values from multiple dictionaries and combine
    their keys if needed.
    Requires: import numpy as np
    Returns: a new dictionary
    """
    a = np.array([(k, v)                 # key, value pairs
                  for d in ds            # dict in dictionaries
                  for k, v in d.items()  # get the key, values from items
                  ])
    ks, idx = np.unique(a[:, 0], True)
    ks = ks[np.lexsort((ks, idx))]       # optional sort by appearance
    uniq = [np.unique(a[a[:, 0] == i][:, 1]) for i in ks]
    nd = [" ".join(u.tolist()) for u in uniq]
    new_d = dict(zip(ks, nd))
    return new_d


def find_dups(a_list):
    """Find dups in a list using an Ordered dictionary, return a list of
    duplicated elements
    """
    from collections import OrderedDict
    counter = OrderedDict()
    for item in a_list:
        if item in counter:
            counter[item] += 1
        else:
            counter[item] = 1
    return [item for item, counts in counter.items() if counts > 1]
# ---- (5) demos  -------------------------------------------------------------
#

# ----------------------------------------------------------------------------
# ---- __main__ .... code section --------------------------------------------
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
#    _demo()
