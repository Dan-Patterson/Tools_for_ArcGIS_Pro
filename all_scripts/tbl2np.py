# -*- coding: utf-8 -*-
"""
tbl2np
======

Script :   tbl2np.py

Author :   Dan_Patterson@carleton.ca

Modified : 2018-09-24

Purpose:  tools for working with numpy arrays

Useage :

References
----------
`<http://pro.arcgis.com/en/pro-app/arcpy/data-access/numpyarraytotable.htm>`_.
`<http://pro.arcgis.com/en/pro-app/arcpy/data-access/tabletonumpyarray.htm>`_.
---------------------------------------------------------------------
"""

# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

import sys
from textwrap import dedent
import numpy as np
from art_common import (tweet, de_punc, _describe, fc_info, fld_info,
                        null_dict, tbl_arr)
from arcpy.da import Describe

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=5, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

# ----------------------------------------------------------------------
# .... final code section producing the featureclass and extendtable

if len(sys.argv) == 1:
    testing = True
    pth = script.split("/")[:-2]
    pth = "/".join(pth) + "/array_tools.gdb/pnts_2000"
    a = tbl_arr(pth)
    frmt = "Result...\n{}"
    print(frmt.format(a))
else:
    testing = False
    in_tbl = sys.argv[1]
    desc = Describe(in_tbl)
    pth = desc['catalogPath']
#    out_folder = sys.argv[2]
#    out_name = sys.argv[3]
    out_arr = sys.argv[2]  # + "/" + out_name
    # ---- call section for processing function
    #
    a = tbl_arr(pth)
    np.save(out_arr, a)
    args = [a, out_arr]
    msg = """
    :------------------------------------------------------------

    Input table... {}
    Output array.... {}

    Conversion complete...
    You can reload the array using np.load(drive:/path/name.py)

    :------------------------------------------------------------
    """
    msg = dedent(msg).format(*args)
    tweet(msg)
if testing:
    print('\nScript source... {}'.format(script))
# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
