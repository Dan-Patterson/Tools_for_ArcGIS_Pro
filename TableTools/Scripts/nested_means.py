# -*- coding: utf-8 -*-
"""
script name here
=======

Script :   template.py

Author :   Dan_Patterson@carleton.ca

Modified : 2019-

Purpose :  Tools for

Notes:

References:

"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

import sys
import os
import numpy as np
import arcpy

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=100, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

#script = sys.argv[0]  # print this should you need to locate the script

# ===========================================================================
# ---- def section: def code blocks go here ---------------------------------
def tweet(msg):
    """Print a message for both arcpy and python.
    : msg - a text message
    """
    m = "\n{}\n".format(msg)
    arcpy.AddMessage(m)
    print(m)


def mean_split(a, minSize=3, cmax=9):
    """split at means"""
    def slice_array(a):
        m = np.mean(a)
        yes = a <= m  # check for a less than the overal mean
        a_left, a_rght = a[~yes], a[yes]  # slice the arrays
        return m, a_left, a_rght
    # ----
    m, L, R = slice_array(a)
    m0, L, _ = slice_array(L)
    m1, _, R = slice_array(R)
    means = [m0, m, m1]
    while ((len(L) > minSize) and (len(R) > minSize) and (len(means) <= cmax)):
        m0, L, _ = slice_array(L)
        m1, _, R = slice_array(R)
        means.extend([m0, m1])   
    return sorted(means)

# ===========================================================================
# ---- main section: testing or tool run ------------------------------------
#
def _common_():
    """Stuff common to _demo_ and _tool()
    """
    script = sys.argv[0]
    return script

def _demo_():
    """Run in spyder
    """
    testing = True
    script = _common_()
    msg0 = "\nRunning... {} in Spyder\n".format(script)
    tbl = r"C:\Arc_projects\Table_tools\Table_tools.gdb\pnts_2K_normal"
    in_fld = "Norm"
    out_fld = "New_class"
    cut_off = 9
    return msg0, testing, tbl, in_fld, out_fld, cut_off

def _tool_():
    """run from a tool in arctoolbox in arcgis pro
    """
    testing = False
    script = _common_()
    tbl = sys.argv[1]
    in_fld = sys.argv[2]
    out_fld = sys.argv[3]
    cut_off = int(sys.argv[4])
    msg0 = "\nRunning... {} in in ArcGIS Pro\n".format(script)
    return msg0, testing, tbl, in_fld, out_fld, cut_off
# ===========================================================================
# ---- main section: testing or tool run ------------------------------------
#
if len(sys.argv) == 1:
    msg, testing, tbl, in_fld, out_fld, cut_off = _demo_()
else:
    msg, testing, tbl, in_fld, out_fld, cut_off = _tool_()

print(msg)

# ---- Do some work
# (1)  Get the field from the table and make it a simple array
arr = arcpy.da.TableToNumPyArray(tbl, ['OID@', in_fld], "", True, None)
a = arr[in_fld]

# (2)  Set up for the results
out_arr = np.copy(arr)
out_fld = arcpy.ValidateFieldName(out_fld, os.path.dirname(tbl))
out_arr.dtype.names = ['OID@', out_fld]

# (3)  Run the mean_split script... note minSize!!! set it smartly or...
means = mean_split(a, minSize=3, cmax=cut_off)
means = sorted(means)
classed = np.digitize(a, bins=means)

# (4)  Send the results to the output array and add it back to arcgis pro
#
out_arr[out_fld] = classed
#arcpy.da.ExtendTable(tbl, 'OID@', out_arr, 'OID@')

frmt = """
means  : {}
counts : {}
"""
c, m= np.histogram(a, means)
tweet(frmt.format(m, c))
# ==== Processing finished ====
# ===========================================================================
#
if __name__ == "__main__":
    """optional location for parameters"""
    msg = _demo_()

