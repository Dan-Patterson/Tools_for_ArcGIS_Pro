# -*- coding: utf-8 -*-
"""
npy2ras
=======

Script :   npy2ras.py

Author :   Dan_Patterson@carleton.ca

Modified : 2018-09-26

Purpose :  tools for working with numpy arrays

References
----------
`<http://pro.arcgis.com/en/pro-app/arcpy/functions/numpyarraytoraster
-function.htm>`_.
`<http://pro.arcgis.com/en/pro-app/arcpy/data-access/numpyarraytotable.htm>`_.
`<http://pro.arcgis.com/en/pro-app/arcpy/data-access/tabletonumpyarray.htm>`_.
---------------------------------------------------------------------
"""

import sys
import numpy as np
from arcpy import Point
from arcgisscripting import NumPyArrayToRaster
from arcpy import env

env.overwriteOutput = True

#from arcpytools import fc_info, tweet  #, frmt_rec, _col_format

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

# ----------------------------------------------------------------------
# .... final code section producing the featureclass and extendtable
if len(sys.argv) == 1:
    testing = True
    pth = script.split("/")[:-1]
    pth0 = "/".join(pth) + "/Data/r00.npy"
    LL_x = 300000.
    LL_y = 5030000.
    cell_sze =10.
    no_data = 255
    in_arr = np.load(pth0)
    pth1 = "/".join(pth) + "/Data/r01.tif"
    # parameters here
else:
    testing = False
    pth0 = sys.argv[1]
    LL_x = float(sys.argv[2])
    LL_y = float(sys.argv[3])
    cell_sze = float(sys.argv[4])
    pth1 = sys.argv[5]
    if pth1[-4:] != ".tif":
        pth1 += ".tif"
    in_arr = np.load(pth0)
    # parameters here
#
to_pro = True  # ---- change to True to produce tif for ArcGIS PRO

dt_kind = in_arr.dtype.kind
if dt_kind in ('u', 'i'):
    no_data = np.iinfo(in_arr.dtype.str).max
elif dt_kind in ('f'):
    no_data = np.iinfo(in_arr.dtype.str).max
else:
    no_data = None
if to_pro:
    ras = NumPyArrayToRaster(in_arr,
                             lower_left_corner=Point(LL_x, LL_y),
                             x_cell_size=cell_sze,
                             value_to_nodata=no_data
                             )
    ras.save(pth1)

if testing:
    print('\nScript source... {}'.format(script))
# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
