# -*- coding: utf-8 -*-
"""
==============
transect_lines
==============

Script : template.py

Author : Dan_Patterson@carleton.ca

Modified : 2019-02-23

Purpose :  Tools for producing transect lines

Notes:

References:

"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

import sys
import numpy as np
import arcpy

arcpy.overwriteOutputs = True

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

# ===========================================================================
# ---- def section: def code blocks go here ---------------------------------
#
def transect_lines(N=5, orig=None, dist=1, x_offset=0, y_offset=0,
                   bearing=0, as_ndarray=True):
    """Construct transect lines from origin-destination points given a
    distance and bearing from the origin point

    Parameters
    ----------
    N : number
        The number of transect lines
    orig : array-like
         A single origin.  If None, the cartesian origin (0, 0) is used
    dist : number or array-like
        The distance(s) from the origin
    x_offset, y_offset : number
        If the `orig` is a single location, you can construct offset lines
        using these values
    bearing : number or array-like
        If a single number, parallel lines are produced. An array of values
        equal to the `orig` can be used.

    Returns
    -------
    Two outputs are returned, the first depends on the `as_ndarray` setting.

    1. True, a structured array. False - a recarray
    2. An ndarray with the field names in case the raw data are required.

    Notes
    -----
    It is easiest of you pick a `corner`, then use x_offset, y_offset to
    control whether you are moving horizontally and vertically from the origin.
    The bottom left is easiest, and positive offsets move east and north from.

    Use XY to Line tool in ArcGIS Pro to convert the from/to pairs to a line.
    See references

    Examples
    --------
    >>> out, data = transect_lines(N=5, orig=None,
                                   dist=100, x_offset=10,
                                   y_offset=0, bearing=45, as_ndarray=True)
    >>> data
    array([[  0.  ,   0.  ,  70.71,  70.71],
           [ 10.  ,   0.  ,  80.71,  70.71],
           [ 20.  ,   0.  ,  90.71,  70.71],
           [ 30.  ,   0.  , 100.71,  70.71],
           [ 40.  ,   0.  , 110.71,  70.71]])
    >>> out
    array([( 0., 0.,  70.71, 70.71), (10., 0.,  80.71, 70.71),
    ...    (20., 0.,  90.71, 70.71), (30., 0., 100.71, 70.71),
    ...    (40., 0., 110.71, 70.71)],
    ...   dtype=[('X_from', '<f8'), ('Y_from', '<f8'),
    ...          ('X_to', '<f8'), ('Y_to', '<f8')])
    ...
    ... Create the table and the lines
    >>> tbl = 'c:/folder/your.gdb/table_name'
    >>> # arcpy.da.NumPyArrayToTable(a, tbl)
    >>> # arcpy.XYToLine_management(
    ... #       in_table, out_featureclass,
    ... #       startx_field, starty_field, endx_field, endy_field,
    ... #       {line_type}, {id_field}, {spatial_reference}
    ... This is general syntax, the first two are paths of source and output
    ... files, followed by coordinates and options parameters.
    ...
    ... To create compass lines
    >>> b = np.arange(0, 361, 22.5)
    >>> a, data = transect_lines(N=10, orig=[299000, 4999000],
                                 dist=100, x_offset=0, y_offset=0,
                                 bearing=b, as_ndarray=True)

    References
    ----------
    `<https://community.esri.com/blogs/dan_patterson/2019/01/17/transect-
    lines-parallel-lines-offset-lines>`_.

    `<http://pro.arcgis.com/en/pro-app/tool-reference/data-management
    /xy-to-line.htm>`_.
    """
    def _array_struct_(a, fld_names=['X', 'Y'], kinds=['<f8', '<f8']):
        """Convert an array to a structured array"""
        dts = list(zip(fld_names, kinds))
        z = np.zeros((a.shape[0],), dtype=dts)
        for i in range(a.shape[1]):
            z[fld_names[i]] = a[:, i]
        return z
    #
    if orig is None:
        orig = np.array([0., 0.])
    args = [orig, dist, bearing]
    arrs = [np.atleast_1d(i) for i in args]
    orig, dist, bearing = arrs
    # o_shp, d_shp, b_shp = [i.shape for i in arrs]
    #
    rads = np.deg2rad(bearing)
    dx = np.sin(rads) * dist
    dy = np.cos(rads) * dist
    #
    n = len(bearing)
    N = [N, n][n > 1]  # either the number of lines or bearings
    x_orig = np.arange(N) * x_offset + orig[0]
    y_orig = np.arange(N) * y_offset + orig[1]
    x_dest = x_orig + dx
    y_dest = y_orig + dy
    # ---- create the output array
    names = ['X_from', 'Y_from', 'X_to', 'Y_to']
    cols = len(names)
    kind = ['<f8']*cols
    data = np.vstack([x_orig, y_orig, x_dest, y_dest]).T
    if as_ndarray:  # **** add this as a flag
        out = _array_struct_(data, fld_names=names, kinds=kind)
    else:
        out = data.transpose()
        out = np.core.records.fromarrays(out, names=names, formats=kind)
    return out  #, data


# ===========================================================================
# ---- main section: testing or tool run ------------------------------------
# transect_lines(N=5, orig=None, dist=1, x_offset=0, y_offset=0,
#                   bearing=0, as_ndarray=True)
if len(sys.argv) == 1:
    create_output = True
    in_pth = script.split("/")[:-2] + ["Polygon_lineTools.gdb"]
    origin = "Bottom left"  # sys.argv[1]
    origin_x = 300100.      # sys.argv[2] orig = [origin_x, origin_y]
    origin_y = 5000000.     # sys.argv[3]
    N = 10                  # sys.argv[4] transects
    dist = 1000.            # sys.argv[5] line length
    x_offset = 50          # sys.argv[6] 
    y_offset = 0          # sys.argv[7]
    dens_dist = 100
    bearing = 0.0           # sys.argv[8]
    out_fc = "/".join(in_pth) + "/a0"  # sys.argv[9]
    SR = 2951
    #
    if origin == "Top left":
        y_offset = -y_offset
    orig = [origin_x, origin_y]
    arr = transect_lines(N, orig, dist, x_offset, y_offset, bearing, True)
else:
    create_output = True
    origin = sys.argv[1]
    origin_x = float(sys.argv[2])
    origin_y = float(sys.argv[3])
    N = int(sys.argv[4])
    dist = float(sys.argv[5])
    x_offset = float(sys.argv[6])
    y_offset = float(sys.argv[7])
    bearing = float(sys.argv[8])
    out_fc = sys.argv[9]
    SR = sys.argv[10]
    #
    if origin == "Top left":
        y_offset = -y_offset
    orig = [origin_x, origin_y]
    arr = transect_lines(N, orig, dist, x_offset, y_offset, bearing, True)

# ---- Create the table and the lines
if create_output:
    tbl = out_fc
    arcpy.da.NumPyArrayToTable(arr, tbl)
    out_fc = tbl + "_t"
    arcpy.XYToLine_management(tbl, out_fc, 'X_from', 'Y_from', 'X_to', 'Y_to',
                              spatial_reference=SR)
    #       startx_field, starty_field, endx_field, endy_field,
    #       {line_type}, {id_field}, {spatial_reference}

# ==== Processing finished ====
# ===========================================================================
#
if __name__ == "__main__":
    """optional location for parameters"""
    

