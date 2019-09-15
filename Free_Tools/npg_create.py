# -*- coding: utf-8 -*-
"""
==========
npg_create
==========

Script :
    npg_create.py

Author :
    Dan_Patterson@carleton.ca

Modified :
    2019-09-06

Purpose :
    Tools for creating arrays of various geometric shapes

Notes
-----
Originally part of the `arraytools` module.

References

"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

import sys
import numpy as np


ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=500, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

__all__ = ['code_grid', 'rot_matrix',
           'arc_', 'arc_sector',
           'circle', 'circle_mini', 'circle_ring',
           'ellipse',
           'hex_flat', 'hex_pointy',
           'mesh_xy',
           'pyramid',
           'rectangle',
           'triangle',
           'pnt_from_dist_bearing',
           'xy_grid',
           'transect_lines',
           'spiral_archim',
           'repeat', 'mini_weave'
           ]


def code_grid(cols=1, rows=1, zero_based=False, shaped=True, bottom_up=False):
    """produce spreadsheet like labelling, either zero or 1 based
    see: code_grid.py for more details
    """
    alph = list(" ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    UC = [("{}{}").format(alph[i], alph[j]).strip()
          for i in range(27)
          for j in range(1, 27)]
    z = [1, 0][zero_based]
    rc = [1, 0][zero_based]
    c = ["{}{:02.0f}".format(UC[c], r)  # pull in the column heading
         for r in range(z, rows + rc)   # label in the row letter
         for c in range(cols)]          # label in the row number
    c = np.asarray(c)
    if shaped:
        c = c.reshape(rows, cols)
        if bottom_up:
            c = np.flipud(c)
    return c


# ---- helpers ---- rot_matrix -----------------------------------------------
#
def rot_matrix(angle=0, nm_3=False):
    """Return the rotation matrix given points and rotation angle

    Parameters
    ----------
    Rotation angle in degrees and whether the matrix will be used with
    homogenous coordinates.

    Returns
    -------
    rot_m : matrix
        Rotation matrix for 2D transform.

    Rotate around...  translate(-x, -y).rotate(theta).translate(x, y).
    """
    rad = np.deg2rad(angle)
    c = np.cos(rad)
    s = np.sin(rad)
    rm = np.array([[c, -s, 0.],
                   [s, c, 0.],
                   [0., 0., 1.]])
    if not nm_3:
        rm = rm[:2, :2]
    return rm


# ---- arc_sector, convex hull, circle ellipse, hexagons, rectangles,
#      triangle, xy-grid --
#
def arc_(radius=100, start=0, stop=1, step=0.1, xc=0.0, yc=0.0):
    """Create an arc from a specified radius, centre and start/stop angles

    Parameters
    ----------
    radius : number
        cirle radius from which the arc is obtained
    start, stop, step : numbers
        angles in degrees
    xc, yc : number
        center coordinates in projected units
    as_list : boolean
        False, returns an array.  True yields a list

    Returns
    -------
      Points on the arc as an array

    >>> # arc from 0 to 90 in 5 degree increments with radius 2 at (0, 0)
    >>> a0 = arc_(radius=2, start=0, stop=90, step=5, xc=0.0, yc=0.0)
    """
    start, stop = sorted([start, stop])
    angle = np.deg2rad(np.arange(start, stop, step))
    x_s = radius*np.cos(angle)         # X values
    y_s = radius*np.sin(angle)         # Y values
    pnts = np.array([x_s, y_s]).T + [xc, yc]
    return pnts


def arc_sector(outer=10, inner=9, start=1, stop=6, step=0.1):
    """Form an arc sector bounded by a distance specified by two radii

    Parameters
    ----------
    outer : number
        outer radius of the arc sector
    inner : number
        inner radius
    start : number
        start angle of the arc
    stop : number
        end angle of the arc
    step : number
        the angle densification step

    Requires
    --------
      `arc_` is used to produce the arcs, the top arc is rotated clockwise and
      the bottom remains in the order produced to help form closed-polygons.
    """
    s_s = [start, stop]
    s_s.sort()
    start, stop = s_s
    top = arc_(outer, start, stop, step, 0.0, 0.0)
    top = top[::-1]
    bott = arc_(inner, start, stop, step, 0.0, 0.0)
    close = top[0]
    pnts = np.concatenate((top, bott, [close]), axis=0)
    return pnts


def circle(radius=100, clockwise=True, theta=1, rot=0.0, scale=1,
           xc=0.0, yc=0.0):
    """Produce a circle/ellipse depending on parameters.

    Parameters
    ----------
    radius : number
        In projected units.
    clockwise : boolean
        True for clockwise (outer rings), False for counter-clockwise
        (for inner rings).
    theta : number
        Angle spacing. If theta=1, angles between -180 to 180, are returned
        in 1 degree increments. The endpoint is excluded.
    rot : number
         Rotation angle in degrees... used if scaling is not equal to 1.
    scale : number
         For ellipses, change the scale to <1 or > 1. The resultant
         y-values will favour the x or y-axis depending on the scaling.

    Returns
    -------
    List of coordinates for the circle/ellipse

    Notes
    -----
    You can also use np.linspace if you want to specify point numbers.
    np.linspace(start, stop, num=50, endpoint=True, retstep=False)
    np.linspace(-180, 180, num=720, endpoint=True, retstep=False)
    """
    if clockwise:
        angles = np.deg2rad(np.arange(180.0, -180.0-theta, step=-theta))
    else:
        angles = np.deg2rad(np.arange(-180.0, 180.0+theta, step=theta))
    x_s = radius*np.cos(angles)            # X values
    y_s = radius*np.sin(angles) * scale    # Y values
    pnts = np.array([x_s, y_s]).T
    if rot != 0:
        rot_mat = rot_matrix(angle=rot)
        pnts = (np.dot(rot_mat, pnts.T)).T
    pnts = pnts + [xc, yc]
    return pnts


def circle_mini(radius=1.0, theta=10.0, xc=0.0, yc=0.0):
    """Produce a circle/ellipse depending on parameters.

    Parameters
    ----------
    radius : number
        Distance from centre
    theta : number
        Angle of densification of the shape around 360 degrees

    """
    angles = np.deg2rad(np.arange(180.0, -180.0-theta, step=-theta))
    x_s = radius*np.cos(angles) + xc    # X values
    y_s = radius*np.sin(angles) + yc    # Y values
    pnts = np.array([x_s, y_s]).T
    return pnts


def circle_ring(outer=100, inner=0, theta=10, rot=0, scale=1, xc=0.0, yc=0.0):
    """Create a multi-ring buffer around a center point (xc, yc).  AKA, a
    buffer ring.

    Parameters
    ----------
    outer : number
        Outer radius.
    inner : number
        Inner radius.
    theta : number
        See below.
    rot : number
        Rotation angle, used for non-circles.
    scale : number
        Used to scale the y-coordinates.

    Notes
    -----
    Angles to use to densify the circle::

    - 360+ circle
    - 120  triangle
    - 90   square
    - 72   pentagon
    - 60   hexagon
    - 45   octagon
    - etc
    """
    top = circle(outer, clockwise=True, theta=theta, rot=rot, scale=scale,
                 xc=xc, yc=yc)
    if inner == 0.0:
        return top
    bott = circle(inner, clockwise=False, theta=theta, rot=rot, scale=scale,
                  xc=xc, yc=yc)
    return np.concatenate((top, bott), axis=0)


def circ_3pa(arr):
    """same as circ3p but with 3 pnt arr"""
    p, q, r = arr
    cx, cy, radius = circ_3p(p, q, r)
    return cx, cy, radius


def circ_3p(p, q, r):
    """Three point circle center and radius.  A check is made for three points
    on a line.
    """
    temp = q[0] * q[0] + q[1] * q[1]
    bc = (p[0] * p[0] + p[1] * p[1] - temp) / 2
    cd = (temp - r[0] * r[0] - r[1] * r[1]) / 2
    # three points on a line check
    det = (p[0] - q[0]) * (q[1] - r[1]) - (q[0] - r[0]) * (p[1] - q[1])
    if abs(det) < 1.0e-6:
        return None, None, np.inf
    # Center of circle
    cx = (bc*(q[1] - r[1]) - cd*(p[1] - q[1])) / det
    cy = ((p[0] - q[0]) * cd - (q[0] - r[0]) * bc) / det
    radius = np.sqrt((cx - p[0])**2 + (cy - p[1])**2)
    return cx, cy, radius


def ellipse(x_radius=1.0, y_radius=1.0, theta=10., xc=0.0, yc=0.0):
    """Produce an ellipse depending on parameters.

    Parameters
    ----------
    radius : number
        Distance from centre in the X and Y directions.
    theta : number
        Angle of densification of the shape around 360 degrees.
    """
    angles = np.deg2rad(np.arange(180.0, -180.0-theta, step=-theta))
    x_s = x_radius*np.cos(angles) + xc    # X values
    y_s = y_radius*np.sin(angles) + yc    # Y values
    pnts = np.concatenate((x_s, y_s), axis=0)
    return pnts


def hex_flat(dx=1, dy=1, cols=1, rows=1):
    """Generate the points for the flat-headed hexagon.

    Parameters
    ----------
    dy_dx : number
        The radius width, remember this when setting hex spacing.
    dx : number
        Increment in x direction, +ve moves west to east, left/right
    dy : number
        Increment in y direction, -ve moves north to south, top/bottom
    """
    f_rad = np.deg2rad([180., 120., 60., 0., -60., -120., -180.])
    X = np.cos(f_rad) * dy
    Y = np.sin(f_rad) * dy            # scaled hexagon about 0, 0
    seed = np.array(list(zip(X, Y)))  # array of coordinates
    dx = dx * 1.5
    dy = dy * np.sqrt(3.)/2.0
    hexs = [seed + [dx * i, dy * (i % 2)] for i in range(0, cols)]
    m = len(hexs)
    for j in range(1, rows):  # create the other rows
        hexs += [hexs[h] + [0, dy * 2 * j] for h in range(m)]
    return hexs


def hex_pointy(dx=1, dy=1, cols=1, rows=1):
    """Pointy hex angles, convert to sin, cos, zip and send.  Also called
    ``traverse hexagons`` by some.

    Parameters
    ----------
    dy_dx - number
        The radius width, remember this when setting hex spacing.
    dx : number
        Increment in x direction, +ve moves west to east, left/right.
    dy : number
        Increment in y direction, -ve moves north to south, top/bottom.
    """
    p_rad = np.deg2rad([150., 90, 30., -30., -90., -150., 150.])
    X = np.cos(p_rad) * dx
    Y = np.sin(p_rad) * dy      # scaled hexagon about 0, 0
    seed = np.array(list(zip(X, Y)))
    dx = dx * np.sqrt(3.)/2.0
    dy = dy * 1.5
    hexs = [seed + [dx * i * 2, 0] for i in range(0, cols)]
    m = len(hexs)
    for j in range(1, rows):  # create the other rows
        hexs += [hexs[h] + [dx * (j % 2), dy * j] for h in range(m)]
    return hexs


def mesh_xy(L=0, B=0, R=5, T=5, dx=1, dy=1, as_rec=True):
    """Create a mesh of coordinates within the specified X, Y ranges

    Parameters
    ----------
    L(eft), R(ight), dx : number
        Coordinate min, max and delta x for X axis.
    B(ott), T(op), dy  : number
        Same as above for Y axis.
    as_rec : boolean
        Produce a structured array (or convert to a record array).

    Returns
    -------
    -  A list of coordinates of X,Y pairs and an ID if as_rec is True.
    -  A mesh grid X and Y coordinates is also produced.
    """
    dt = [('Pnt_num', '<i4'), ('X', '<f8'), ('Y', '<f8')]
    x = np.arange(L, R + dx, dx, dtype='float64')
    y = np.arange(B, T + dy, dy, dtype='float64')
    mesh = np.meshgrid(x, y, sparse=False)
    if as_rec:
        xs = mesh[0].ravel()
        ys = mesh[1].ravel()
        p = list(zip(np.arange(len(xs)), xs, ys))
        pnts = np.array(p, dtype=dt)
    else:
        p = list(zip(mesh[0].ravel(), mesh[1].ravel()))
        pnts = np.array(p)
    return pnts, mesh


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


def rectangle(dx=1, dy=1, cols=1, rows=1):
    """Create the array of pnts to construct a rectangle.

    Parameters
    ----------
    dx : number
        Increment in x direction, +ve moves west to east, left/right.
    dy : number
        Increment in y direction, -ve moves north to south, top/bottom.
    rows, cols : ints
        The number of rows and columns to produce.
    """
    X = [0.0, 0.0, dx, dx, 0.0]       # X, Y values for a unit square
    Y = [0.0, dy, dy, 0.0, 0.0]
    seed = np.array(list(zip(X, Y)))  # [dx0, dy0] keep for insets
    a = [seed + [j * dx, i * dy]      # make the shapes
         for i in range(0, rows)      # cycle through the rows
         for j in range(0, cols)]     # cycle through the columns
    a = np.asarray(a)
    return a


def triangle(dx=1, dy=1, cols=1, rows=1):
    """Create a row of meshed triangles.

    Parameters
    ----------
    dx : number
        Increment in x direction, +ve moves west to east, left/right.
    dy : number
        Increment in y direction, -ve moves north to south, top/bottom.
    rows, cols : ints
        The number of rows and columns to produce.
    """
    grid_type = 'triangle'
    a, dx, b = dx/2.0, dx, dx*1.5
    Xu = [0.0, a, dx, 0.0]   # X, Y values for a unit triangle, point up
    Yu = [0.0, dy, 0.0, 0.0]
    Xd = [a, b, dx, a]       # X, Y values for a unit triangle, point down
    Yd = [dy, dy, 0.0, dy]   # shifted by dx
    seedU = np.vstack((Xu, Yu)).T  # np.array(list(zip(Xu, Yu)))
    seedD = np.vstack((Xd, Yd)).T  # np.array(list(zip(Xd, Yd)))
    seed = np.array([seedU, seedD])
    a = [seed + [j * dx, i * dy]       # make the shapes
         for i in range(0, rows)       # cycle through the rows
         for j in range(0, cols)]      # cycle through the columns
    a = np.asarray(a)
    s1, s2, s3, s4 = a.shape
    a = a.reshape(s1*s2, s3, s4)
    return a, grid_type


def pnt_from_dist_bearing(orig=(0, 0), bearings=None, dists=None, prn=False):
    """Point locations given distance and bearing from an origin.
    Calculate the point coordinates from distance and angle.

    References
    ----------
    `<https://community.esri.com/thread/66222>`_.

    `<https://community.esri.com/blogs/dan_patterson/2018/01/21/
    origin-distances-and-bearings-geometry-wanderings>`_.

    Notes
    -----
    Planar coordinates are assumed.  Use Vincenty if you wish to work with
    geographic coordinates.

    Sample calculation::

        bearings = np.arange(0, 361, 22.5)  # 17 bearings
        dists = np.random.randint(10, 500, len(bearings)) * 1.0  OR
        dists = np.full(bearings.shape, 100.)
        data = dist_bearing(orig=orig, bearings=bearings, dists=dists)

    Create a featureclass from the results::

        shapeXY = ['X_to', 'Y_to']
        fc_name = 'C:/path/Geodatabase.gdb/featureclassname'
        arcpy.da.NumPyArrayToFeatureClass(
            out, fc_name, ['X_to', 'Y_to'], "2951")
        # ... syntax
        arcpy.da.NumPyArrayToFeatureClass(
            in_array=out, out_table=fc_name, shape_fields=shapeXY,
            spatial_reference=SR)
    """
    error = "An origin with distances and bearings of equal size are required."
    orig = np.array(orig)
    if bearings is None or dists is None:
        raise ValueError(error)
    iterable = np.all([isinstance(i, (list, tuple, np.ndarray))
                       for i in [dists, bearings]])
    if iterable:
        if not (len(dists) == len(bearings)):
            raise ValueError(error)
    else:
        raise ValueError(error)
    rads = np.deg2rad(bearings)
    dx = np.sin(rads) * dists
    dy = np.cos(rads) * dists
    x_t = np.cumsum(dx) + orig[0]
    y_t = np.cumsum(dy) + orig[1]
    stack = (x_t, y_t, dx, dy, dists, bearings)
    names = ["X_to", "Y_to", "orig_dx", "orig_dy", "distance", "bearing"]
    data = np.vstack(stack).T
    N = len(names)
    if prn:  # ---- just print the results ----------------------------------
        frmt = "Origin ({}, {})\n".format(*orig) + "{:>10s}"*N
        print(frmt.format(*names))
        frmt = "{: 10.2f}"*N
        for i in data:
            print(frmt.format(*i))
        return data
    # ---- produce a structured array from the output -----------------------
    names = ", ".join(names)
    kind = ["<f8"]*N
    kind = ", ".join(kind)
    out = data.transpose()
    out = np.core.records.fromarrays(out, names=names, formats=kind)
    return out


def xy_grid(x, y=None, top_left=True):
    """Create a 2D array of locations from x, y values.  The values need not
    be uniformly spaced just sequential. Derived from `meshgrid` in References.

    Parameters
    ----------
    x, y : array-like
        To form a mesh, there must at least be 2 values in each sequence
    top_left: boolean
        True, y's are sorted in descending order, x's in ascending

    References
    ----------
    `<https://github.com/numpy/numpy/blob/master/numpy/lib/function_base.py>`_.
    """
    if y is None:
        y = x
    if x.ndim != 1:
        return "A 1D array required"
    xs = np.sort(np.asanyarray(x))
    ys = np.asanyarray(y)
    if top_left:
        ys = np.argsort(-ys)
    xs = np.reshape(xs, newshape=((1,) + xs.shape))
    ys = np.reshape(ys, newshape=(ys.shape + (1,)))
    xy = [xs, ys]
    xy = np.broadcast_arrays(*xy, subok=True)
    shp = np.prod(xy[0].shape)
    final = np.zeros((shp, 2), dtype=xs.dtype)
    final[:, 0] = xy[0].ravel()
    final[:, 1] = xy[1].ravel()
    return final


def transect_lines(N=5, orig=None, dist=1, x_offset=0, y_offset=0,
                   bearing=0, as_ndarray=True):
    """Construct transect lines from origin-destination points given a
    distance and bearing from the origin point.

    Parameters
    ----------
    N : number
        The number of transect lines.
    orig : array-like
         A single origin.  If None, the cartesian origin (0, 0) is used.
    dist : number or array-like
        The distance(s) from the origin
    x_offset, y_offset : number
        If the `orig` is a single location, you can construct offset lines
        using these values.
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
    See references.

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
    return out, data


def spiral_archim(pnts, n, inward=False, clockwise=True):
    """Create an Archimedes spiral in the range 0 to N points with 'n' steps
    between each incrementstep.  You could use np.linspace.

    Parameters
    ----------
    N : integer
        The range of the spiral.
    n : integer
        The number of points between steps.
    scale : number
        The size between points.
    outward : boolean
        Radiate the spiral from the center.

    Notes
    -----
    When n is small relative to N, then you begin to form rectangular
    spirals, like rotated rectangles.

    With N = 360, n = 20 yields 360 points with 2n points (40) to complete each
    360 degree loop of the spiral.
    """
    rnge = np.arange(0.0, pnts)
    if inward:
        rnge = rnge[::-1]
    phi = rnge/n * np.pi
    xs = phi * np.cos(phi)
    ys = phi * np.sin(phi)
    if clockwise:
        xy = np.c_[ys, xs]
    else:
        xy = np.c_[xs, ys]
    # wdth, hght = np.ptp(xy, axis=0)
    return xs, ys, xy


def repeat(seed=None, corner=[0, 0], cols=1, rows=1, angle=0):
    """Create the array of pnts to pass on to arcpy using numpy magic to
    produce a fishnet of the desired in_shp.

    Parameters
    ----------
    seed : use grid_array, hex_flat or hex_pointy.
        You specify the width and height or its ratio when making the shapes.
    corner : array-like
        Lower left corner of the shape pattern.
    dx, dy : numbers
        Offset of the shapes... this is different.
    rows, cols : integers
        The number of rows and columns to produce.
    angle : number
        Rotation angle in degrees.
    """
    def rotate(pnts, angle=0):
        """Rotate points about the origin in degrees, (+ve for clockwise)."""
        angle = np.deg2rad(angle)                 # convert to radians
        s = np.sin(angle)
        c = np.cos(angle)    # rotation terms
        aff_matrix = np.array([[c, s], [-s, c]])  # rotation matrix
        XY_r = np.dot(pnts, aff_matrix)           # numpy magic to rotate pnts
        return XY_r
    # ----
    if seed is None:
        a = rectangle(dx=1, dy=1, cols=3, rows=3)
    else:
        a = np.asarray(seed)
    if angle != 0:
        a = [rotate(p, angle) for p in a]      # rotate the scaled points
    pnts = [p + corner for p in a]            # translate them
    return pnts


def mini_weave(n):
    """
    n : segments
       z is sliced to ensure compliance

    >>> a = mini_weave(11)
    >>> e_leng(a)
    | total 14.142135623730953,
    | segment [1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41]
    """
    # root2 = np.full(n, np.sqrt(2.))
    one_s = np.ones(n)
    zero_s = np.zeros(n)
#    x = np.arange(n)
#    y = np.arange(n)
#    z = np.asarray([*sum(zip(zero_s, root2), ())])
    x = np.arange(n)
    y = np.zeros(n)
    z = np.asarray([*sum(zip(zero_s, one_s), ())])[:n]
    a = np.vstack((x, y, z)).T
    return a


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    print("Script path {}".format(script))
