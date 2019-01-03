# -*- coding: UTF-8 -*-
"""
surface
=======

Script:   surface.py

Author:   Dan.Patterson@carleton.ca

Modified: 2018-04-27

Purpose :
    Calculate slope, aspect, hillshade and other terrain derivatives
    Plus basic array ramblings so I don't forget.

---------------------------
1. Truth test and slicing:
---------------------------

Example:

>>> a = np.arange(3*3*3).reshape(3,3,3)
>>> art.f_(a)
Array... shape (3, 3, 3), ndim 3, not masked
  0,  1,  2     9, 10, 11    18, 19, 20
  3,  4,  5    12, 13, 14    21, 22, 23
  6,  7,  8    15, 16, 17    24, 25, 26
 sub (0)       sub (1)       sub (2)

>>> np.all((a[...,2], a[...,:,2], a[:,:,2])) # => True
a[:,:,2]
array([[ 2,  5,  8],
       [11, 14, 17],
       [20, 23, 26]])


2. Filters:
----------
- f_dxyz : as implemented in arcmap after Burrough

  np.array([[1,2,1], [2,0,2], [1,2,1]], dtype="float64")

- f_plus   - maximum rise/fall after eppl7

  np.array([[0,1,0], [1,0,1], [0,1,0]], dtype="float64")

-  f_cross  -

   np.array([[1,0,1], [0,0,0], [1,0,1]], dtype="float64")

-   f_d8   t = np.sqrt(2.0)  f_plus + t*cross used for distance

   np.array([[t,1,t], [1,0,1], [t,1,t]], dtype="float64")


3. Slope calculation:
---------------------
- 3rd order finite distance Horn (1981) see Burrough
>>> filter = [[a, b, c], [d, e, f], [g, h, i]]
    [dz/dx] = ((c + 2f + i) - (a + 2d + g)) / (8 * x_cellsize)
    [dz/dy] = ((g + 2h + i) - (a + 2b + c)) / (8 * y_cellsize)
    rise_run = sqrt([dz/dx]2 + [dz/dy]2)
             = sqrt((0.05)2 + (-3.8)2)
             = sqrt(0.0025 + 14.44) = 3.80032
    slope_radians = ATAN(sqrt([dz/dx]2 + [dz/dy]2) )

::

    [dz/dx] = ((c + 2f + i) - (a + 2d + g) / (8 * x_cellsize)
            = ((50 + 60 + 10) - (50 + 60 + 8)) / (8 * 5)
            = (120 - 118) / 40 = 0.05

    [dz/dy] = ((g + 2h + i) - (a + 2b + c)) / (8 * y_cellsize)
            = ((8 + 20 + 10) - (50 + 90 + 50)) / (8 * 5)
            = (38 - 190 ) / 40 = -3.8

    rise_run = sqrt(([dz/dx]2 + [dz/dy]2)
             = sqrt((0.05)2 + (-3.8)2)
             = sqrt(0.0025 + 14.44) = 3.80032

    slope_degrees = ATAN(rise_run) * 57.29578
                  = ATAN(3.80032) * 57.29578
                  = 1.31349 * 57.29578 = 75.25762

Or via code:

    >>> a= np.array([[50,45,50],[30,30,30],[8, 10, 10]]).reshape(1,1,3,3)
    >>> f_dxyz = np.array([[1, 2, 1], [2, 0, 2], [1, 2, 1]], dtype="float64")
    >>> cell_size = 5. * 8
    >>> a_s = a * f_dxyz
    >>> dz_dx = ((a_s[...,:,2] - a_s[...,:,0]).sum(axis=2)).squeeze()/cell_size
    >>> dz_dy = ((a_s[...,2,:] - a_s[...,0,:]).sum(axis=2)).squeeze()/cell_size
    dz_dx  # =>  0.050000000000000003
    dz_dy  # => -3.7999999999999998
    # after finishing all the math... = 75.25762

Note:
    These are all the same... since we are working with a 4d array

The same applies for dz_dy

- dz_dx = ((a_s[...,:,2] - a_s[...,:,0]).sum(axis=-1))/cell_size
- dz_dx = ((a_s[...,:,2] - a_s[...,:,0]).sum(axis=2)).squeeze()/cell_size
- dz_dx = ((a_s[...,:,2] - a_s[...,:,0]).flatten()).sum()/cell_size


4. Aspect calculation:
----------------------

- [dz/dx] = ((c + 2f + i) - (a + 2d + g)) / 8

- [dz/dy] = ((g + 2h + i) - (a + 2b + c)) / 8

- aspect = 57.29578 * atan2 ([dz/dy], -[dz/dx])

-  aspect rule:  **** general way, but not the most efficient ****

>>> if aspect < 0:       cell = 90.0 - aspect
    elif aspect > 90.0:  cell = 360.0 - aspect + 90.0
    else:                cell = 90.0 - aspect
    return cell

or ...

>>> azimuth = np.mod((450.0 - 45), 360)      # => 45  **** best one ****

or ...

::

   divmod((450.0 - 45), 360)[1]   => (1.0, 45.0)
   [dz/dx] = ((c + 2f + i) - (a + 2d + g)) / 8
           = ((85 + 170 + 84)) - (101 + 202 + 101)) / 8  = -8.125

   [dz/dy] = ((g + 2h + i) - (a + 2b + c)) / 8
           = ((101 + 182 + 84) - (101 + 184 + 85)) / 8   = -0.375

   aspect = 57.29578 * atan2 ([dz/dy], -[dz/dx])
          = 57.29578 * atan2 (-0.375, 8.125)    = -2.64:

   a= np.array([[50,45,50],[30,30,30],[8, 10, 10]]).reshape(1,1,3,3)


5. Hillshade:
-------------

::

  Hillshade = 255.0 * ((cos(Zenith_rad) * cos(Slope_rad)) +
                 (sin(Zenith_rad) * sin(Slope_rad) *
                  cos(Azimuth_rad - Aspect_rad)))
  Zenith_deg = 90 - Altitude
  Convert to radians
  Zenith_rad = Zenith * pi / 180.0

  Azimuth_math = 360.0 - Azimuth + 90       # normally they are doing this
  Note that if Azimuth_math >= 360.0, then: # to rotate back to the x-axis
  Azimuth_math = Azimuth_math - 360.0       # I skip this and use North

  Convert to radians
  Azimuth_rad = Azimuth_math * pi / 180.0


Computing slope and aspect

6. Curvature:
-------------

::

   Z1  Z2  Z3 - L cell width, L2=L^2, L3=L^3, L4=L^4
   Z4  Z5  Z6
   Z7  Z8  z9
   from: http://desktop.arcgis.com/en/arcmap/latest/tools/
               spatial-analyst-toolbox/how-curvature-works.htm
   For each cell, a fourth-order polynomial of the form:
   Z = Ax²y² + Bx²y + Cxy² + Dx² + Ey² + Fxy + Gx + Hy + I
   is fit to a surface composed of a 3x3 window. The coefficients
   a, b, c, and so on, are calculated from this surface.

::

   A = [(Z1 + Z3 + Z7 + Z9)/4  - (Z2 + Z4 + Z6 + Z8)/2 + Z5]/ L4
   B = [(Z1 + Z3 - Z7 - Z9)/4 - (Z2 - Z8)/2] / L3
   C = [(-Z1 + Z3 - Z7 + Z9)/4 + (Z4 - Z6)]/2] / L3
   D = [(Z4 + Z6)/2 - Z5]/L2
   E = [(Z2 + Z8)/2 - Z5]/L2
   F = (-Z1 + Z3 + Z7 - Z9) / 4L2
   G = (-Z4 + Z6)/2L
   H = (Z2 - Z8)/2L
   I = Z5
   slope = sqrt(G^2 + H^2)  aspect = arctan(-H/-G)
   profile curv.  =  2(DG^2 + EH^2 + FGH)/(G^2 + H^2)
   planform curv. = -2(DH^2 + EG^2 - FGH)/(G^2 + H^2)
   curvature... slope of the slope
   Curvature = -2(D + E) * 100
   or...  (100/L2)*(3*Z5 - [Z2+Z4+Z6+Z8+Z5])
   Bill's shortcut
   in comments http://gis.stackexchange.com/questions/37066/
                     how-to-calculate-terrain-curvature


6. Axis angles conversion:
--------------------------

::

  axis angle and azimuth relationships..
  an_az = np.array([[180, 270], [135, 315], [90, 0], [45, 45], [0, 90],
                    [-45, 135], [-90, 180], [-135, 225],[-180, 270]])
  an = an_az[:,0] => [180, 135, 90, 45, 0, -45, -90, -135, -180])
  az = an_az[:,1] => [270, 315, 0, 45, 90, 135, 180, 225, 270]

  azimuth np.mod((450.0 - an), 360)   **** best one ****


7. single_demo:
---------------

-  pre-made orientations at 45 deg.  Example with a dx/dy of 2

::

     dx=2 - slope  45.0 asp:   0.0 hshade: 204.0
     dx=2 - slope: 35.3 asp:  45.0 hshade: 137.0
     dx=2 - slope: 45.0 asp:  90.0 hshade:   0.0
     dx=2 - slope: 35.3 asp: 135.0 hshade:  19.0
     dx=2 - slope: 45.0 asp: 180.0 hshade:   0.0
     dx=2 - slope: 35.3 asp: 225.0 hshade: 137.0
     dx=2 - slope: 45.0 asp: 270.0 hshade: 204.0
     dx=2 - slope: 35.3 asp: 315.0 hshade: 254.0


interweave arrays:
------------------

Various examples :

list comprehensions, column_stack, row_stack, r_, c_, vstack, hstack

::

  a = np.arange(5)
  b = np.arange(5,0,-1)
  (1)... [val for pair in zip(a, b) for val in pair] ....
     =>   [0, 5, 1, 4, 2, 3, 3, 2, 4, 1]
  (2)... np.ravel(np.column_stack((a, b))) ....
     ... np.r_[a, b]  # note square brackets
     =>   array([0, 5, 1, 4, 2, 3, 3, 2, 4, 1])
  (3)... np.column_stack((a, b))
     ... np.c_[a, b]  # note square brackets
     =>  array([[0, 5],
                [1, 4],
                [2, 3],
                [3, 2],
                [4, 1]])
  (4)... np.row_stack((a, b))
     ... np.vstack((a, b))
     =>  array([[0, 1, 2, 3, 4],
                [5, 4, 3, 2, 1]])
  (5)... np.hstack((a,b))
         array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1])
  (6)... various inter-weave options
   years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']
   => [z[0] +"-"+ z[1] for z in zip(years[:-1], years[1:])]  # or ****
   => [z[0] + "-" + z[1] for z in np.c_[years[:-1], years[1:]]]  # or
   => ["{}-{}".format(*z) for z in np.c_[years[:-1], years[1:]]]
   => ['2010-2011', '2011-2012', '2012-2013', '2013-2014', '2014-2015',
       '2015-2016', '2016-2017']


Masked array:
-------------

::

  'nan', 'nan_to_num', 'nanargmax', 'nanargmin', 'nanmax', 'nanmean',
  'nanmin', 'nanstd', 'nansum', 'nanvar'
  a_sf = np.array(a_s)*no_cnt
  m = np.where(a_sf==-99, -99,0)
  a_sm = np.ma.array(a_sf, mask = m, fill_value=mask_val)


References:
----------

`<https://pro.arcgis.com/en/pro-app/tool-reference/spatial-analyst/\
how-hillshade-works.htm>`_

`<http://pangea.stanford.edu/~samuelj/musings/dems-in-python-pt-3-slope-/
and-hillshades-.html>`_

`<http://stackoverflow.com/questions/4936620/using-strides-for-an-/
efficient-moving-average-filter>`_

`<http://pangea.stanford.edu/~samuelj/musings/dems-in-python-pt-3-slope-/
and-hillshades-.html>`_

`<https://github.com/perrygeo/gdal_utils/blob/master/gis-bin/hillshade.py>`_

`<http://matplotlib.org/examples/specialty_plots/topographic_hillshading.html>`_

`<https://blogs.esri.com/esri/arcgis/2015/05/21/take-your-terrain-mapping-to/
-new-heights/>`_

`<http://gis.stackexchange.com/questions/146296/how-to-create-composite-
hillshade>`_

`<https://github.com/Blarghedy/TerrainGen-Python>`_

`<http://vterrain.org/Elevation/Artificial/>`_

`<http://geogratis.gc.ca/site/eng/extraction?id=2016_56ae834dd24892.15336554>`_

`<http://www.jennessent.com/downloads/dem%20surface%20tools%20for%20arcgis.pdf>`_

`<www.geog.ucsb.edu/~kclarke/G232/terrain/Zhang_etal_1999.pdf>`_


Requires:
---------
  requires arraytools.tools and stride from there
"""
# pylint: disable=C0103
# pylint: disable=R1710
# pylint: disable=R0914

# ---- begin with imports ----

import sys
from textwrap import dedent, indent
import numpy as np
import matplotlib.pyplot as plt
#from numpy.lib.stride_tricks import as_strided
from arraytools.tools import stride


ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.2f}'.format}
np.set_printoptions(edgeitems=3, linewidth=100, precision=2,
                    suppress=True, threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')

script = sys.argv[0]

__all__ = ['pad_a',
           'kernels',
           'stride',
           'filter_a',
           'slope_a',
           'aspect_a',
           'hillshade_a']

# ---- constants ----
surface_kernel = np.array([[1, 2, 1], [2, 0, 2], [1, 2, 1]])
all_f = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
no_cnt = np.array([[1, 1, 1], [1, np.nan, 1], [1, 1, 1]])
cross_f = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
plus_f = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])


# ---- functions ----
#
def pad_a(a):
    """Pads the input array using mode='edge' replicating the values
    at the edges and corners.
    """
    a_pad = np.pad(a, 1, mode='edge')
    return a_pad


def kernels(k=None):
    """Kernels used for slope calculations

    Requires:
    ---------
    - `f_dxyz`: default after ArcMap and Burrough, surface_kernel as above
    - `f_plus`:  maximum rise/fall after eppl7
    - `f_cross`
    - `'f_d8`
    """
    if k == 'plus_f':  # maximum rise/fall after eppl7
        k = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype="float64")
    elif k == 'cross_f':
        k = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype="float64")
    elif k == 'f_d8':
        t = np.sqrt(2.0)  # f_plus + t*cross used for distance
        k = np.array([[t, 1, t], [1, 0, 1], [t, 1, t]], dtype="float64")
    else:  # f_dxyz or None or none of the above
        k = np.array([[1, 2, 1], [2, 0, 2], [1, 2, 1]], dtype="float64")
    return k


def filter_a(a_s, a_filter=surface_kernel, cell_size=1):
    """Used by aspect, slope and hillshade to filter a raster/array

    Requires:
    --------
    - a_s : a strided array with the shape r*c*3x3 is input
    - filter : a 3x3 filter to apply to a_s
    - cell_size - for slope, (actual size)*8; for aspect (8 is required)
    """
    cs = cell_size*8.0
    f_dxyz = a_filter
    if a_filter is None:
        f_dxyz = np.array([[1, 2, 1], [2, 0, 2], [1, 2, 1]], dtype="float64")
    a_s = a_s * f_dxyz
    dz_dx = ((a_s[..., :, 2] - a_s[..., :, 0]).sum(axis=-1))/cs
    dz_dy = ((a_s[..., 2, :] - a_s[..., 0, :]).sum(axis=-1))/cs
    return dz_dx, dz_dy


# @time_deco
def slope_a(a, cell_size=1, kern=None, degrees=True, verb=False, keep=False):
    """Return slope in degrees for an input array using 3rd order
    finite difference method for a 3x3 moing window view into the array.

    Requires:
    ---------
    - a : an input 2d array. X and Y represent coordinates of the Z values
    - cell_size : cell size, must be in the same units as X and Y
    - kern : kernel to use
    - degrees : True, returns degrees otherwise radians
    - verb : True, to print results
    - keep : False, to remove/squeeze extra dimensions
    - filter :
        np.array([[1, 2, 1], [2, 0, 2], [1, 2, 1]]) **current default

    Notes:
    ------

    ::

        dzdx: sum(col2 - col0)/8*cellsize
        dzdy: sum(row2 - row0)/8*celsize
        Assert the array is ndim=4 even if (1,z,y,x)
        general         dzdx      +    dzdy     =    dxyz
        [[a, b, c],  [[1, 0, 1],   [[1, 2, 1]       [[1, 2, 1]
         [d, e, f]    [2, 0, 2], +  [0, 0, 0]   =    [2, 0, 2],
         [g, h, i]    [1, 0, 1]]    [1, 2, 1]]       [1, 2, 1]]

    """
    frmt = """\n    :----------------------------------------:
    :{}\n    :input array...\n    {}\n    :slope values...\n    {!r:}
    :----------------------------------------:
    """
    # ---- stride the data and calculate slope for 3x3 sliding windows ----
    np.set_printoptions(edgeitems=10, linewidth=100, precision=1)
    a_s = stride(a, win=(3, 3), stepby=(1, 1))
    if a_s.ndim < 4:
        new_shape = (1,) * (4-len(a_s.shape)) + a_s.shape
        a_s = a_s.reshape(new_shape)
    #
    kern = kernels(kern)  # return the kernel if specified
    # ---- default filter, apply the filter to the array ----
    #
    dz_dx, dz_dy = filter_a(a_s, a_filter=kern, cell_size=cell_size)
    #
    s = np.sqrt(dz_dx**2 + dz_dy**2)
    if degrees:
        s = np.rad2deg(np.arctan(s))
    if not keep:
        s = np.squeeze(s)
    if verb:
        p = "    "
        args = ["Results for slope_a... ",
                indent(str(a), p), indent(str(s), p)]
        print(dedent(frmt).format(*args))
    return s


def aspect_a(a, cell_size=1, flat=0.1, degrees=True, keepdims=False):
    """Return the aspect of a slope in degrees from North.

    Requires:
    --------
    - a :
        an input 2d array. X and Y represent coordinates of the Z values
    - cell_size :
        needed to proper flat calculation
    - flat :
        degree value, e.g. flat surface <= 0.05 deg

        0.05 deg => 8.7e-04 rad   0.10 deg => 1.7e-02 rad
    """
    if not isinstance(flat, (int, float)):
        flat = 0.1
    a_s = stride(a, win=(3, 3), stepby=(1, 1))
    if a_s.ndim < 4:
        new_shape = (1,) * (4-len(a_s.shape)) + a_s.shape
        a_s = a_s.reshape(new_shape)
    f_dxyz = np.array([[1, 2, 1], [2, 0, 2], [1, 2, 1]], dtype="float64")
    a_s = a_s * f_dxyz
    #
    dz_dx, dz_dy = filter_a(a_s, a_filter=f_dxyz, cell_size=1)
    #
    asp = np.arctan2(dz_dy, -dz_dx)     # relative to East
    # get the slope
    s = np.sqrt((dz_dx*cell_size)**2 + (dz_dy*cell_size)**2)
    asp = np.rad2deg(asp)
    asp = np.mod((450.0 - asp), 360.)   # simplest way to get azimuth
    asp = np.where(s <= flat, -1, asp)
    if not keepdims:
        asp = np.squeeze(asp)
    if not degrees:
        asp = np.deg2rad(asp)
    return asp


def aspect_dem(a):
    """Return aspect relative to north """
    a = np.asarray(a)
    dzdx_a = (a[:, 2] - a[:, 0]).sum() / 8.0  # aspect: col2 - col0
    dzdy_a = (a[2] - a[0]).sum() / 8.0        # aspect: row2 - row0
    s = np.degrees(np.arctan2(dzdy_a, -dzdx_a))
    aspect = np.where(s < 0, 90. - s,
                      np.where(s > 90, 450.0 - s, 90.0 - s))
    return aspect


def angle2azim(val):
    """correct x-oriented angle (in degrees) to N-oriented azimuth"""
    if val < 0:
        az = 90.0 - val
    elif val > 90.0:
        az = 450.0 - val
    else:
        az = 90.0 - val
    return az


def a2z(vals):
    """a numpy version of angle2azim for single values"""
    out = np.where(vals < 0, 90. - vals,
                   np.where(vals > 90, 450.0 - vals, 90.0 - vals))
    return out


def hillshade_a(a, cell_size=1, sun_azim=315, sun_elev=45):
    """Hillshade calculation as outlined in Burrough and implemented by
    : esri in ArcMap and ArcGIS Pro.  All measures in radians.

    - z, az :
        sun's zenith angle and azimuth
    - sl, asp :
        surface properties, slope and aspect
    - hillshade:
        255.0 * ((cos(z) * cos(sl)) + (sin(z) * sin(sl) * cos(az-asp)))
    """
    s_azi = np.deg2rad(sun_azim)
    s_elev = np.deg2rad(90.0 - sun_elev)
    a_a = aspect_a(a, degrees=False)
    a_s = slope_a(a, cell_size=cell_size, degrees=False)
    out = 255*((np.cos(s_elev) * np.cos(a_s)) +
               (np.sin(s_elev) * np.sin(a_s) * np.cos(s_azi - a_a)))
    out = np.where(out < 0, 0, out)
    return out.astype('int')


# ---- Demo section ----------------------------------------------------------
#
def _slope_aspect_demo_(cell_size=2):
    """Demo of the data set below"""
    a = np.array([[0, 1, 2, 3, 3, 3, 2, 1, 0],
                  [1, 2, 3, 4, 4, 4, 3, 2, 1],
                  [2, 3, 4, 5, 5, 5, 4, 3, 2],
                  [3, 4, 5, 5, 5, 5, 5, 4, 3],
                  [3, 4, 5, 5, 5, 5, 5, 4, 3],
                  [3, 4, 5, 5, 5, 5, 5, 4, 3],
                  [2, 3, 4, 5, 5, 5, 4, 3, 2],
                  [1, 2, 3, 4, 4, 4, 3, 2, 1],
                  [0, 1, 2, 3, 3, 3, 2, 1, 0]])
    r, c = a.shape
    data = stride(a, win=(3, 3), stepby=(1, 1))  # produce strided array
    slope = slope_a(a, cell_size=cell_size)
    slope = np.array(slope)
    aspect = [aspect_a(data[i][j])
              for i in range(data.shape[0])
              for j in range(data.shape[1])]
    aspect = np.array(aspect).reshape((r-2, c-2))
    frmt = """
    :---- Slope, Aspect Demo ------------------------------------------------
    :Sample DEM with a cell size of {} units.
    {}\n
    Slope (degrees) ...
    {}\n
    Aspect (degrees) ...
    {}
    """
    args = [cell_size]
    pre = '  ..'
    args.extend([indent(str(i), pre) for i in [a, slope, aspect]])
    print(dedent(frmt).format(*args))
    return a, slope, aspect


def pyramid(core=10, steps=10, stepby=2, incr=(1, 1), posi=True):
    """Create a pyramid see pyramid_demo.py"""
    a = np.array([core])
    a = np.atleast_2d(a)
    for i in range(stepby, steps+1, stepby):
        val = core-i
        if posi and (val <= 0):
            val = 0
        a = np.lib.pad(a, incr, "constant", constant_values=(val, val))
    return a


def _demo(cell_size=5, pad=False):
    """Demonstration of calculations
    :
    """
    frmt = """
    :------------------------------------------------------------------
    :{}{}\n    :input array\n    {}\n    :slope values\n    {}
    :aspect values\n    {}\n    :
    :------------------------------------------------------------------
    """
    ft0 = {'float': '{: 0.1f}'.format}
    np.set_printoptions()
    np.set_printoptions(linewidth=100, precision=1, formatter=ft0)
    # p = "    "
    # a = pyramid(core=5, steps=4, incr=(1,1), posi=True)
    t = 1.0e-8
    u = 1.0e-4
    # n = np.nan
    x = 4
    y = 2
    d0 = [[0, 0, 0, 2, 3, 4, 3, 2, 2, 4, 2, 3, 2],
          [0, 0, 0, 2, 4, 5, 4, 3, 3, 3, 3, 2, 1],
          [0, 0, 0, 2, 5, 6, 5, 4, 4, 2, 4, 1, 0]]
    d1 = [[0, 0, 0, 0, 0, 0, 0, 2, 2, 2, y, x, x, 2],
          [t, 0, u, u, t, 0, 0, 2, 2, 2, y, x, x, 2],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, y, x, x, 2],
          [1, 2, 3, 4, 5, 3, 3, 2, 1, 0, y, x, x, 2],
          [2, 3, 4, 5, 6, 4, 4, 2, 1, 0, y, x, x, 2],
          [3, 4, 5, 6, 3, 3, 3, 2, 1, 0, y, x, x, 2],
          [2, 2, 4, 5, 2, 2, 2, 2, 0, 0, y, x, x, 2],
          [1, 1, 2, 4, 1, 1, 1, 2, 0, 0, y, x, x, 2]]
    ds = [[50, 45, 50], [30, 30, 30], [8, 10, 10]]  # for slope
    da = [[101, 92, 85], [101, 92, 85], [101, 91, 84]]  # for aspect
    ds = [d0, d1, ds, da]
    for d in ds:
        a = np.array(d, dtype='float64')
        if pad:
            a = pad_a(a)
        sl = slope_a(a, cell_size, kern=None, degrees=True,
                     verb=False, keep=False)
        asp = aspect_a(a, cell_size=cell_size, flat=0.1)
        args = ["Surface properties for array shape ", a.shape, a, sl, asp]
        print(dedent(frmt).format(*args))
    return a, d0, d1


def circle_a(radius, pad=0, as_int=True):
    """Create a circle
    """
    np.set_printoptions(edgeitems=10, linewidth=100, precision=1)
    r = radius + pad
    circ = np.zeros((2*r + 1, 2*r + 1))
    y, x = np.ogrid[-r:r+1, -r:r+1]
    mask = x**2 + y**2 <= radius**2
    circ[mask] = 1
    if as_int:
        circ = circ.astype('int')
    return circ


def demo2():
    """ from 2.x """
    cell_size = 1
    a = pyramid(core=10, steps=10, stepby=1, incr=(1, 1), posi=True)
    sa = slope_a(a, cell_size=cell_size)
    dataExtent = [0, 21, 0, 21]
    hs = hillshade_a(a, cell_size=cell_size, sun_azim=270,
                     sun_elev=45.5)
    # bilinear
    f2 = plt.imshow(sa, interpolation='none', cmap='coolwarm',
                    vmin=a.min(), vmax=a.max(), extent=dataExtent)
    f1 = plt.imshow(hs, interpolation='bilinear', cmap='gray', alpha=0.7,
                    extent=dataExtent)
    plt.gca().invert_yaxis()
    plt.show()
#    plt.close()
    # print("\npyramid\n{}\nslope:\n{}\nhillshade\n{}".format(a, sa, hs))
    return f1, f2


def plot_(a, hs=None, interp='none'):
    """plot array simply with hillshade if available"""
    dataExtent = [0, a.shape[0], 0, a.shape[1]]
    f1 = plt.imshow(a, interpolation=interp, cmap='coolwarm',
                    vmin=a.min(), vmax=a.max(), extent=dataExtent)  # bilinear
    f1.show()
    if hs is not None:
        f2 = plt.imshow(hs, interpolation='bilinear', cmap='gray',
                        alpha=0.5, extent=dataExtent)
        f2.show()
    plt.gca().invert_yaxis()
#    plt.show()
#    plt.close()


def circ_demo(rmax=50, plot=False):
    """ plot a circle and normalize the plot"""
    circs = []
    for r in range(1, rmax):
        p = rmax-r-1
        c = circle_a(radius=r, pad=p, as_int=True)
        circs.append(c)
    cs = np.sum(circs, axis=0)
    cs[rmax-1, rmax-1] = rmax
    if plot:
        # cn = cs/float(rmax)  # normalize relative to array max
        plot_(cs)
    return cs


def single_demo():
    """Some finite single slope examples.
    :
    """
    cellsizes = [1, 2, 5, 10]  # cell size
    degrees = True
    a = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])
    a0 = [np.array([[0, 0, 0], [2, 2, 2], [4, 4, 4]]), 'North']
    a1 = [np.rot90(a, 3), 'NE']
    a2 = [np.rot90(a0[0], 3), 'E']
    a3 = [np.rot90(a, 2), 'SE']
    a4 = [np.rot90(a0[0], 2), 'S']
    a5 = [np.rot90(a, 1), 'SW']
    a6 = [np.rot90(a0[0], 1), 'W']
    a7 = [np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]]), 'NW']
    a_s = [a0, a1, a2, a3, a4, a5, a6, a7]
    for r, t in a_s:
        print("\nArray, facing...{}\n{}".format(t, r))
        print("   dx  slope  aspect hillshade""")
        for cs in cellsizes:
            sa = slope_a(r, cell_size=cs, degrees=degrees)
            asp = aspect_a(r, cell_size=cs, degrees=degrees)
            hs = hillshade_a(r, cell_size=cs)
            frmt = "{:>5.0f} {:6.1f} {:>7.1f} {:>7.1f}"
            args = [cs]  # cs is a scalar, rest are arrays
            args.extend([np.asscalar(i) for i in [sa, asp, hs]])
            print(dedent(frmt).format(*args))


# ---------------------------------------------------------------------
#
if __name__ == "__main__":
    """Main section...   """
#    print("Script... {}".format(script))
    #
#    axis = (-1,-2)
#    a, d0, d1 = _demo(cell_size=5, degrees=True, pad=False, verbose=True)
#    circ = circle_a(5, pad=3)
#    a, sa, hs = demo2()  # demo2...........
#    single_demo()

#    a = np.array([[0, 1, 2, 3, 3, 3, 2, 1, 0],
#                  [1, 2, 3, 4, 4, 4, 3, 2, 1],
#                  [2, 3, 4, 5, 5, 5, 4, 3, 2],
#                  [3, 4, 5, 5, 5, 5, 5, 4, 3],
#                  [3, 4, 5, 5, 5, 5, 5, 4, 3],
#                  [3, 4, 5, 5, 5, 5, 5, 4, 3],
#                  [2, 3, 4, 5, 5, 5, 4, 3, 2],
#                  [1, 2, 3, 4, 4, 4, 3, 2, 1],
#                  [0, 1, 2, 3, 3, 3, 2, 1, 0]])
