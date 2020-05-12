# -*- coding: utf-8 -*-
"""
triangulate
===========

Script :   triangulate.py

Author :   Dan_Patterson@carleton.ca

Modified : 2019-02-07

Purpose:  triangulate poly* features using scipy/qhull functions.

Useage :

>>> tri = Voronoi(pnts)
>>> dir(tri)
['__class__', '__del__', '__delattr__', '__dict__', '__dir__', '__doc__',
 '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__',
 '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__',
 '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__',
 '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_add_points',
 '_points', '_qhull', '_ridge_dict', '_update', 'add_points', 'close',
 'max_bound', 'min_bound', 'ndim', 'npoints', 'point_region', 'points',
 'regions', 'ridge_dict', 'ridge_points', 'ridge_vertices', 'vertices']

>>> tri.__class__
<class 'scipy.spatial.qhull.Voronoi'>

>>> tri.__dict__.keys()
dict_keys(['_qhull', 'vertices', 'ridge_points', 'ridge_vertices', 'regions',
 'point_region', '_ridge_dict', '_points', 'ndim', 'npoints', 'min_bound',
 'max_bound'])

>>> tri.min_bound, tri.max_bound
(array([-4217.93, -3832.13]), array([5268.65, 5495.64]))

Notes
-----
To get the centroid geometry, you can do the following.

>>> v = vor_pnts(aa, testing=False)
>>> cents = [[p.centroid.X, p.centroid.Y] for p in polys]
>>> polys = poly([v], SR)
>>> pnts = [arcpy.PointGeometry(Point(i[0], i[1])) for i in cents]
>>> out_fc = 'C:\\GIS\\A_Tools_scripts\\PointTools\\voronoi_delaunay.gdb\\p6'
>>> arcpy.FeatureToPoint_management (pnts, out_fc)

References
----------
voronoi/delaunay links:

`<http://zderadicka.eu/voronoi-diagrams/>`_.
`<http://scipy.github.io/devdocs/generated/scipy.spatial.Voronoi.html>`_.
`<https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram>`_.
`<https://stackoverflow.com/questions/36063533/clipping-a-voronoi-diagram
-python?noredirect=1&lq=1>`_.
`<https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/
scipy.spatial.Delaunay.html>`_.
---------------------------------------------------------------------
"""

import sys
import numpy as np
from scipy.spatial import Delaunay, Voronoi
#
from arcpy import Exists, AddMessage
from arcpy.management import (Delete, CopyFeatures, MakeFeatureLayer)
from arcpy.analysis import Clip
from arcpy.da import Describe, FeatureClassToNumPyArray
from arcpy.geoprocessing import env
from arcpy.arcobjects import Array, Point
from arcpy.arcobjects.geometries import Polygon

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=5, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

env.overwriteOutput = True

# ----
#
def tweet(msg):
    """Print a message for both arcpy and python.

    msg - a text message
    """
    m = "\n{}\n".format(msg)
    AddMessage(m)
    print(m)


def circle(radius=1.0, theta=10.0, xc=0.0, yc=0.0):
    """Produce a circle/ellipse depending on parameters.

    `radius` : number
        Distance from centre
    `theta` : number
        Angle of densification of the shape around 360 degrees
    """
    angles = np.deg2rad(np.arange(180.0, -180.0-theta, step=-theta))
    x_s = radius*np.cos(angles) + xc    # X values
    y_s = radius*np.sin(angles) + yc    # Y values
    pnts = np.c_[x_s, y_s]
    return pnts


def infinity_circle(a, fac=10):
    """Create an infinity circle to append to the original point list.

    Parameters:
    -----------
    a : array
        2D array of x, y point coordinates
    fac : number
        The factor to multiply the largest of the extent width or height for
        the input point array
    """
    fac = max(fac, 1)
    L, B = a.min(axis=0)
    R, T = a.max(axis=0)
    xc, yc = np.average(a, axis=0)
    circle_radius = max([R-L, T-B]) * fac #increase radius by a factor of 10
    circPnts = circle(radius=circle_radius, theta=10.0, xc=xc, yc=yc)
    return circPnts


def poly(pnt_groups, SR):
    """Short form polygon creation
    """
    polygons = []
    for pnts in pnt_groups:
        for pair in pnts:
            arr = Array([Point(*xy) for xy in pair])
            pl = Polygon(arr, SR)
            polygons.append(pl)
    return polygons


def vor_pnts(pnts, testing=False):
    """Return the point indices"""
    out = []
    avg = np.mean(pnts, axis=0)
    p =  pnts - avg
    tri = Voronoi(p)
    for region in tri.regions:
        r = [i for i in region if i != -1]
        if (len(r) > 2):
            poly = np.array([tri.vertices[i] + avg for i in r])
            out.append(poly)
            if testing:
                print("{}".format(poly.T))
    return out


def tri_pnts(pnts, testing=False):
    """Triangulate the points and return the triangles

    Parameters:
    -----------
    pnts : np.array
        Points in array format.
    out : array
        an array of triangle points

    Notes:
    ------
    >>> pnts = pnts.reshape((1,) + pnts.shape)  # a 3D set of points (ndim=3)
    >>> [pnts]  # or pass in as a list
    """
    out = []
    ps = np.unique(pnts, axis=0)  # get the unique points only
    avg = np.mean(ps, axis=0)
    p =  ps - avg
    tri = Delaunay(p)
    simps = tri.simplices
    new_pnts = [p[s]+avg for s in simps]
    if testing:
        print("{}".format(new_pnts))
    out.append(new_pnts)
    out = np.array(out).squeeze()
    return out


# ---- Do the work
#
def _tri_demo(tri_type='Delaunay'):
    """Triangulation demo.
    """
    from scipy.spatial import delaunay_plot_2d, voronoi_plot_2d
    import matplotlib.pyplot as plt
    xs =[ 48,   8, 623, 615, 196, 368, 112, 918, 318, 316, 427,
         364, 849, 827, 438, 957, 495, 317, 985, 534]
    ys = [674, 215, 834, 235, 748, 630, 876, 407,  33, 872, 893,
          360, 397, 902, 420, 430, 678, 523, 604, 292]
    aa = np.array(list(zip(xs, ys)))
    c = infinity_circle(aa, fac=0)
    a = np.vstack((aa, c))
    d = v = None  # initialize the output to None
    if tri_type == 'Delaunay':
        d = Delaunay(aa)
        plot = delaunay_plot_2d(d)
        x0, y0 = [0., 0.]
        x1, y1 = [1000., 1000.]
    else:
        c = infinity_circle(a, fac=2)
#        a = np.vstack((a, c))
        x0, y0 = a.min(axis=0)
        x1, y1 = a.max(axis=0)
        v = Voronoi(a, qhull_options='Qbb Qc Qx')
        plot = voronoi_plot_2d(v, show_vertices=True, line_colors='y',
                               line_alpha=0.8, point_size=5)
    # ----
    plot.set_figheight(10)
    plot.set_figwidth(10)
    plt.axis([0, 1000, 0, 1000])
#    plt.axis([x0, x1, y0, y1])
    plt.show()
    return aa, (d or v)

# ----
#
def _tri_tool():
    """Triangulation for tool
    """
    in_fc = sys.argv[1]
    tri_type = sys.argv[2]
    out_fc = sys.argv[3]
    xtent = sys.argv[4]
    desc = Describe(in_fc)
    SR = desc['spatialReference']
    flds = ['SHAPE@X', 'SHAPE@Y']
    allpnts = False
    z = FeatureClassToNumPyArray(in_fc, flds, "", SR, allpnts)
    a = np.zeros((z.shape[0], 2), dtype='<f8')
    a[:, 0] = z['SHAPE@X']
    a[:, 1] = z['SHAPE@Y']
    #
    if tri_type == 'Delaunay':
        tweet("Delaunay... clip extent {}".format(xtent))
        t = tri_pnts(a, True)  # must be a list of list of points
        polys = poly(t, SR)
        if Exists(out_fc):
            Delete(out_fc)
        CopyFeatures(polys, "in_memory/temp")
        MakeFeatureLayer("in_memory/temp", "temp")
        if xtent not in ("", None):
            Clip("temp", xtent, out_fc, None)
        else:
            CopyFeatures("temp", out_fc)
    else:
        tweet("Voronoi... clip extent {}".format(xtent))
        c = infinity_circle(a, fac=10)
        aa = np.vstack((a, c))
        v = vor_pnts(aa, testing=False)
        polys = poly([v], SR)
        if Exists(out_fc):
            Delete(out_fc)
        CopyFeatures(polys, "in_memory/temp")
        MakeFeatureLayer("in_memory/temp", "temp")
        if xtent not in ("", None, ):
            Clip("temp", xtent, out_fc, None)
        else:
            CopyFeatures("temp", out_fc)

# ---- main section .... calls demo or the tool
#
# uncomment the t = _tri... line below to graph
if len(sys.argv) == 1:
    testing = True
    a, t = _tri_demo('Delaunay')  # 'Delaunay' 'Voronoi'
else:
    testing = False
    _tri_tool()
# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
