# -*- coding: utf-8 -*-
"""
triangulate
===========

Script :   triangulate.py

Author :   Dan_Patterson@carleton.ca

Modified : 2018-08-24

Purpose:  triangulate poly* features

Useage :

References
----------
`<http://pro.arcgis.com/en/pro-app/arcpy/data-access/numpyarraytotable.htm>`_.

`<http://pro.arcgis.com/en/pro-app/arcpy/data-access/tabletonumpyarray.htm>`_.

`<https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/
scipy.spatial.Delaunay.html>`_.

`<https://tereshenkov.wordpress.com/2017/11/28/building-concave-hulls-alpha-
shapes-with-pyqt-shapely-and-arcpy/>`_.
---------------------------------------------------------------------
"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

import sys
import numpy as np
from scipy.spatial import Delaunay, Voronoi
from scipy.spatial import voronoi_plot_2d, delaunay_plot_2d

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


# ----
#
def v_plot(vor):
    """from Voronoi help"""
#    import matplotlib.pyplot as plt
    voronoi_plot_2d(vor)
#    plt.show()


def d_plot(tri):
    """delaunay plots, requires new points and simplices
    See : matplotlib.pyplot.triplot ... for additional options"""
#    import matplotlib.pyplot as plt
    delaunay_plot_2d(tri)
#    plt.show()


def Vor_pnts(pnts, testing=True, plot=True):
    """Requires a set of points deemed to be a cluster to delineate as a
    Voronoi diagram. You can do multiple point groupings by using this within
    a loop to return the geometries.
    """
    avg = np.mean(pnts, axis=0)
    p = pnts - avg
    tri = Voronoi(p)
    out = []
    for region in tri.regions:
        if not -1 in region:
            polygon = np.array([tri.vertices[i] + avg for i in region])
            out.append(polygon)
            if testing:
                print("{}".format(polygon.T))
    if plot:
        voronoi_plot_2d(tri)
    return out


def Del_pnts(pnts, testing=False, plot=True):
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
    pnts = np.unique(pnts, axis=0)  # get the unique points only
    avg = np.mean(pnts, axis=0)
    p = pnts - avg
    tri = Delaunay(p)
    simps = tri.simplices
    del_pnts = [p[s]+avg for s in simps]
    if testing:
        print("{}".format(del_pnts))
    if plot:
        delaunay_plot_2d(tri)
    return del_pnts, pnts, simps


# ---- Do the work
#

#pnts = np.array([[0, 0], [0, 100], [100, 100], [100, 80], [20,  80],
#                 [20, 20], [100, 20], [100, 0], [0, 0]])
# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
