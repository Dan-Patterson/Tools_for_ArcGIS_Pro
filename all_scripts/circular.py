# -*- coding: UTF-8 -*-
"""
:Script:   circular.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-12-31
:Purpose:  See the documentation for the functions
:Notes:
:
:References:
:
"""
# pylint: disable=C0103
# pylint: disable=R1710
# pylint: disable=R0914

# ---- imports, formats, constants ----

import sys
import numpy as np


ft = {'bool': lambda x: repr(x.astype('int32')),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2,
                    suppress=True, threshold=100, formatter=ft)

_script = sys.argv[0]

__all__ = ["plot_",
           "rot_matrix",
           "_arc",
           "_circle",
           "arc_sector",
           "buffer_ring",
           "multiring_buffer_demo",
           "multi_sector_demo"
           ]


# ---- functions ----

def plot_(pnts):
    """plot a circle, arc sector etc
    """
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    fig, ax = plt.subplots()
    patches = []
    for i in pnts:  # Points need to form a closed loop
        polygon = Polygon(i, closed=False)  # closed=True if 1st/last pnt !=
        patches.append(polygon)
    p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=1.0)
    colors = 100*np.random.rand(len(patches))
    p.set_array(np.array(colors))
    ax.add_collection(p)
    plt.axis('equal')
    plt.show()
#    plt.close()


def rot_matrix(angle=0, nm_3=False):
    """Return the rotation matrix given points and rotation angle

    Requires:
    --------
      - rotation angle in degrees and whether the matrix will be used with
        homogenous coordinates

    Returns:
    -------
      - rot_m - rotation matrix for 2D transform
      - rotate around  translate(-x, -y).rotate(theta).translate(x, y)
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


def _arc(radius=100, start=0, stop=1, step=0.1, xc=0.0, yc=0.0):
    """Create an arc from a specified radius, centre and start/stop angles

    Requires:
    ---------
    `radius` : number
        cirle radius from which the arc is obtained
    `start`, `stop`, `step` : numbers
        angles in degrees
    `xc`, `yc` : number
        center coordinates in projected units

    Returns:
    --------
      points on the arc
    """
    start, stop = sorted([start, stop])
    angle = np.deg2rad(np.arange(start, stop, step))
    x_s = radius*np.cos(angle)         # X values
    y_s = radius*np.sin(angle)         # Y values
    pnts = np.c_[x_s, y_s]
    pnts = pnts + [xc, yc]
    p_lst = pnts.tolist()
    return p_lst


def _circle(radius=100, clockwise=True, theta=1, rot=0.0, scale=1,
            xc=0.0, yc=0.0):
    """Produce a circle/ellipse depending on parameters.

    Requires
    --------
    `radius` : number
        in projected units
    `clockwise` : boolean
        True for clockwise (outer rings), False for counter-clockwise
        (for inner rings)
    `theta` : number
        Angle spacing. If theta=1, angles between -180 to 180, are returned
        in 1 degree increments. The endpoint is excluded.
    `rot` : number
         rotation angle in degrees... used if scaling is not equal to 1
    `scale` : number
         For ellipses, change the scale to <1 or > 1. The resultant
         y-values will favour the x or y-axis depending on the scaling.

    Returns:
    -------
      list of coordinates for the circle/ellipse

    Notes:
    ------
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
    pnts = np.c_[x_s, y_s]
    if rot != 0:
        rot_mat = rot_matrix(angle=rot)
        pnts = (np.dot(rot_mat, pnts.T)).T
    pnts = pnts + [xc, yc]
    return pnts


def arc_sector(outer=10, inner=9, start=1, stop=6, step=0.1):
    """Form an arc sector bounded by a distance specified by two radii

    `outer` : number
        outer radius of the arc sector
    `inner` : number
        inner radius
    `start` : number
        start angle of the arc
    `stop` : number
        end angle of the arc
    `step` : number
        the angle densification step

    Requires:
    --------
      `_arc` is used to produce the arcs, the top arc is rotated clockwise and
      the bottom remains in the order produced to help form closed-polygons.
    """
    s_s = [start, stop]
    s_s.sort()
    start, stop = s_s
    top = _arc(outer, start, stop, step, 0.0, 0.0)
    top.reverse()
    bott = _arc(inner, start, stop, step, 0.0, 0.0)
    top = np.array(top)
    bott = np.array(bott)
    close = top[0]
    pnts = np.asarray([i for i in [*top, *bott, close]])
    return pnts


def buffer_ring(outer=100, inner=0, theta=10, rot=0, scale=1, xc=0.0, yc=0.0):
    """Create a multi-ring buffer around a center point (xc, yc)
     outer - outer radius
     inner - inner radius
     theta - angles to use to densify the circle...
        - 360+ for circle
        - 120 for triangle
        - 90  for square
        - 72  for pentagon
        - 60  for hexagon
        - 45  for octagon
        - etc
     rot - rotation angle, used for non-circles
     scale - used to scale the y-coordinates

    """
    top = _circle(outer, clockwise=True, theta=theta, rot=rot,
                  scale=scale, xc=xc, yc=yc)
    if inner != 0.0:
        bott = _circle(inner, clockwise=False, theta=theta, rot=rot,
                       scale=scale, xc=xc, yc=yc)
        pnts = np.asarray([i for i in [*top, *bott]])
    else:
        pnts = top
    return pnts


# ---- demo functions ----

def multiring_buffer_demo():
    """Do a multiring buffer
     rads - buffer radii
     theta - angle density... 1 for 360 ngon, 120 for triangle
     rot - rotation angle for ellipses and other shapes
     scale - scale the y-values to produce ellipses
    """
    buffers = []
    radii = [10, 20, 40, 80, 100]  # , 40, 60, 100]
    theta = 10
    rot = 22.5
    scale = 0.7
    for r in range(1, len(radii)):
        ring = buffer_ring(radii[r], radii[r-1], theta, rot, scale)
        buffers.append(ring)
    plot_(buffers)
    # return buffers


def multi_sector_demo():
    """Produce multiple sectors  """
    sectors = []
    outer = 10
    inner = 9
    incr = np.arange(0, 91, 1)  # (0,361,5)
    for outer in range(6, 10):
        inner = outer - 1
        for i in range(0, len(incr)):
            st = incr[i]
            end = incr[i-1]
            arc = arc_sector(outer, inner, start=st, stop=end, step=0.1)
            sectors.append(arc)
    plot_(sectors)


def help_():
    """Print the docs"""
    args = ['_arc ....', _arc.__doc__,
            'arc_sector ....', arc_sector.__doc__,
            '_circle ....', _circle.__doc__,
            'buffer_ring ....', buffer_ring.__doc__,
            'rot_matrix ....', rot_matrix.__doc__,
            'buffer_ring ....', buffer_ring.__doc__]
    frmt = "-"*60 + "\ncircular.py ....\n\n" + "{}\n"*len(args)
    print(frmt.format(*args))
    del frmt, args


# ----------------------
if __name__ == "__main__":
    """Uncomment what you want to see"""
#    print("Script... {}".format(_script))
#    circ_pnts = _circle(radius=1, theta=30, xc=5, yc=5)
#    print("\ncircle points...\n{}".format(circ_pnts))
#    arc_pnts = _arc(radius=10, start=0, stop=90.5, step=5, xc=0.0, yc=0.0)
#    print("\narc points...\n{}".format(arc_pnts))
#    pnts = arc_sector()
#    pnts = buffer_ring()
#    multi_sector_demo()
#    multiring_buffer_demo()
