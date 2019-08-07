"""
===============
smallest_circle
===============

Script :
    smallest_circle.py for npgeom

Author :
    Dan_Patterson@carleton.ca

Modified :
    2019-08-05

Purpose :
    Returns the smallest circle enclosing a shape in the form of a center and
    radius.  Original in smallestCircle.py in Bounding Containers.

Requires :
    Must have at least two points.

References
----------

de Berg et al., Computational Geometry with Applications, Springer-Verlag.

Welzl, E. (1991), Smallest enclosing disks (balls and ellipsoids),
Lecture Notes in Computer Science, Vol. 555, pp. 359-370.

`<https://stackoverflow.com/questions/27673463/smallest-enclosing-circle-in-
python-error-in-the-code>`_.

>>> cent = array([ 421645.83745955, 4596388.99204294])
>>> Xc, Yc, radius = 421646.74552, 4596389.82475, 24.323246

mean of points :
    [  421645.83745955  4596388.99204294] ## correct mean
translated to origin :
    (0.9080813432488977, 0.8327111343034483, 24.323287017466253)
direct calculation :
    (421646.74554089626, 4596389.8247540779, 24.323287017466253)
"""

import numpy as np

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 8.2f}'.format}

np.set_printoptions(
        edgeitems=5,
        threshold=500,
        floatmode='maxprec',
        precision=2, suppress=True, linewidth=120,
        nanstr='nan', infstr='inf', sign='-',
        formatter=ft)


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


# ---- smallest circle implementation ----------------------------------------
# helpers : farthest, center, distance
def farthest(a, check=False):
    """Distance matrix calculation for 2D points using einsum, yielding the
    two points which have the greatest distance between them.
    """
    if check:
        a = np.unique(a, axis=0)
    b = a.reshape(a.shape[0], 1, a.shape[-1])
    diff = a - b
    dist_arr = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    t_low = np.tril(dist_arr)            # np.triu(dist_arr)
    dist_order = np.unique(t_low)[::-1]  # largest to smallest unique dist
    mx = dist_order[0]
    r, c = np.where(t_low == mx)
    return a[c][0], a[r][0], dist_order


def center(p0, p1):
    """Center point between two points."""
    return np.mean((p0, p1), axis=0)


def distance(p0, p1):
    """Distance between two points."""
    return np.hypot(*(p1 - p0))


def small_circ(a):
    """Return the minimum area bounding circle for a points array.
    The ``unique`` points are used since np.unique removes reduncant calls and
    sorts the points in ascending order.

    Notes
    -----
    This incarnation uses a mix of pure python and numpy functionality where
    appropriate.  A simple check is first made to see if the farthest points
    enclose the point set.  If not, then the search continues to attempt to
    find 2, 3 or more points that form a circle to completely enclose all the
    points.
    """
    a = np.unique(a, axis=0)
    N = a.shape[0]
    if N <= 1:
        return a[0], 0.0, a
    if N == 2:
        cent = center(*a[:2])
        radius = distance(cent, a[0])
        return cent, radius, a[:2]
    # ---- corner-cases/garbage checking over
    p0, p1, _ = farthest(a, check=False)
    cent = center(p0, p1)
    radius = distance(cent, p0)
    check = np.sqrt(np.einsum('ij,ij->i', a-cent, a-cent)) <= radius
    if not np.all(check):  # degenerate case found
        for i in range(1, N):
            ptP = a[i]
            if distance(cent, ptP) > radius:
                prev = i - 1
                cent, radius = sub_1(a, prev, ptP)
    check = np.sqrt(np.einsum('ij,ij->i', a-cent, a-cent)) - radius
#    pnts = a[np.isclose(check, 0.)]
    return cent[0], cent[1], radius  # , pnts


# -------------------------------------------------------------------
def sub_1(pnts, prev, ptQ):
    """Stage 1 check.  Calls sub_2 to complete the search."""
    N = prev
    cent = center(pnts[0], ptQ)
    radius = distance(cent, ptQ)
    for i in range(1, N + 1):
        ptP = pnts[i]
        if distance(ptP, cent) > radius:
            N = i - 1
            cent, radius = sub_2(pnts, N, ptQ, ptP)
    return cent, radius


# -------------------------------------------------------------------
def sub_2(pnts, N, ptQ, ptP):
    """Returns the {cent, radius} for the smallest disc enclosing the points
    in the list with PointR and PointQ on its boundary.
    """
    if pnts.size == 0:
        pnts = np.array([[1.0, 1.0]])  # check
        N = 0
        ptQ = np.array([0.0, 0.0])
        ptR = np.array([1.0, 0.0])
    else:
        ptR = ptP
    cent = center(ptR, ptQ)
    radius = distance(cent, ptQ)
    ptO = np.array([0.0, 0.0])
    ptB = ptR - ptQ
    c2 = (distance(ptR, ptO)**2 - distance(ptQ, ptO)**2)/2.0
    for i in range(0, N + 1):
        ptP = pnts[i]
        if distance(ptP, cent) > radius:
            if np.all([0.0, 0.0] == ptB):
                cent = center(ptP, ptQ)
                radius = distance(ptQ, cent)
            else:
                ptA = ptQ - ptP
                xDelta = ptA[0] * ptB[1] - (ptA[1] * ptB[0])
                if abs(xDelta) >= 1.0e-06:  # 0.0:
                    c1 = (distance(ptQ, ptO)**2 - (distance(ptP, ptO)**2))/2.0
                    x = (ptB[1] * c1 - (ptA[1] * c2)) / xDelta
                    y = (ptA[0] * c2 - (ptB[0] * c1)) / xDelta
                    cent = [x, y]
                    radius = distance(cent, ptP)
    return cent, radius


# ===========================================================================
#
if __name__ == "__main__":
    """optional location for parameters"""
