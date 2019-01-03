# -*- coding: UTF-8 -*-
"""
n_spaced
========

Script :   n_spaced.py

Author :   Dan.Patterson@carleton.ca

Modified: 2018-04-09

Purpose:
--------
  Produce a point set whose interpoint spacing is no closer than a specified
  distance within a specified bounds.

References:
-----------
`<http://stackoverflow.com/questions/6835531/sorting-a-python-array-
recarray-by-column>`_.

"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

# ---- imports, formats, constants ------------------------------------------

import sys
from textwrap import dedent
import numpy as np

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.1f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2,
                    suppress=True, threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')

script = sys.argv[0]

# ---- functions ------------------------------------------------------------
#
def not_closer(a, min_d=1, ordered=False):
    """Find the points that are separated by a distance greater than
    min_d.  This ensures a degree of point spacing

    Parameters:
    --------
    a : array
      2D array of coordinates.
    min_d : number
      minimum separation distance
    ordered : boolean
      order the input points

    Returns:
    -------
    - b : points where the spacing condition is met
    - c : the boolean array indicating which of the input points were valid.
    - d : the distance matrix
    """
    if ordered:
        a = a[np.argsort(a[:, 0])]
    b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
    diff = b - a
    d = np.einsum('ijk,ijk->ij', diff, diff)
    d = np.sqrt(d).squeeze()
    c = ~(np.triu(d <= min_d, 1)).any(0)
    b = a[c]
    return b, c, d


def n_spaced(L=0, B=0, R=10, T=10, min_space=1, num=10, verbose=True):
    """Produce num points within the bounds specified by the extent (L,B,R,T)

    Parameters:
    ---------

    L(eft), B, R, T(op) : numbers
      extent coordinates
    min_space : number
      minimum spacing between points.
    num : number
      number of points... this value may not be reached if the extent
      is too small and the spacing is large relative to it.
    """
    #
    def _pnts(L, B, R, T, num):
        """Create the points"""
        xs = (R-L) * np.random.random_sample(size=num) + L
        ys = (T-B) * np.random.random_sample(size=num) + B
        return np.array(list(zip(xs, ys)))

    def _not_closer(a, min_space=1):
        """Find the points that are greater than min_space in the extent."""
        b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
        diff = b - a
        dist = np.einsum('ijk,ijk->ij', diff, diff)
        dist_arr = np.sqrt(dist).squeeze()
        case = ~(np.triu(dist_arr <= min_space, 1)).any(0)
        return a[case]
    #
    cnt = 1
    n = num * 2  # check double the number required as a check
    result = 0
    frmt = "Examined: {}  Found: {}  Need: {}"
    a0 = []
    while (result < num) and (cnt < 6):  # keep using random points
        a = _pnts(L, B, R, T, num)
        if cnt > 1:
            a = np.vstack((a0, a))
        a0 = _not_closer(a, min_space)
        result = len(a0)
        if verbose:
            print(dedent(frmt).format(n, result, num))
        cnt += 1
        n += n
    # perform the final sample and calculation
    use = min(num, result)
    a0 = a0[:use]  # could use a0 = np.random.shuffle(a0)[:num]
    a0 = a0[np.argsort(a0[:, 0])]
    return a0


def _demo():
    """ """
    # L, R, B, T = [300000, 300100, 5025000, 5025100]
    L, B, R, T = [1, 1, 10, 10]
    tol = 1
    N = 10
    a = n_spaced(L, B, R, T, tol, num=N, verbose=True)
    return a


if __name__ == "__main__":
    """ run the demos, comment out what you don't want"""
    # print("Script... {}".format(script))
#    a = np.array([[0, 0], [0, 2], [2, 2], [2, 0]], dtype='float64')
#    b = _demo()

# z = np.zeros((3,), dtype=[('A', 'int', (2,)), ('B', 'float')])
# z["A"] = np.arange(6).reshape(3,2)
