# -*- coding: UTF-8 -*-
"""
raster_functions
================

Script :   raster_functions.py

Author :   Dan.Patterson@carleton.ca

Modified: 2018-03-28

Purpose:  tools for working with numpy arrays

Useage : See header for each function

References:

 ---------------------------------------------------------------------
"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

# ---- imports, formats, constants ----
import sys
import numpy as np


ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

__all__ = ['ufunc_add',
           'ufunc_sub',
           'rotate_mesh',
           'mesh_arr'
           ]


def ufunc_add(xx=None, yy=None, scale_x=1, scale_y=1):
    """A ufunc that just permits adding a meshgrid xx and yy values with x, y
    scaling

    Parameters:
    -----------
    - xx : meshgrid xx values
    - yy : meshgrid yy values
    - scale_x : scale the xx range values prior to addition
    - scale_y : same but for yy range values

     >>> xx, yy = mesh_arr(x_ax=(0, 5., 1.0), y_ax=(0, 5., 1.0))
     >>> ufunc_add(xx=xx, yy=yy)
     array([[ 0.,  1.,  2.,  3.,  4.],
            [ 1.,  2.,  3.,  4.,  5.],
            [ 2.,  3.,  4.,  5.,  6.],
            [ 3.,  4.,  5.,  6.,  7.],
            [ 4.,  5.,  6.,  7.,  8.]])

    Notes:
    ------
    Other examples
    ::
        z = xx * 1 + yy * 1  # same as above
        z + np.flipud(z)
        array([[  4.,   6.,   8.,  10.,  12.],
               [  4.,   6.,   8.,  10.,  12.],
               [  4.,   6.,   8.,  10.,  12.],
               [  4.,   6.,   8.,  10.,  12.],
               [  4.,   6.,   8.,  10.,  12.]])

        z + np.fliplr(z)
        array([[  4.,   4.,   4.,   4.,   4.],
               [  6.,   6.,   6.,   6.,   6.],
               [  8.,   8.,   8.,   8.,   8.],
               [ 10.,  10.,  10.,  10.,  10.],
               [ 12.,  12.,  12.,  12.,  12.]])

    I suppose you have figured out that the documentation is larger than the
    actual script code.

    See `mesh_arr` for meshgrid construction.
    """
    z = xx * scale_x + yy * scale_y
    return z


def ufunc_sub(xx=None, yy=None, scale_x=1, scale_y=1):
    """A ufunc that just permits subtrace a meshgrid xx and yy values with x, y
    scaling.

    See:
    ----
    ufunc_add : for full details

    Other examples
    ::
        z = xx * 1 - yy * 1  # see ufunc_add
        array([[ 0.,  1.,  2.,  3.,  4.],
               [-1.,  0.,  1.,  2.,  3.],
               [-2., -1.,  0.,  1.,  2.],
               [-3., -2., -1.,  0.,  1.],
               [-4., -3., -2., -1.,  0.]])

        z + np.flipud(z)
        array([[-4., -2.,  0.,  2.,  4.],
               [-4., -2.,  0.,  2.,  4.],
               [-4., -2.,  0.,  2.,  4.],
               [-4., -2.,  0.,  2.,  4.],
               [-4., -2.,  0.,  2.,  4.]])

        z + np.fliplr(z)
        array([[ 4.,  4.,  4.,  4.,  4.],
               [ 2.,  2.,  2.,  2.,  2.],
               [ 0.,  0.,  0.,  0.,  0.],
               [-2., -2., -2., -2., -2.],
               [-4., -4., -4., -4., -4.]]

    See `mesh_arr` for meshgrid construction.
    """
    z = xx * scale_x - yy * scale_y  # see other ufunc examples
    return z


def rotate_mesh(x_ax=(0, 10., 1.0), y_ax=(0, 10., 1.0), rot_angle=0):
    """Generate a meshgrid and rotate it by rot_angle degrees.

    Parameters:
    -----------
    x_ax : tuple
        x_min, x_max, dx of integers or floats
    y_ax : tuple
        y_min, y_max, dy
    rot_angle : number
        rotation angle in degrees

    Returns:
    --------
    Rotated xx, yy meshgrid parameters with a clockwise rotation

    References:
    -----------
    https://stackoverflow.com/questions/29708840/rotate-meshgrid-with-numpy

    Extra:
    ------
    https://stackoverflow.com/questions/32544636/transform-image-data-in-3d/\
    32546099#32546099

    https://stackoverflow.com/questions/31816754/\
    numpy-einsum-for-rotation-of-meshgrid

    Similar to above, but rotation in 3d

    >>> x, y, z = np.meshgrid(np.linspace(0, 1, 4),
                              np.linspace(0, 1, 3),
                              [.5], indexing='xy')
    >>> M = np.array([[ 0., -1.,  0.],
                      [ 1.,  0.,  0.],
                      [ 0.,  0.,  1.]])

    Einsum is used to do the rotation given, x, y, z and a rotation matrix.

    >>> xp, yp, zp = np.einsum('ij,jklm->iklm', M, [x, y, z])

    >>> xp.squeeze()
    array([[ 0. ,  0. ,  0. ,  0. ],
           [-0.5, -0.5, -0.5, -0.5],
           [-1. , -1. , -1. , -1. ]])
    >>> yp.squeeze()
    array([[ 0. ,  0.3,  0.7,  1. ],
           [ 0. ,  0.3,  0.7,  1. ],
           [ 0. ,  0.3,  0.7,  1. ]])
    >>> zp.squeeze()
    array([[ 0.5,  0.5,  0.5,  0.5],
           [ 0.5,  0.5,  0.5,  0.5],
           [ 0.5,  0.5,  0.5,  0.5]])

    plus more to come.
    """
    # 2D rotation matrix.  Clockwise rotation
    ang_rad = np.radians(rot_angle)
    rot_matrix = np.array([[np.cos(ang_rad), np.sin(ang_rad)],
                           [-np.sin(ang_rad), np.cos(ang_rad)]])
    xs = np.arange(*x_ax)
    ys = np.arange(*y_ax)
    x, y = np.meshgrid(xs, ys, indexing='xy')
    return np.einsum('ji, mni -> jmn', rot_matrix, np.dstack([x, y]))


def mesh_arr(x_ax=(0, 10., 1.0), y_ax=(0, 10., 1.0)):
    """Construct a mesh grid given the above ranges for x and y

    Parameters:
    -----------
    x_ax : tuple of x_min, x_max, dx
    y_ax : tuple of y_min, y_max, dy

    Returns:
    --------
    A meshgrid

    >>> xx, yy = mesh_arr(x_ax=(0, 5, 1), y_ax=(0, 3, 1))
    xx
    array([[0, 1, 2, 3, 4]])
    yy
    array([[0],
           [1],
           [2]])

    See also:
    ---------
    `ufunc_add` and `ufunc_sub` have examples, including 90 degree rotations
    in the x and y directions to produce altered meshgrids.
    """
    x = np.arange(*x_ax)
    y = np.arange(*y_ax)
    xx, yy = np.meshgrid(x, y, sparse=True, indexing='xy')
    return xx, yy


#def slope_arr(dx, dy):
#    """Create an array with a preferred slope
#
#    Parameters:
#    -----------
#    - dx : width in the x-direction ie. the columns of the array
#    - dy : height in the y-direction ie. the rows of the array
#
#    """


def _demo():
    """
    : -
    """
    pass


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
    _demo()
