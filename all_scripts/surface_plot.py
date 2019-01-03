# -*- coding: UTF-8 -*-
"""
:Script:   surface_plot.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-12-31
:References
:  https://en.m.wikipedia.org/wiki/Great_Pyramid_of_Giza

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def show_3d(a, surf_shw=True, wire_shw=True, cont_shw=False):
    """Show a 3D array either as a surface, wireframe or contour or
    :  a combination
    : - a = np.random.random((nx, ny)) # original
    """
    nx, ny = a.shape
    x = range(nx)
    y = range(ny)
    ax = plt.figure()  # should call this fig
    ha = ax.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)  # plot_surface expects x and y data to be 2D
    if surf_shw:
        surf = ha.plot_surface(X, Y, a, rstride=1, cstride=1,
                               cmap=cm.coolwarm, linewidth=0,
                               antialiased=False)
        ax.colorbar(surf, shrink=0.5, aspect=5)  # ditto change to fig
    if wire_shw:  # was ax and 5,5
        ha.plot_wireframe(X, Y, a, rstride=1, cstride=1, color='black')
    if cont_shw:
        ha.contourf(X, Y, a, extend3d=True, cmap=cm.coolwarm)
    plt.show()


def pyramid(core=9, steps=11, incr=(1, 1), prn_arr=False):
    """create a pyramid with a core value, a certain number of steps
    :  decreasing by incr until done
    """
    a = np.array([core])
    a = np.atleast_2d(a)
    for i in range(1, steps):
        val = max(0, core - i)
        a = np.lib.pad(a, incr, "constant", constant_values=(val, val))
    if prn_arr:
        frmt = "\nSimple pyramid array... shape {}, ndim {} \n{}"
        print(frmt.format(a.shape, a.ndim, a))
    return a


if __name__ == "__main__":
    """create a pyramid and display it"""
    a = pyramid()
    show_3d(a, surf_shw=True, wire_shw=False, cont_shw=False)
