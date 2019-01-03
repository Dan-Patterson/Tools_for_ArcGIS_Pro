# coding: utf-8
"""
Script:  slope_calc_demo.py
Author:  Dan.Patterson@carleton.ca
Purpose: Slope and aspect calculations using numpy
References:

Windows for slope:
f_dxyz - as implemented in arcmap after Burrough
   np.array([[1,2,1],
             [2,0,2],
             [1,2,1]], dtype="float64")

f_plus   - maximum rise/fall after eppl7
   np.array([[0,1,0],
             [1,0,1],
             [0,1,0]], dtype="float64")
f_cross  -
   np.array([[1,0,1],
             [0,0,0],
             [1,0,1]], dtype="float64")
f_d8   t = np.sqrt(2.0)  f_plus + t*cross used for distance
   np.array([[t,1,t],
             [1,0,1],
             [t,1,t]], dtype="float64")
:-----------------------
Notes:
:
Slope calculation   3rd order finite distance Horn (1981) see Burrough
-----------------
:    [a, b, c],    [1, 1, 1],
:    [d, *, f],    [3, 3, 3],
:    [g, h, i]])   [5, 5, 5]
    [dz/dx] = ((c + 2f + i) - (a + 2d + g)) / (8 * x_cellsize)
    [dz/dy] = ((g + 2h + i) - (a + 2b + c)) / (8 * y_cellsize)

      rise_run = sqrt([dz/dx]**2 + [dz/dy]**2)
               = sqrt((0)**2 + (2)**2) = sqrt(0 + 4) = 2
      slope = atan( sqrt([dz/dx]**2 + [dz/dy]**2) ) in radians
            = 63.434 degrees
      [dz/dx] = ((c + 2f + i) - (a + 2d + g)) / (8 * x_cellsize)
              = ((1 + 2*3 +5) - (1 + 2*3 +5)) / (8. * 1)
              = 0.0

      [dz/dy] = ((g + 2h + i) - (a + 2b + c)) / (8 * y_cellsize)
              = ((5 + 2*5 +5) - (1 + 2*1 +1)) / (8. * 1)
              = 2.0
      rise_run = sqrt(([dz/dx]**2 + [dz/dy]**2)
               = sqrt((0)**2 + (2.0)**2) = sqrt(4.0) = 2.0
      slope = np.degrees(np.arctan(2)) = 63.43...
:------------------
Aspect calculation:
-------------------
    [dz/dx] = ((c + 2f + i) - (a + 2d + g)) / 8
    [dz/dy] = ((g + 2h + i) - (a + 2b + c)) / 8
     aspect = 57.29578 * atan2 ([dz/dy], -[dz/dx])
        if aspect < 0
            cell = 90.0 - aspect  else if aspect > 90.0
            cell = 450.0 - aspect
         else
            cell = 90.0 - aspect

          [dz/dx] = ((c + 2f + i) - (a + 2d + g)) / 8.0
                  = ((85 + 170 + 84)) - (101 + 202 + 101)) / 8.0
                  = -8.125

          [dz/dy] = ((g + 2h + i) - (a + 2b + c)) / 8.0
                  = ((101 + 182 + 84) - (101 + 184 + 85)) / 8.0
                  = -0.375

        aspect = 57.29578 * atan2 ([dz/dy], -[dz/dx])
                 = 57.29578 * atan2 (-0.375, 8.125)
                 = -2.64

        aspect rule:
          if aspect < 0
            cell = 90.0 - aspect  else if aspect > 90.0
            cell = 360.0 - aspect + 90.0
          else
            cell = 90.0 - aspect
:--------------------
    interweave a list: this type works with both dtypes
       [val for pair in zip(l1, l2) for val in pair]
    interweave an array: only if they are the same dtype
      intl = np.ravel(np.column_stack((a,b)))

    years = ['1999', '2000', '2001', '2002', '2003',
             '2004', '2005', '2006','2007', '2008']
    text = [z[0] +"_"+ pair[1] for z in zip(years[:-1], years[1:])]
"""
# ---- begin with imports ----
import numpy as np
from numpy.lib.stride_tricks import as_strided
np.set_printoptions(edgeitems=3, linewidth=80, precision=2,
                    suppress=True, threshold=100)
import matplotlib.pyplot as plt

__all__ = ["slide_a"]


def kernels(k=None):
    """Kernels used for slope calculations

    Requires:
    ---------
    - `f_dxyz`: default after ArcMap and Burrough
    - `f_plus`:  maximum rise/fall after eppl7
    - `f_cross`
    - `'f_d8`
    """
    if k == 'f_plus':  # maximum rise/fall after eppl7
        k = np.array([[0,1,0], [1,0,1], [0,1,0]], dtype="float64")
    elif k == 'f_cross':
        k = np.array([[1,0,1], [0,0,0], [1,0,1]], dtype="float64")
    elif k == 'f_d8':
        t = np.sqrt(2.0)  # f_plus + t*cross used for distance
        k = np.array([[t,1,t], [1,0,1], [t,1,t]], dtype="float64")
    else:  # f_dxyz or None or none of the above
        k = np.array([[1,2,1], [2,0,2], [1,2,1]], dtype="float64")
    return k

def slide_a(a, block=(3, 3)):
    """Provide a 2D sliding/moving array view.  There is no edge
    correction for outputs.
    """
    r, c = block  # 3x3 block default
    a = np.ascontiguousarray(a)
    shape = (a.shape[0] - r + 1, a.shape[1] - c + 1) + block
    strides = a.strides * 2
    s_a = as_strided(a, shape=shape, strides=strides)
    return s_a


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


def slope_dem(a, cell_size=1, kernel=None):
    """Return slope in degrees for an input array using the 2nd order
    :  finite difference method
    """
    def cal_slope(win, cell_size):
        """Calculate the slope for the window"""
        dzdx_s = (win[:, 2] - win[:, 0]).sum()/cell_size  # slope: col2 - col0
        dzdy_s = (win[2] - win[0]).sum()/cell_size        # slope: row2 - row0
        slope = np.sqrt(dzdx_s**2 + dzdy_s**2)
        slope = np.rad2deg(np.arctan(slope))
        return slope
    # ---- read array and parse to calculate slope
    a = np.ascontiguousarray(a)
    ndim = a.ndim
    shp = a.shape
    kern = kernels(kernel)  # factor
    cell_size = (8.0 * cell_size)   # cell size
    a = a * kern                  # apply slope filter to array
    if ndim == 2:
        slope = cal_slope(a, cell_size)
    elif ndim == 3:
        slope = [cal_slope(a[i], cell_size)
                 for i in range(a.shape[0])]  # shape (0,x)
        slope = np.asarray(slope)
    elif ndim == 4:
        s0, s1, s2, s3 = shp
        slope = [cal_slope(a[i][j], cell_size)
                 for i in range(a.shape[0])   # shape (0,x,x,x)
                 for j in range(a.shape[1])]  # shape (x,1,x,x)
        slope = np.asarray(slope).reshape((s0, s1))
    return slope


def slope_map():
    """variant of slope """
    f_123 = np.array([1, 2, 3]).reshape((-1, 1))
    a0 = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    a1 = np.array([[1, 1, 1], [3, 3, 3], [5, 5, 5]])
    a2 = a0 * f_123
    a3 = [np.rot90(a0, i) for i in range(4)]  # 4 3x3 arrays in each
    a4 = [np.rot90(a1, i) for i in range(4)]
    a5 = [np.rot90(a2, i) for i in range(4)]
    #
    b0 = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])
    b1 = b0 * 2
    b2 = b0 * f_123
    b3 = [np.rot90(b0, i) for i in range(4)]
    b4 = [np.rot90(b1, i) for i in range(4)]  # 16 3x3 arrays total
    b5 = [np.rot90(b2, i) for i in range(4)]
    #
    arrs = np.array(a3 + b3 + a4 + b4 + a5 + b5)
    print("arrays shape {}".format(arrs.shape))
    a_s = np.array(arrs).reshape((6, 4, 3, 3))  # reshape to 4x4
    return a_s, arrs


def aspect_dem(a):
    """Return aspect relative to north """
    a = np.asarray(a)
    dzdx_a = (a[:, 2] - a[:, 0]).sum() / 8.0  # aspect: col2 - col0
    dzdy_a = (a[2] - a[0]).sum() / 8.0        # aspect: row2 - row0
    s = np.arctan2(dzdy_a, -dzdx_a)
    s = np.rad2deg(s)
    aspect = np.where(s < 0, 90. - s,
                      np.where(s > 90, 450.0 - s, 90.0 - s))
    return aspect


def aspect_demo():
    """Rotate a simple slope and determine the slope"""
    a = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])
    z = np.array([[1, 0, 1], [2, 2, 2], [3, 4, 3]])
    j = 0
    for i in range(4):
        b = np.rot90(a, i)
        c = aspect_dem(b)
        d = np.rot90(z, i)
        e = aspect_dem(d)
        print("\n({}) Array aspect...{}\n{}".format(j, e, d))
        j += 1
        print("\n({}) Array aspect...{}\n{}".format(j, c, b))
        j += 1
    return a


def slope_demo():
    """Rotate a simple slope and determine the slope
       The z-values are as shown, dx is varied to produce the slope
       values in both degrees and percent"""
    # a = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])
    a = np.array([[1, 1, 1], [3, 3, 3], [5, 5, 5]])
    j = 0
    print("Slope face...\n{}".format(a))
    print(" N     dx    deg.    %")
    dx = [0.5, 1., 2, 4, 6, 8, 10, 12.5, 15, 20, 25, 40, 80, 100]
    i = 1
    for j in dx:
        b = slope_dem(a, j)
        print("({:<2}) {:>5.1f}{:>7.1f} {:>6.1f}".format(i, j, b, (200.0/j)))
        i += 1
    return a


def main_demo():
    """Demo of the data set below"""
    data = np.array([[50, 45, 50],
                     [30, 30, 30],
                     [8, 10, 10]], dtype="float64")  # for slope
    # data = np.array([[101,92,85],[101,92,85],[101,91,84]]) # for aspect
    a = np.arange(36).reshape((6, 6))
    za = np.array([[0, 1, 2, 3, 3, 3, 2, 1, 0],
                   [1, 2, 3, 4, 4, 4, 3, 2, 1],
                   [2, 3, 4, 5, 5, 5, 4, 3, 2],
                   [3, 4, 5, 5, 5, 5, 5, 4, 3],
                   [3, 4, 5, 5, 5, 5, 5, 4, 3],
                   [3, 4, 5, 5, 5, 5, 5, 4, 3],
                   [2, 3, 4, 5, 5, 5, 4, 3, 2],
                   [1, 2, 3, 4, 4, 4, 3, 2, 1],
                   [0, 1, 2, 3, 3, 3, 2, 1, 0]])
    a = za
    r, c = a.shape
    data = slide_a(a, block=(3, 3))
    slope = slope_dem(data, cell_size=1, kernel=None)
    slope = np.array(slope)
    aspect = [aspect_dem(data[i][j])
              for i in range(data.shape[0])
              for j in range(data.shape[1])]
    aspect = np.array(aspect).reshape((r-2, c-2))
    frmt = "Dem...\n{}\nSlope...\n{}\nAspect\n{}"
    print(frmt.format(a, slope, aspect))
    return a, slope, aspect


def plot_grid(a, title="Grid"):
    """ """
    import matplotlib.pyplot as plt
    block = a
    plt.legend("hello")
    plt.matshow(block, cmap=plt.cm.gray)  # samplemat(d))
    plt.show()
    plt.close()


# -------------------------------------------------------------------
if __name__ == "__main__":
    """run sample for slope and aspect determinations for dem data"""
    #
#    a_s, arrs = slope_map()
#    slope = slope_dem(a_s, cell_size=5)
#    a = aspect_demo()
#    a = slope_demo()
    a, slope, aspect = main_demo()



"""
cell = (8.0*5)  # cell size
letters = [i for i in "abcdefghi"]
labels = np.array(letters).reshape((3,3))
#

t = np.sqrt(2.0)
f_d8 =np.array([[t,1,t], [1,1,1], [t,1,t]], dtype="float64")
#
print("array numbers and letters\n{}\n{}\n".format(data,labels))
# ----- Now create the values from the input data and the filter

lead = ["slope","aspect"] # np.array(lead)
args = [slope, aspect]                # np.array(args)
frmt = "{: <10s} {:>8.3f}\n"*len(args)
out = [val for pair in zip(lead, args) for val in pair]
print(frmt.format(*out))
"""


#vals = np.arange(-180.,180.,45.)
#for v in vals:
#    az = angle2azim(v)
#    print("angle {}  azimuth {}".format(v,az))

#block = three((3,3))
#plt.matshow(block, cmap=plt.cm.gray) #samplemat(d))
#plt.show()
#plt.close()
#years = ['1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006','2007', '2008']
#text = ["{}_{}".format(z[0], z[1]) for z in zip(years[:-1], years[1:])] #this type works with both dtypes
#z=-1
#zz = -(z + 180 % 360 - 180)  ' Convert to the range -180..180