# -*- coding: UTF-8 -*-
"""
:Script:   diamond_square.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-12-20
:Purpose:
:-------
:The diamond-square algorithm is used to generate synthetic terrain samples.
:These are used for slope, aspect and hillshade algorithm testing.
:
:Useage:
:------
:Specify a power (n) to get a square array of a certain size/shape (see below).
:The default min/max and rows/columns are specified within d_s.  You can use
:'scale_range' to change the values after if you like.  The number of rows/cols
:are used to tile/replicate the base array in the x,y directions during array
:creation, or you can use the np.fliplr, np.flipud to do this yourself later.
:Roughness (r) can be specified if you want to add noise to the array.  Values
:between 0-1 are accepted, but I suggest keeping them small
:
:Notes: relative timing (n= power, time, array shape)
:  3   0.3 ms (9, 9)
:  7  47.4 ms (129, 129)
:  8 189.  ms (257, 257)
   9 770.  ms (513, 513)
: 10   3.1 s  (1025, 1025)
:
:References:
:----------
:  https://scipython.com/blog/cloud-images-using-the-diamond-square-algorithm/
:  https://gist.github.com/CMCDragonkai/6444bf7ea41b4f43766abb9f4294cd69
:  https://raw.githubusercontent.com/buckinha/DiamondSquare/master
:        /DiamondSquare.py
:  *** http://paulbourke.net/fractals/noise/
:  *** https://en.m.wikipedia.org/wiki/Diamond-square_algorithm
:       https://github.com/buckinha/DiamondSquare
:       https://github.com/Crowgers/Diamond_Square
:       https://github.com/elite174/TextureGen
:     - https://joecrossdevelopment.wordpress.com/2012/04/30/
:       2d-random-terrain-iterative-diamond-square-algorithm/
:     - https://scipython.com/blog/cloud-images-using-the-diamond-square-
:       algorithm/
:**http://www-cs-students.stanford.edu/~amitp/
:       game-programming/polygon-map-generation/
:---------------------------------------------------------------------:
"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

# ---- imports, formats, constants ----
import sys
import numpy as np
import matplotlib.pyplot as plt

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=5, linewidth=80, precision=3, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

__all__ = ['scale_range',
           'normalize_to',
           'n_p',
           'd_s',
           '_diamond_',
           '_square_']


def scale_range(a, new_min=0, new_max=1):
    """scale an input array-like to a mininum and maximum number the input
    :array must be of a floating point array if you have a non-floating
    :point array, convert to floating using `astype('float')` this works
    :with n-dimensional arrays it will mutate in place.
    :min and max can be integers
    :Source:
    :--- https://gist.github.com/CMCDragonkai/6444bf7ea41b4f43766abb9f4294cd69
    def scale_range (input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input
    """
    dt = a.dtype
    a.dtype = 'float64'
    if (new_max - new_min) == 0.:
        raise ValueError("Max == Min {}=={}".format(new_max, new_min))
    a += -(np.min(a))
    a /= np.max(a) / (new_max - new_min)
    a += new_min
    a = a.astype(dt)
    return a


def normalize_to(a, upper=1, as_=float):
    """normalize an array
    :Normalised [0, 1]  a0 = (a - np.max(a))/-np.ptp(a)
    :Normalised [-1,1]  a1 = 2*(a - np.max(a))/-np.ptp(a)-1
    :Normalised [0,255] as integer
    :  a2 = (255*(a - np.max(a))/-np.ptp(a)).astype(int)
    :  a3 = (a - np.mean(a)) / np.std(a)
    """
    b = (upper*(a - np.max(a))/-np.ptp(a)).astype(as_)
    return b


def n_p(r_c=[10, 10], pmax=10):
    """Return the array to form based on desired size. This will be sliced
    :  to desired dimensions after processing.
    : r_c - rows, cols (y, x)
    : pmax - maximum power to control maximum array size.
    :    2**pmax + 1 = N
    :    pairs (pmax, N): (10, 1025), (11, 2049), (12, 4097), (13, 8193)
    """
    max_size = max(r_c)
    for p in range(1, pmax):
        N = (2**p) + 1
        if max_size <= N:
            return N, p
    return 2**pmax + 1, pmax


# ---- main portion of the algorithm -----------------------------------------
#
def d_s(twoPow=2, low=0, high=1, cols=1, rows=1, corner=None, r=0, img=False):
    """Diamond square algorithm
    : twoPow - 2**(x) this is the power to specify, returns rows/cols
    :    (x, N) - (2, 5), (3, 9), (4, 17), (5, 33), (6, 65), (7, 129),
    :             (8, 257), (9, 513), (10, 1025), (11, 2049)
    : high/low - min, max of the array
    : cols/rows - for replication in both directions
    : corner - specify a corner if you want values to seed the algorithm
    :          [UL, UR, LL, LR]
    : r - roughness between 0 and 1
    : img - True to show image, False otherwise
    :Reference:
    :---------
    :  https://raw.githubusercontent.com/buckinha/DiamondSquare/master
    :        /DiamondSquare.py
    """
    # seed the random number generator
    np.random.seed(None)
    if (r < 0.) or (r > 1.):
        raise ValueError("roughness, r, outside the acceptable range or 0-1")
    # ---- array, size (N, N), filled will NaN, with corner specification ----
    p = twoPow
    N = 2**twoPow + 1
    arr = np.full((N, N), np.nan, dtype='float', order='C')
    # ---- seed the corners
    c_pnts = np.random.uniform(low, high, (2, 2))
    if corner is None:
        arr[0::N-1, 0::N-1] = c_pnts
    elif len(corner) == 4:
        c_pnts = np.asarray(corner).reshape(2, 2)
        arr[0::N-1, 0::N-1] = c_pnts
    else:
        arr[0::N-1, 0::N-1] = c_pnts
    # ---- run the algorithm
    print("Input template array with 4 corners defined...\n{}".format(arr))
    for i in range(p):
        r = r  # roughness**i  0**0 = 1 !!!! watch it
        step_size = (N-1) // 2**(i)
        _diamond_(arr, step_size, r)  # ---- diamond step
        #print("power di {}...\narray...\n{}".format(i, arr))
        _square_(arr, step_size, r)   # ---- square step
        #print("power sq {}...\narray...\n{}".format(i, arr))
    # ---- determine whether mirroring or graphing is desired ----
    if cols == 2:
        arr = np.c_[arr, np.fliplr(arr)]
    if cols == 3:
        arr = np.c_[arr, np.fliplr(arr), arr]
    if rows == 2:
        arr = np.r_[arr, np.flipud(arr)]
    if img:
        plt.imshow(arr, cmap=plt.cm.gist_earth)  # gist_earth, Blues, oceans)
        plt.axis('off')
        plt.show()
    return arr


def _diamond_(arr, step_size, r):
    """Diamond step first.  Calculate to locate the diamond corners for
    :  filling.
    """
    def shift_di(a, i, j, hs, r):
        """hs - halfstep
        :defines the midpoint displacement for the diamond step
        """
        T, L, B, R = [i-hs, j-hs, i+hs, j+hs]
        ave = (a[T, L] + a[T, R] + a[B, L] + a[B, R])/4.0
        ran = np.random.uniform(-r, r)
        return (1.0 - ran) * ave
    #
    # ---- main diamond section
    hs = step_size//2
    x_steps = range(hs, arr.shape[0], step_size)
    y_steps = range(hs, arr.shape[1], step_size)  # x_steps[:]
    for i in x_steps:
        for j in y_steps:
            if np.isnan(arr[i, j]):  # == -1.0:  # ** checks for -1
                arr[i, j] = shift_di(arr, i, j, hs, r)
    # ---- end ----


def _square_(arr, step_size, r):
    """Step the square with half-steps for shift_sq
    :Shift the square and determine whether 3 or 4 values are used in the
    :calculation for the average based on whether the square is on an edge.
    :
    """
    def shift_sq(arr, i, j, hs, r):
        """ hs - half step """
        div = 0
        sum_ = 0
        T, L, B, R = [i-hs, j-hs, i+hs, j+hs]
        if i - hs >= 0:
            sum_ += arr[T, j]  # top
            div += 1
        if i + hs < arr.shape[0]:
            sum_ += arr[B, j]  # bottom
            div += 1
        if j - hs >= 0:
            sum_ += arr[i, L]  # left
            div += 1
        if j + hs < arr.shape[0]:
            sum_ += arr[i, R]  # right
            div += 1
        avg = sum_ / div
        ran = np.random.uniform(-r, r)
        return (1.0 - ran) * avg
    #
    hs = step_size//2
    # ---- vertical step
    steps_x_vert = range(hs, arr.shape[0], step_size)
    steps_y_vert = range(0, arr.shape[1], step_size)
    # ---- horizontal step
    steps_x_horiz = range(0, arr.shape[0], step_size)
    steps_y_horiz = range(hs, arr.shape[1], step_size)
    for i in steps_x_horiz:
        for j in steps_y_horiz:
            arr[i, j] = shift_sq(arr, i, j, hs, r)
    for i in steps_x_vert:
        for j in steps_y_vert:
            arr[i, j] = shift_sq(arr, i, j, hs, r)
    # ---- end ----


def _demo(n, img=False):
    np.random.RandomState(1)
    # N = 2**n + 1
    low = 0
    high = 1.0
#    c_pnts = [.25, .75, .5, 1.0]
    c_pnts = [.25, 1.0, .25, 1.0]
    a = d_s(twoPow=n, low=low, high=high, cols=1, rows=1, corner=c_pnts,
            r=0, img=img)
    return a



# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
#    a = _demo(n=1, img=True)  # img... show plot if True
"""
c_pnts = [.25, .75, .5, 1.0]  # used for peak
c_pnts = [.25, 1.0, .25, 1.0] # used for vertical wave
np.save("C:/Temp/ds_1.npy", a)

a = d_s(twoPow=10, low=0, high=10., cols=100, rows=100, corner=c_pnts,
            r=0, img=True)
a0 = np.fliplr(a)
aa0 = np.c_[a, a0]
a1 = np.flipud(aa0)
aa0a1 = np.r_[aa0, a1]
np.save("c:/temp/ds_2.npy", aa0a1)
plt.imshow(aa0a1, cmap=plt.cm.gist_earth)
plt.axis('off')
plt.show()

#
pnts = np.random.randint(100, 1900, size=(20,2))
squares = [np.array([p, p+[50, 50]]) for p in pnts]
zeros = np.zeros_like(aa0a1)
for i in squares:
    zeros[i[0][0]:i[1][0], i[0][1]:i[1][1]] = 10.

bergs = zeros + aa0a1
plt.imshow(bergs, cmap=plt.cm.gist_earth)
plt.axis('off')
plt.show()
"""
"""
d0 = np.diff(bergs, 1, axis=0)
d1 = np.diff(bergs, 1, axis=1)
d0d = np.where(d0 > 1, d0, 0)
d1d = np.where(d1 > 1, d1, 0)
d0d1 = d0d[:2049, :2049] + d1d[:2049, :2049]
"""
