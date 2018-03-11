# -*- coding: utf-8 -*-
"""
:polyline_demo.py

"""
import numpy as np
import numpy.lib.recfunctions as rfn
import arcpy

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.2f}'.format}
np.set_printoptions(edgeitems=3, linewidth=80, precision=3,
                    suppress=True, threshold=100, formatter=ft)


def group_pnts(a, key_fld='ID', keep_flds=['X', 'Y', 'Z']):
    """Group points for a feature that has been exploded to points by
    :  arcpy.da.FeatureClassToNumPyArray.
    :Requires:
    :---------
    : a - a structured array, assuming ID, X, Y, {Z} and whatever else
    :   - the array is assumed to be sorted... which will be the case
    :Returns:
    :--------
    : see np.unique descriptions below
    :References:
    :-----------
    :  https://jakevdp.github.io/blog/2017/03/22/group-by-from-scratch/
    :  http://esantorella.com/2016/06/16/groupby/
    :Notes:
    :------ split-apply-combine
    """
    returned = np.unique(a[key_fld],           # the unique id field
                         return_index=True,    # first occurrence index
                         return_inverse=True,  # indices needed to remake array
                         return_counts=True)   # number in each group
    uniq, idx, inv, cnt = returned
    from_to = [[idx[i-1], idx[i]] for i in range(1, len(idx))]
    subs = [a[keep_flds][i:j] for i, j in from_to]
    groups = [sub.view(dtype='float').reshape(sub.shape[0], -1)
              for sub in subs]
    return groups


def e_leng(a):
    """Length/distance between points in an array using einsum
    : Inputs
    :   a list/array coordinate pairs, with ndim = 3 and the
    :   Minimum shape = (1,2,2), eg. (1,4,2) for a single line of 4 pairs
    :   The minimum input needed is a pair, a sequence of pairs can be used.
    : Returns
    :   d_arr  the distances between points forming the array
    :   length the total length/distance formed by the points
    :-----------------------------------------------------------------------
    """
    def cal(diff):
        """ perform the calculation
        :diff = g[:, :, 0:-1] - g[:, :, 1:]
        : for 4D
        : d = np.einsum('ijk..., ijk...->ijk...', diff, diff).flatten() or
        :   = np.einsum('ijkl, ijkl->ijk', diff, diff).flatten()
        : d = np.sum(np.sqrt(d)
        """
        d_arr = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
        d_leng = d_arr.flatten()
        length = np.sum(d_leng)
        return length, d_leng
    # ----
    diffs = []
    a = np.atleast_2d(a)
    if a.shape[0] == 1:
        return 0.0
    if a.ndim == 2:
        a = np.reshape(a, (1,) + a.shape)
    if a.ndim == 3:
        diff = a[:, 0:-1] - a[:, 1:]
        length, d_leng = cal(diff)
        diffs.append(d_leng)
    if a.ndim == 4:
        length = 0.0
        for i in range(a.shape[0]):
            diff = a[i][:, 0:-1] - a[i][:, 1:]
            leng, d_leng = cal(diff)
            diffs.append(d_leng)
            length += leng
    return length, diffs


def report(subs):
    """print out the data"""
    for i in range(len(subs)):
        total, segs = e_leng(subs[i])
        frmt = """
        3D distances along polyline {}
        Segment distances
        {}
        Total = sum of segments ? {}
        """
        print(frmt.format(total, segs, total == np.sum(segs)))


# ---- inputs ---
fc = r'C:\GIS\Geometry_projects\polyline_demo\polyline_demo.gdb\polylines'

flds = ['OID@', 'SHAPE@X', 'SHAPE@Y', 'SHAPE@Z']
SR = arcpy.Describe(fc).spatialreference
a = arcpy.da.FeatureClassToNumPyArray(fc,
                                      field_names=flds,
                                      spatial_reference=SR,
                                      explode_to_points=True)
dt = [('ID', '<i4'), ('X', '<f8'), ('Y', '<f8'), ('Z', '<f8')]
a.dtype = dt                  # simplify the dtype
      # get the unique id values
groups = group_pnts(a, key_fld='ID', keep_flds=['X', 'Y', 'Z'])
report(groups)  # print the results

# ids = np.unique(a['ID'])
# p_lines = [a[a['ID'] == i] for i in ids]                 # collect polylines
# subs = [np.c_[p['X'], p['Y'], p['Z']] for p in p_lines]  # stack coordinates
"""

xyz_names = list(a.dtype.names[1:])
a[xyz_names]
b = a[xyz_names].view(dtype='float').reshape(a.shape[0], -1)

array 'a' must be sorted, which it will be when you explode a feature to
points
uniq, idx, inv, cnt = np.unique(a['ID'],             # the unique id field
                                return_index=True,   # first occurrence index
                                return_inverse=True, # indices needed to remake
                                return_counts=True)  # number in each group

pnt_fc = r'C:\GIS\Geometry_projects\polyline_demo\polyline_demo.gdb\pnts'
dz = np.array([ln[-1]['Z'] - ln[0]['Z'] for ln in lines])
dx = np.array([ln[-1]['X'] - ln[0]['X'] for ln in lines])
dy = np.array([ln[-1]['Y'] - ln[0]['Y'] for ln in lines])
dist = np.sqrt(dx**2 + dy**2)
print("2D distances first point to last point \n{}".format(dist))

dt = [('ID', '<i8'), ('X', '<f8'), ('Y', '<f8'), ('Z', '<f8')]
ids = np.array([1,1,1,1,2,2,2,2])
x_s = np.array([0,1,2,3,3,2,1,0])
y_s = np.array([0,1,2,3,3,2,1,0])
z_s = np.arange(8, dtype='float')
a = np.zeros((8,), dtype=dt)
a['ID'] = ids
a['X'] = x_s
a['Y'] = y_s
a['Z'] = z_s
ids = np.unique(ids)
lines = [a[a['ID'] == i] for i in ids]

"""