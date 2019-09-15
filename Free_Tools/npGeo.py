# -*- coding: utf-8 -*-
"""
=====
npGeo
=====

Script : npGeo.py
    A geometry class and methods based on numpy.

Author :
    Dan_Patterson@carleton.ca

Modified : 2019-09-06
    Initial creation period 2019-05.

Purpose : geometry tools
    A numpy geometry class, its properties and methods.

Notes
-----
**Class instantiation**

Quote from Subclassing ndarrays::

    As you can see, the object can be initialized in the __new__ method or the
    __init__ method, or both, and in fact ndarray does not have an __init__
    method, because all the initialization is done in the __new__ method.


**General notes**

The Geo class returns a 2D array of points which may consist of single or
multipart shapes with or without inner rings (holes).

The methods defined  in the Geo class allow one to operate on the parts of the
shapes separately or in combination.  Since the coordinate data are represented
as an Nx2 array, it is sometimes easier to perform calculations on the dataset
all at once using numpy ``nan`` functions.  For example, to determine the
minimum for the whole dataset:

>>> np.nanmin(Geo, axis=0)

All null points (nan, nan) are omitted from the calculations.

**Working with np.ndarray and Geo class**

** To check the difference between the np.ndarray and Geo class, use... **

>>> type(g)  # ...  npgeom.npGeo.Geo
>>> arr_set = set(dir(g.base))
>>> geo_set = set(dir(g))
>>> sorted(list(geo_set.difference(arr_set)))
... ['FT', 'IDs', 'IFT', 'Info', 'K', 'N', 'X', 'XY', 'Y', 'Z', '__dict__',
...  '__module__', 'aoi_extent', 'aoi_rectangle', 'areas', 'bit_ids', 'bits',
...  'bounding_circles', 'centers', 'centroids', 'close_polylines',
...  'common_segments', 'convex_hulls', 'densify_by_distance',
...  'extent_rectangles', 'extents', 'fill_holes', 'get', 'holes_to_shape',
...  'info', 'is_clockwise', 'is_convex', 'is_multipart', 'lengths', 'maxs',
...  'means', 'min_area_rect', 'mins', 'moveto_origin',
...  'multipart_to_singlepart', 'od_pairs', 'outer_rings', 'part_cnt', 'parts',
...  'pnt_cnt', 'pnt_on_poly', 'point_info', 'polygon_angles',
...  'polygons_to_polylines', 'polyline_angles', 'polylines_to_polygons',
...  'polys_to_points', 'polys_to_segments', 'pull', 'rotate', 'shapes',
...  'shift', 'split_by', 'translate', 'triangulate', 'unique_segments']

>>> Geo.__dict__.keys()
... dict_keys(
... ['__module__', '__doc__', '__new__', '__array_finalize__',
...  '__array_wrap__', 'shapes', 'parts', 'bits', 'bit_ids', 'part_cnt',
...  'pnt_cnt', 'areas', 'centers', 'centroids', 'lengths', 'aoi_extent',
...  'aoi_rectangle', 'extents', 'extent_rectangles', 'maxs', 'mins', 'means',
...  'get', 'pull', 'outer_rings', 'point_info', 'split_by', 'is_clockwise',
...  'is_convex', 'is_multipart', 'polyline_angles', 'polygon_angles',
...  'moveto_origin', 'shift', 'translate', 'rotate', 'bounding_circles',
...  'convex_hulls', 'min_area_rect', 'triangulate', 'fill_holes',
...  'holes_to_shape', 'multipart_to_singlepart', 'od_pairs',
...  'polylines_to_polygons', 'polygons_to_polylines', 'polys_to_points',
...  'close_polylines', 'densify_by_distance', 'pnt_on_poly',
...  'polys_to_segments', 'common_segments', 'unique_segments', 'info',
...  '__dict__'])

**Useage of methods**

``g`` is a Geo instance with 2 shapes.  Both approaches yield the same results.

>>> Geo.centers(g)
array([[ 5.  , 14.93],
       [15.5 , 15.  ]])
>>> g.centers()
array([[ 5.  , 14.93],
       [15.5 , 15.  ]])

References
----------
`Subclassing ndarrays
<https://docs.scipy.org/doc/numpy/user/basics.subclassing.html>`_.

**Sample file**

Saved in the arraytools folder and on GitHub::

    fname = 'C:/Git_Dan/arraytools/Data/geo_array.npz'
    npzfiles = np.load(fname)   # ---- the Geo, I(ds)F(rom)T(o) arrays
    npzfiles.files              # ---- will show ==> ['s2', 'IFT']
    s2 = npzfiles['s2']         # ---- slice by name from the npz file to get
    IFT = npzfiles['IFT']       #      each array
"""
# pylint: disable=R0902  # Too many instance attributes
# pylint: disable=R0904  # pylint issue
# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect
# pylint: disable=W0201  # attribute defined outside __init__... none in numpy
# pylint: disable=W0212  # Access to a protected member...
# pylint: disable=W0621  # redefining name

import sys
from textwrap import dedent
import numpy as np

from numpy.lib.recfunctions import structured_to_unstructured as stu
from numpy.lib.recfunctions import unstructured_to_structured as uts

import npg_geom as geom
import smallest_circle as sc

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.1f}'.format}
np.set_printoptions(edgeitems=5, linewidth=120, precision=2, suppress=True,
                    threshold=200, formatter=ft)

script = sys.argv[0]  # print this should you need to locate the script

FLOATS = np.typecodes['AllFloat']
INTS = np.typecodes['AllInteger']
NUMS = FLOATS + INTS
TwoPI = np.pi*2.

__all__ = ['Geo', 'Update_Geo', 'dirr']


# ===========================================================================
#
class Geo(np.ndarray):
    """Geo class

    Create a Geo array based on the numpy ndarray.  The class focus is on
    geometry properties and methods.  Construction of geometries can be made
    using File Geodatabase Featureclasses (Esri) or GeoJSON data as the source
    of the base geometries.

    Construction from an ndarray, IFT, Kind and optional Info.

    Parameters
    ----------
    arr : array-like
        A 2D array sequence of points with shape (N, 2).
    IFT : array-like
        Defines, the I(d)F(rom)T(o) values identifying object parts if
        ``arr`` represents polylines or polygons.  Shape (N, 3) required.
    Kind : integer
        Points (0), polylines/lines (1) and polygons (2).
    Info : string (optional)
        Optional information if needed.

    Notes
    -----
    The IDs can either be 0-based or in the case of some data-types,
    1-based.  No assumption is made about IDs being sequential.  In the case
    of featureclass geometry, the OID@ property is read.  For other geometries,
    provide ID values as appropriate.

    Point, polyline, polygon features represented as numpy ndarrays.
    The required inputs are created using ``fc_geometry(in_fc)`` or
    ``Arrays_to_Geo``.

    Attributes
    ----------
    Normal ndarray parameters including shape, ndim, dtype.

    shapes :
        The points for polyline, polygons.
    parts :
        Multipart shapes and/or outer and inner rings for holes.
    bits :
        The final divisions to individual bits constituting the shape.
    is_multipart :
        Array of booleans
    part_cnt :
        ndarray of ids and counts.
    pnt_cnt :
        ndarray of ids and counts.
    geometry properties :
        Areas, centers, centroids and lengths are properties and not methods.

    Comments
    --------
    You can use ``Arrays_to_Geo`` to produce the required 2D array from lists
    of array-like objects of the same dimension, or a single array.
    The IFT will be derived from breaks in the sequence and/or the
    presence of null points within a sequence.

    >>> import npgeom as npg
    >>> g = npg.Geo(a, IFT)
    >>> g.__dict__.keys()
    dict_keys(['IDs', 'FT', 'IFT', 'K', 'Info', 'N', 'X', 'Y', 'XY', 'Z'])
    >>> sorted(g.__dict__.keys())
    ['FT', 'IDs', 'IFT', 'Info', 'K', 'N', 'X', 'XY', 'Y', 'Z']

    See Also
    --------
    __init__.py :
        General comments about the package.
    npg_io.py :
        Import and conversion routines for the Geo class.
    npg_geom :
        Methods/functions for working with the Geo class or used by it.
    npg_table :
        Methods/functions associated with tabular data.
    """
    # ----
    def __new__(cls, arr=None, IFT=None, Kind=2, Info="Geo array"):
        """
        See notes above on construction.
        """
        msg = Geo.__doc__
        arr = np.asarray(arr)
        IFT = np.asarray(IFT)
        cond = [(arr.ndim != 2),
                (IFT.ndim != 2), (IFT.shape[-1] != 3),
                (Kind not in (0, 1, 2))]
        if all(cond):
            print(dedent(msg))
            return None
        # ----
        self = arr.view(cls)     # view as Geo class
        self.IFT = IFT
        self.IDs = IFT[:, 0]
        self.FT = IFT[:, 1:]
        self.K = Kind
        self.Info = Info
        # --- other properties
        self.N = len(np.unique(self.IDs))  # sample size, unique shapes
        if self.shape[1] >= 2:             # X,Y and XY initialize
            self.X = arr[:, 0]
            self.Y = arr[:, 1]
            self.XY = arr[:, :2]
        if self.shape[1] >= 3:  # add Z, although not implemented
            self.Z = arr[:, 2]  # directly, but kept for future additions
        else:
            self.Z = None
        return self

    def __array_finalize__(self, src_arr):
        """The new object... this is where housecleaning takes place for
        explicit, view casting or new from template...
        ``src_arr`` is either None, any subclass of ndarray including our own
        (words from documentation) OR another instance of our own array.
        You can use the following with a dictionary instead of None:

        >>> self.info = getattr(obj,'info',{})
        """
        if src_arr is None:
            return
        self.IFT = getattr(src_arr, 'IFT', None)
        self.IDs = getattr(src_arr, 'IDs', None)
        self.FT = getattr(src_arr, 'FT', None)
        self.K = getattr(src_arr, 'K', None)
        self.Info = getattr(src_arr, 'Info', None)
        self.N = getattr(src_arr, 'N', None)
        self.X = getattr(src_arr, 'X', None)
        self.Y = getattr(src_arr, 'Y', None)
        self.XY = getattr(src_arr, 'XY', None)

    def __array_wrap__(self, out_arr, context=None):
        """Wrap it up"""
        return np.ndarray.__array_wrap__(self, out_arr, context)

    # ------------------------------------------------------------------------
    """To do list:
    buffer
    clip
    contains
    cut
    dominant direction
    """
    # ------------------- End of class definition ----------------------------
    # ---- basic shape properties

    @property
    def shapes(self):
        """Subdivide the array's singlepart or multipart shapes as either an
        object array or ndarray of points.
        """
        uniq = np.unique(self.IDs)
        c = [self.FT[self.IDs == i].ravel() for i in uniq]
        c1 = [(min(i), max(i)) for i in c]
        return np.array([np.asarray(self[f:t]) for f, t in c1])  # .squeeze()

    @property
    def parts(self):
        """Deconstruct the 2D array into its parts, generally returning an
        object array.  The reverse is np.vstack(self).

        Formally: return np.asarray([(self[f:t]) for f, t in self.FT]) but with
        additions to add the FT and IFT properties.
        """
        xy = self.base
        if xy is None:
            xy = self.XY.view(np.ndarray)
        return np.asarray([xy[f:t] for f, t in self.FT])  # np.split equiv.

    @property
    def bits(self):
        """Deconstruct the 2D array then parts of a piece if a piece contains
        multiple parts.  Keeps all rings and removes nan.
        """
        out = []
        prts = self.parts
        for ply in prts:
            s = np.isnan(ply[:, 0])
            if np.any(s):
                w = np.where(s)[0]
                ss = np.split(ply, w)
                for s in ss:   # ---- keep all lines
                    out.append(s[~np.isnan(s[:, 0])])  # ss[0])
            else:
                out.append(ply)
        return np.asarray(out)

    @property
    def bit_ids(self):
        """Return the ID values for each bit in a shape.  If there are multiple
        parts or rings in a part, then the shape ID is repeated for each
        occurrence.
        """
        out_ids = []
        prts = self.parts
        ids = self.IDs
        for cnt, ply in enumerate(prts):
            s = np.isnan(ply[:, 0])
            out_ids.append(ids[cnt])
            if np.any(s):
                w = np.where(s)[0]
                num = len(w)
                out_ids.extend(np.repeat(ids[cnt], num))
        return np.asarray(out_ids)

    @property
    def part_cnt(self):
        """Part count for shapes. Returns IDs and count array"""
        return np.vstack(np.unique(self.IDs, return_counts=True)).T

    @property
    def pnt_cnt(self):
        """Point count for shape bits excluding null points."""
        b_ids = self.bit_ids
        return np.array([(b_ids[i], len(p[~np.isnan(p[:, 0])]))
                         for i, p in enumerate(self.bits)])

    # ---- areas, centrality, lengths/perimeter for polylines/polygons
    #
    @property
    def areas(self):
        """Area for the sub arrays using _e_area for the calculations.  Uses
        ``_area_part_`` to calculate the area.
        """
        def _area_part_(a):
            """Mini e_area, used by areas and centroids"""
            x0, y1 = (a.T)[:, 1:]
            x1, y0 = (a.T)[:, :-1]
            e0 = np.einsum('...i,...i->...i', x0, y0)
            e1 = np.einsum('...i,...i->...i', x1, y1)
            return np.nansum((e0 - e1)*0.5)
        # ----
        if self.K != 2:
            print("Polygons required")
            return None
        subs = [_area_part_(i) for i in self.parts]   # call _area_part_
        ids = self.IDs
        totals = np.bincount(ids, weights=subs)[ids]  # weight by IDs' area
        return totals

    @property
    def centers(self):
        """Return the center of an array's unique non-nan points.  Each part
        of multipart features are determined separately.
        """
        return np.asarray([np.nanmean(np.unique(c, axis=0), axis=0)
                           for c in self.parts])

    @property
    def centroids(self):
        """Centroid of the polygons.  Uses ``_area_centroid_`` to calculate
        values for each shape part.  The centroid is weighted by area for
        multipart features.
        """
        # ----
        def weighted(x_y, I, areas):
            """Weighted coordinate by area, x_y is either the x or y."""
            w = x_y * areas                # area weighted x or y
            w1 = np.bincount(I, w)[I]      # weight divided by bin size
            ar = np.bincount(I, areas)[I]  # areas per bin
            return w1/ar
        # ----
        if self.K != 2:
            print("Polygons required.")
            return None
        centr = []
        areas = []
        ids = self.IDs
        uni = np.unique(ids)
        for ID in uni:
            parts_ = self.FT[self.IDs == ID]
            out = np.asarray([np.asarray(self.XY[p[0]:p[1]]) for p in parts_])
            for prt in out:
                area, cen = geom._area_centroid_(prt)  # ---- determine both
                centr.append(cen)
                areas.append(area)
        centr = np.asarray(centr)
        areas = np.asarray(areas)
        xs = weighted(centr[:, 0], ids, areas)
        ys = weighted(centr[:, 1], ids, areas)
        return np.array(list(zip(xs, ys)))

    @property
    def lengths(self):
        """Polyline lengths or polygon perimeter."""
        def _cal(a):
            """Perform the calculation, mini-e_leng."""
            diff = a[:-1] - a[1:]
            return np.nansum(np.sqrt(np.einsum('ij,ij->i', diff, diff)))
        # ----
        if self.K not in (1, 2):
            print("Polyline/polygon representation is required.")
            return None
        lengs = [_cal(i) for i in self.parts]
        ids = self.IDs
        totals = np.bincount(ids, weights=lengs)[ids]
        return np.asarray(totals)

    # ---- methods section ---------------------------------------------------
    # ---- extents and extent shapes
    #
    def aoi_extent(self):
        """The full geographic extent of the dataset `aoi` (area of interest).
        """
        return np.concatenate((np.nanmin(self.XY, axis=0),
                               np.nanmax(self.XY, axis=0)))

    def aoi_rectangle(self):
        """The polygon bounds derived from the aoi_extent.
        """
        bounds = self.aoi_extent()
        L, B, R, T = bounds
        return np.array([[L, B], [L, T], [R, T], [R, B], [L, B]])

    def extents(self, by_part=False):
        """Extents are returned as L(eft), B(ottom), R(ight), T(op).
        """
        def _extent_(i):
            """Extent of a sub-array in an object array"""
            return np.concatenate((np.nanmin(i, axis=0), np.nanmax(i, axis=0)))
        # ----
        if self.N == 1:
            by_part = True
        return np.asarray([_extent_(i) for i in self.split_by(by_part)])

    def extent_rectangles(self, by_part=False):
        """Return extent polygons for all shapes.  Points are ordered clockwise
        from the bottom left, with the first and last points the same.
        Requires an Advanced license in Pro for equivalent functionality.

        See Also
        --------
        ``aoi_extent`` and ``aoi_rectangles``
        """
        ext_polys = []
        for ext in self.extents(by_part):
            L, B, R, T = ext
            poly = np.array([[L, B], [L, T], [R, T], [R, B], [L, B]])
            ext_polys.append(poly)
        return np.asarray(ext_polys)

    # ---- maxs, mins, means, pnts for all features
    #
    def maxs(self, by_part=False):
        """Maximums per feature or part."""
        if len(self.part_cnt) == 1:
            return np.asarray(np.nanmax(self, axis=0))
        return np.asarray([np.nanmax(i, axis=0)
                           for i in self.split_by(by_part)])

    def mins(self, by_part=False):
        """Minimums per featureor part."""
        if len(self.part_cnt) == 1:
            return np.asarray(np.nanmin(self, axis=0))
        return np.asarray([np.nanmin(i, axis=0)
                           for i in self.split_by(by_part)])

    def means(self, by_part=False, remove_dups=True):
        """Mean per feature or part, optionally keep duplicates."""
        if len(self.part_cnt) == 1:
            chunks = [self]
        else:
            chunks = self.split_by(by_part)
        if remove_dups:
            chunks = [np.unique(i, axis=0) for i in chunks]
        return np.asarray([np.nanmean(i, axis=0) for i in chunks])

    # ---- slicing, sampling equivalents
    #
    def get(self, ID, asGeo=True):
        """Return a Geo or ndarray associated with the feature ID.  The ID must
        exist, otherwise None is returned.

        Parameters
        ----------
        ID : integer
            A single integer value.
        asGeo : Boolean
            True, returns an updated Geo array.  False returns an ndarray or
            object array.
        """
        if not isinstance(ID, (int)):
            print("Integer ID is required, see ``pull`` for multiple values.")
            return None
        if ID not in np.unique(self.IDs):
            print("ID not in possible values.")
            return None
        f_t = self.IFT[self.IDs == ID]
        s_e = f_t.ravel()
        shp = self[s_e[1]: s_e[-1]]
        if asGeo:
            return Geo(shp, IFT=f_t, Kind=self.K)
        return np.asarray(shp)

    def pull(self, ID_list, asGeo=True):
        """Pull multiple shapes, in the order provided.  The original IDs are
        kept but the point sequence is altered to reflect the new order.

        Parameters
        ----------
        ID_list : array-like
            A list, tuple or ndarray of ID values identifying which features
            to pull from the input.
        asGeo : Boolean
            True, returns an updated Geo array.  False returns an ndarray or
            object array.

        Notes
        -----
        >>> a.pull(np.arange(3:8))  # get shapes over a range of values
        >>> a.pull([1, 3, 5])  # get selected shapes
        """
        if not isinstance(ID_list, (list, tuple, np.ndarray)):
            print("An array/tuple/list of IDs are required, see ``get``.")
            return None
        if not np.all([a in self.IDs for a in ID_list]):
            print("Not all required IDs are in the list of provided")
            return None
        parts_ = np.vstack([self.IFT[self.IDs == i] for i in ID_list])
        vals = [np.asarray(self.XY[p[1]:p[2]]) for p in parts_]
        if asGeo:
            ids = parts_[:, 0]
            too = np.cumsum([len(i) for i in vals])
            frum = np.concatenate(([0], too))
            IFT = np.array(list(zip(ids, frum, too)))
            vals = np.vstack(vals)
            return Geo(vals, IFT, self.K)
        return np.asarray(vals)

    def split_by(self, by_part=False):
        """Split points by shape or by parts for each shape. **keep for now**
        Use self.parts or self.shapes directly."""
        return self.parts if by_part else self.shapes

    def outer_rings(self, asGeo=False):
        """Collect the outer ring of a polygon shape.  Returns a list of
        ndarrays or optionally a new Geo array.
        """
        if self.K != 2:
            print("Polygons required")
            return None
        a_2d = []
        id_too = []
        for ift in self.IFT:
            i, f, t = ift
            ar = self[f:t]
            p = geom._o_ring_(ar)           # ---- call ``_o_ring_``
            a_2d.append(np.asarray(p))
            id_too.append([i, len(p)])
        info = "{} outer_rings".format(str(self.Info))
        a_2d = np.vstack(a_2d)
        id_too = np.asarray(id_too)
        if asGeo:
            return Update_Geo(a_2d, 2, id_too, info)  # ---- update Geo
        return np.asarray(a_2d)

    # ==== Some of these methods use functions from npg_geom ================
    #
    # ---- **is** section, condition/case checking, kept to a minimum
    def is_clockwise(self, is_closed_polyline=False):
        """Utilize `shoelace` area calculation to determine whether polygon
        rings are clockwise or not.  If the geometry represent a closed-loop
        polyline, then set the `is_closed_polyline` to True.  Validity of the
        geometry is not checked.
        """
        def _area_part_(a):
            """Mini e_area, used by areas and centroids"""
            x0, y1 = (a.T)[:, 1:]
            x1, y0 = (a.T)[:, :-1]
            e0 = np.einsum('...i,...i->...i', x0, y0)
            e1 = np.einsum('...i,...i->...i', x1, y1)
            return np.nansum((e0 - e1)*0.5)
        # ----
        msg = "Polygons or closed-loop polylines are required."
        if self.K not in (1, 2):
            print(msg)
            return None
        if self.K == 1:
            if not is_closed_polyline:
                print(msg)
                return None
        ids = self.bit_ids
        cw = np.asarray([1 if _area_part_(i) > 0. else 0
                         for i in self.bits])
        return uts(np.asarray(list(zip(ids, cw))), names=['IDs', 'Clockwise'])

    def is_convex(self, by_part=True):
        """Return True for convex, False for concave.  Holes are excluded,
        multipart shapes are included by setting by_part=True.
        """
        def _x_(a):
            """cross product"""
            dx, dy = a[0] - a[-1]
            if np.allclose(dx, dy):    # closed loop
                a = a[:-1]
            ba = a - np.roll(a, 1, 0)   # vector 1
            bc = a - np.roll(a, -1, 0)  # vector 2
            return np.cross(ba, bc)
        # ----
        if self.K != 2:
            print("Polygons are required.")
            return None
        chunks = self.split_by(by_part)
        check = []
        for p in chunks:
            p = geom._o_ring_(p)  # ---- run ``_o_ring_
            check.append(_x_(p))  # cross-product
        return np.array([np.all(np.sign(i) >= 0) for i in check])

    def is_multipart(self, as_structured=False):
        """For each shape, returns whether it has multiple parts.  An ndarray
        is returned with the first column being the shape number and the second
        is coded as 1 for True and 0 for False.
        """
        partcnt = self.part_cnt
        w = np.where(partcnt[:, 1] > 1, 1, 0)
        ids = np.unique(self.IDs)
        arr = np.array(list(zip(ids, w)))
        if as_structured:
            dt = np.dtype([('IDs', '<i4'), ('Parts', '<i4')])
            return uts(arr, dtype=dt)
        return arr

    # ---- Methods to determine angles, convexity and other properties that
    #      enable you use methods by part or by whole.
    #
    def polyline_angles(self, fromNorth=False):
        """Polyline/segment angles."""
        s = self.polys_to_segments()
        dxy = s[:, -2:] - s[:, :2]
        ang = np.degrees(np.arctan2(dxy[:, 1], dxy[:, 0]))
        if fromNorth:
            ang = np.mod((450.0 - ang), 360.)
        return ang

    def polygon_angles(self, inside=True, in_deg=True):
        """Sequential 3 point angles from a poly* shape.  The outer ring for
        each part is used.  see ``_angles_`` and ``_o_ring_``.
        """
        # --- run _angle_ and _o_ring_ on the chunks
        return [geom._angles_(p, inside, in_deg) for p in self.parts]

    # ---- return altered geometry
    #
    def moveto_origin(self):
        """Shift the dataset so that the origin is the lower-left corner.
        see also ``shift`` and ``translate``.
        """
        dx, dy = np.nanmin(self.XY, axis=0)
        return Geo(self.XY + [-dx, -dy], self.IFT)

    def shift(self, dx=0, dy=0):
        """see ``translate``"""
        return Geo(self.XY + [dx, dy], self.IFT)

    def translate(self, dx=0, dy=0):
        """Move/shift/translate by dx, dy to a new location."""
        return Geo(self.XY + [dx, dy], self.IFT)

    def rotate(self, about_center=True, angle=0.0, clockwise=False):
        """Rotate shapes about their center, if center=True, otherwise rotate
        about the X/Y axis origin (0, 0).
        """
        if clockwise:
            angle = -angle
        angle = np.radians(angle)
        c, s = np.cos(angle), np.sin(angle)
        R = np.array(((c, s), (-s, c)))
        chunks = self.shapes
        out = []
        if about_center:
            uniqs = []
            for chunk in chunks:
                _, idx = np.unique(chunk, True, axis=0)
                uniqs.append(chunk[np.sort(idx)])
            cents = [np.nanmean(i, axis=0) for i in uniqs]
            for i, chunk in enumerate(chunks):
                ch = np.einsum('ij,jk->ik', chunk-cents[i], R) + cents[i]
                out.append(ch)
        else:
            cent = np.nanmean(self, axis=0)
            for chunk in chunks:
                ch = np.einsum('ij,jk->ik', chunk-cent, R) + cent
                out.append(ch)
        info = "{} rotate".format(self.Info)
        return Update_Geo(np.vstack(out), self.K, self.IFT, Info=info)

    # ---- changes to geometry, derived from geometry, convex_hulls, minimum
    #    area bounding rectangle  **see also** extent properties above
    #
    def bounding_circles(self, angle=5, return_xyr=False):
        """Bounding circles for features.

        Parameters
        ----------
        angle : number
            Angles to form n-gon.  A value of 10 will yield 36 point circle.
        xyr : tuple (optional)
            The circle center x, y and radius.
        """
        shps = self.shapes
        shps = [s[~np.isnan(s[:, 0])] for s in shps]
        xyr = [sc.small_circ(s) for s in shps]
        circs = []
        for vals in xyr:
            x, y, r = vals
            circs.append(sc.circle_mini(r, angle, x, y))
        if return_xyr:
            return xyr, circs
        return circs

    def convex_hulls(self, by_part=False, threshold=50):
        """Convex hull for shapes.  Calls ``_ch_`` to control method used.

        by_part : boolean
            False for whole shape.  True for shape parts if present.
        threshold : integer
            Points... less than threshold uses simple CH method, greater than,
            uses scipy.
        """
        # ----
        shps = self.parts if by_part else self.shapes
        # ---- run convex hull, _ch_, on point groups
        ch_out = [geom._ch_(s, threshold) for s in shps]
        for i, c in enumerate(ch_out):  # check for closed
            if np.all(c[0] != c[-1]):
                ch_out[i] = np.vstack((c, c[0]))
        return ch_out

    def min_area_rect(self, as_structured=False):
        """Determines the minimum area rectangle for a shape represented
        by a list of points.  If the shape is a polygon, then only the outer
        ring is used.  This is the MABR... minimum area bounding rectangle.
       """
        def _extent_area_(a):
            """Area of an extent polygon."""
            LBRT = np.concatenate((np.nanmin(a, axis=0), np.nanmax(a, axis=0)))
            dx, dy = np.diff(LBRT.reshape(2, 2), axis=0).squeeze()
            return dx * dy, LBRT

        def _extents_(a):
            """Extents are returned as L(eft), B(ottom), R(ight), T(op)."""
            def _sub_(i):
                """Extent of a sub-array in an object array"""
                return np.concatenate((np.nanmin(i, axis=0),
                                       np.nanmax(i, axis=0)))
            p_ext = [_sub_(i) for i in a]
            return np.asarray(p_ext)
        # ----
        chs = self.convex_hulls(False, 50)
        ang_ = [geom._angles_(i) for i in chs]
        xt = _extents_(chs)
        cent_ = np.c_[np.mean(xt[:, 0::2], axis=1),
                      np.mean(xt[:, 1::2], axis=1)]
        rects = []
        for i, p in enumerate(chs):
            # ---- np.radians(np.unique(np.round(ang_[i], 2))) # --- round
            uni_ = np.radians(np.unique(ang_[i]))
            area_old, LBRT = _extent_area_(p)
            for angle in uni_:
                c, s = np.cos(angle), np.sin(angle)
                R = np.array(((c, s), (-s, c)))
                ch = np.einsum('ij,jk->ik', p - cent_[i], R) + cent_[i]
                area_, LBRT = _extent_area_(ch)
                Xmin, Ymin, Xmax, Ymax = LBRT
                vals = [area_, Xmin, Ymin, Xmax, Ymax]
                if area_ < area_old:
                    area_old = area_
                    Xmin, Ymin, Xmax, Ymax = LBRT
                    vals = [area_, Xmin, Ymin, Xmax, Ymax]   # min_area,
            rects.append(vals)
        rects = np.asarray(rects)
        if as_structured:
            dt = np.dtype([('Rect_area', '<f8'), ('Xmin', '<f8'),
                           ('Ymin', '<f8'), ('Xmax', '<f8'), ('Ymax', '<f8')])
            return uts(rects, dtype=dt)
        return rects

    def triangulate(self, as_polygon=True):
        """Delaunay triangulation for point groupings."""
        out = [geom._tri_pnts_(s) for s in self.shapes]
        kind = 2 if as_polygon else 1
        return Update_Geo(out, K=kind)

    #
    # ---- conversions -------------------------------------------------------
    #
    def fill_holes(self):
        """Fill holes in polygon shapes.  Returns a Geo class."""
        a_2d = []
        id_too = []
        if self.K < 2:
            print("Polygon geometry required.")
            return None
        for i, p in enumerate(self.parts):
            nan_check = np.isnan(p[:, 0])  # check the Xs for nan
            if np.any(nan_check):          # split at first nan
                w = np.where(np.isnan(p[:, 0]))[0]
                p = np.split(p, w)[0]      # keep the outer ring
            a_2d.append(np.array(p))
            id_too.append([i, len(p)])
        a_2d = np.vstack(a_2d)
        id_too = np.asarray(id_too)
        info = "{} fill_holes".format(self.Info)
        return Update_Geo(a_2d, 2, id_too, info)  # run update

    def holes_to_shape(self):
        """Return holes in polygon shapes.  Returns a Geo class or None."""
        a_2d = []
        id_too = []
        if self.K < 2:
            print("Polygon geometry required.")
            return None
        for i, p in enumerate(self.parts):  # work with the parts
            nan_check = np.isnan(p[:, 0])   # check the Xs for nan
            if np.any(nan_check):           # split at first nan
                w = np.where(np.isnan(p[:, 0]))[0]
                p_new = np.split(p, w)[1]     # keep the outer ring
                p_new = p_new[1:][::-1]
                a_2d.append(np.array(p_new))
                id_too.append([i, len(p_new)])
        if not a_2d:  # ---- if empty
            return None
        id_too = np.asarray(id_too)
        return Update_Geo(a_2d, 2, id_too)  # run update

    def multipart_to_singlepart(self, info=""):
        """Convert multipart shapes to singleparts and return a new Geo array.
        """
        ift = self.IFT
        data = np.vstack(self.parts)
        ift[:, 0] = np.arange(len(self.parts))
        return Geo(data, IFT=ift, Kind=self.K, Info=info)

    def od_pairs(self):
        """Construct origin-destination pairs for traversing around the
        perimeter of polygons, along polylines or between point sequences.

        Returns
        -------
        An object array of origin-destination pairs is returned.

        See Also
        --------
        polys_to_segments
        """
        return np.asarray([np.c_[p[:-1], p[1:]] for p in self.bits])

    def polylines_to_polygons(self):
        """Return a polygon Geo type from a polyline Geo.  It is assumed that
        the polylines form closed-loops, otherwise use ``close_polylines``.
        """
        if self.K == 2:
            print("Already classed as a polygon.")
            return self
        polygons = self.copy()
        polygons.K = 2
        return polygons

    def polygons_to_polylines(self):
        """Return a polyline Geo type from a polygon Geo.
        """
        if self.K == 1:
            print("Already classed as a polyline.")
            return self
        polylines = self.copy()
        polylines.K = 1
        return polylines

    def polys_to_points(self, keep_order=True, as_structured=False):
        """Convert all feature vertices to an ndarray of unique points.
        NaN's are removed.  Optionally, retain point order."""
        if as_structured:
            return geom._polys_to_unique_pnts_(self, as_structured=True)
        a = self[~np.isnan(self.X)]
        uni, idx = np.unique(a, True, axis=0)
        if keep_order:
            uni = a[np.sort(idx)]
        return np.asarray(uni[~np.isnan(uni[:, 0])])

    def close_polylines(self, out_kind=1):
        """Attempt to produce closed-loop polylines (1) or polygons (2)
        from polylines.  Multipart features are converted to single part.
        """
        polys = []
        for s in self.bits:  # shape as bits
            if len(s) > 2:
                if np.all(s[0] == s[-1]):
                    polys.append(s)
                else:
                    polys.append(np.concatenate((s, s[..., :1, :]), axis=0))
        return Update_Geo(polys, K=out_kind)

    def densify_by_distance(self, spacing=1):
        """Densify poly features by a specified distance.  Converts multipart
        to singlepart features during the process.
        Calls ``_pnts_on_line_`` for Geo bits.
        """
        polys = [geom._pnts_on_line_(a, spacing) for a in self.bits]
        return Update_Geo(polys, K=self.K)

    def densify_by_percent(self, percent=50):
        """Densify poly features by a percentage for each segment.  Converts
        multipart to singlepart features during the process.
        Calls ``_percent_along``
        """
        polys = [geom._pnts_on_line_(a, spacing=percent, is_percent=True)
                 for a in self.bits]
        return Update_Geo(polys, K=self.K)

    def pnt_on_poly(self, by_dist=True, val=1):
        """Point on polyline/polygon by distance or percent. Emulates
        `arcpy Polyline class, positionAlongLine (value, {use_percentage})
        <https://pro.arcgis.com/en/pro-app/arcpy/classes/polyline.htm>`_.
        """
        if by_dist:
            r = [geom._dist_along_(a, dist=val) for a in self.bits]
            return Update_Geo(r, K=0)
        r = [geom._dist_along_(a, percent=val) for a in self.bits]
        return Update_Geo(r, K=0)

    # ---- segments for poly* boundaries
    #
#    def segments(self):
#        """Segment count for bits.  Derived from bit_ids.  Returns the bit ID
#        and the segments, which is 1 less than the number of points per part.
#        """
#        b_vals = self.bits
#        b_ids = self.bit_ids
#        segs = [[b_ids[i], len(b) - 1] for i, b in enumerate(b_vals)]
#        return np.asarray(segs)

    def polys_to_segments(self, as_basic=True, as_3d=False):
        """Segment poly* structures into o-d pairs from start to finish.

        Parameters
        ----------
        as_basic : boolean
            True returns the basic od pairs as an Nx5 array in the form
            [X_orig', Y_orig', 'X_orig', 'Y_orig', 'Length'] as an ndarray.
            If False, the content is returned as a structured array with the
            same content.
        as_3d : boolean
            True, the point pairs are returned as a 3D array in the form
            [[X_orig', Y_orig'], ['X_orig', 'Y_orig']], without the distances.

        Notes
        -----
        Any row containing np.nan is removed since this would indicate that the
        shape contains the null_pnt separator.
        Use ``prn_tbl`` if you want to see a well formatted output.
        """
        if self.K not in (1, 2):
            print("Poly* features required")
            return None
        b_vals = self.bits
        fr_to = np.vstack([np.hstack((b[:-1], b[1:])) for b in b_vals])
        # ---- shortcut to 3d from-to representation
        if as_3d:
            fr_to = fr_to[:, :4]
            s0, s1 = fr_to.shape
            return fr_to.reshape(s0, s1//2, s1//2)
        # ---- ndarray or structured array section
        dist = (np.sqrt(np.sum((fr_to[:, :2] - fr_to[:, 2:4])**2, axis=1)))
        if as_basic:
            return np.hstack((fr_to, dist.reshape(-1, 1)))
        # ---- add bit ids to the output array
        b_ids = self.bit_ids
        segs = np.asarray([[b_ids[i], len(b) - 1]
                           for i, b in enumerate(b_vals)])
        s_ids = np.concatenate([np.repeat(i[0], i[1]) for i in segs])
        fr_to = np.hstack((fr_to, s_ids.reshape(-1, 1), dist.reshape(-1, 1)))
        dt = np.dtype([('X_orig', 'f8'), ('Y_orig', 'f8'),
                       ('X_dest', 'f8'), ('Y_dest', 'f8'),
                       ('Orig_id', 'i4'), ('Length', 'f8')])
        return uts(fr_to, dtype=dt)

    def common_segments(self):
        """Return the common segments in poly features.  Result is an array of
        from-to pairs of points.  ft, tf pairs are evaluated to denote common
        and duplicates.
        """
        h = self.polys_to_segments()
        if h is None:
            return None
        h_0 = uts(h)
        names = h_0.dtype.names
        h_1 = h_0[list(names[-2:] + names[:2])]
        idx = np.isin(h_0, h_1)
        common = h_0[idx]
        return stu(common)

    def unique_segments(self):
        """Return the unique segments in poly features as an array of
        from-to pairs of points.
        """
        h = self.polys_to_segments()
        if h is None:
            return None
        h_0 = uts(h)
        names = h_0.dtype.names
        h_1 = h_0[list(names[-2:] + names[:2])]
        idx0 = ~np.isin(h_0, h_1)
        uniq0 = h_0[idx0]
        uniq1 = h_0[~idx0]
        uniq01 = np.hstack((uniq0, uniq1))
        return stu(uniq01)

    # ---- info section
    #
    def info(self, prn=True, start=0, end=50):
        """Convert an IFT array to full information.

        Parameters
        ----------
        prn : boolean
            If True, the top and bottom ``rows`` will be printed.
            If False, the information will be returned and one can use
            ``prn_tbl`` for more control over the tabular output.
        start, end : integers
            The start to end locations within the geo-array to print or view.

        Notes
        -----
        Point count will include any null_pnts used to separate inner and
        outer rings.

        To see the data structure, use ``prn_geo``.
        """
        ift = self.IFT
        ids = ift[:, 0]
        frum = ift[:, 1]
        too = ift[:, 2]
        uni, cnts = np.unique(ids, return_counts=True)
        part_count = np.concatenate([np.arange(i) for i in cnts])
        pnts = np.array([len(p) for p in self.parts])
        id_len2 = np.stack((ids, part_count, pnts, frum, too), axis=-1)
        dt = np.dtype({
            'names': ['IDs', 'Part', 'Points', 'From_pnt', 'To_pnt'],
            'formats': ['i4', 'i4', 'i4', 'i4', 'i4']})
        IFT_2 = uts(id_len2, dtype=dt)
        frmt = "-"*14 + \
            "\nShapes :{:>6.0f}\nParts  :{:>6.0f}" + \
            "\nPoints :{:>6.0f}\n  min  :{:>6.0f}\n  med  :{:>6.0f}" + \
            "\n  max  :{:>6.0f}"
        shps = len(uni)  # ---- zero-indexed, hence add 1
        _, cnts = np.unique(IFT_2['Part'], return_counts=True)
        p0 = np.sum(cnts)
        p3 = np.sum(IFT_2['Points'])
        p4 = np.min(IFT_2['Points'])
        p5 = np.median(IFT_2['Points'])
        p6 = np.max(IFT_2['Points'])
        msg = dedent(frmt).format(shps, p0, p3, p4, p5, p6)
        if prn:
            frmt = "{:>8} "*5
            start, end = sorted([abs(int(i)) if isinstance(i, (int, float))
                                 else 0
                                 for i in [start, end]])
            print(msg)
            print(frmt.format(*IFT_2.dtype.names))
            N = IFT_2.shape[0]
            for i in range(min(N, end)):
                print(frmt.format(*IFT_2[i]))
            # prn_tbl(IFT_2, rows)
        else:
            return IFT_2

    def point_info(self, by_part=True, with_null=False):
        """Point count by feature or parts of feature.

        Parameters
        ----------
        by part: boolean
            True for each feature part or False for the whole feature.
        with_null : boolean
            True, to include nan/null points.
        """
        chunks = self.split_by(by_part)
        if with_null:
            return self.FT[:, 1] - self.FT[:, 0]
        return np.array([len(i[~np.isnan(i[:, 0])]) for i in chunks])
    #
    # ----------------End of class definition-

#
# ==== update Geo array, or create one from a list of arrays ================


def Update_Geo(a_2d, K=None, id_too=None, Info=None):
    """Create a new Geo array from a list of arrays.

    Parameters
    ----------
    a_2d : list/tuple/array
        Some form of nested 2D array-like structure that can be stacked.
    K : integer
        Points (0), polylines (1) or polygons (2).
    id_too : array-like
        If None, then the structure will be created.
    Info : text (optional)
        Provide any information that will help in identifying the array.

    Returns
    -------
    A new Geo array is returned given the inputs.
    """
    if not isinstance(a_2d, (np.ndarray, list, tuple)):
        return None
    if K not in (0, 1, 2):
        print("Output type not specified, or not in (0, 1, 2).")
        return None
    if id_too is None:
        if K == 0:
            ids = np.arange(0, len(a_2d))
            frum = ids
            too = np.arange(1, len(a_2d) + 1)
        else:
            id_too = [(i, len(a)) for i, a in enumerate(a_2d)]
            a_2d = np.vstack(a_2d)
            id_too = np.array(id_too)
            ids = id_too[:, 0]
            too = np.cumsum(id_too[:, 1])
            frum = np.concatenate(([0], too))
    else:
        a_2d = np.vstack(a_2d)
        ids = id_too[:, 0]
        too = np.cumsum(id_too[:, 1])
        frum = np.concatenate(([0], too))
    IFT = np.array(list(zip(ids, frum, too)))
    return Geo(a_2d, IFT, K, Info)


# ----------------------------------------------------------------------
# ---- (3) dirr ... code section ... -----------------------------------------
#
def dirr(obj, colwise=False, cols=3, prn=True):
    """The non-numpy version of dirr
    """
    from itertools import zip_longest as zl
    if isinstance(obj, np.ndarray):
        a = ['... Geo class ...']
        a.extend(sorted(list(set(dir(obj)).difference(set(dir(np.ndarray))))))
        a.extend(['... np_geom ...'] + sorted(geom.__all__))
    else:
        a = dir(obj)
    w = max([len(i) for i in a])
    frmt = (("{{!s:<{}}} ".format(w)))*cols
    csze = len(a) / cols  # split it
    csze = int(csze) + (csze % 1 > 0)
    if colwise:
        a_0 = [a[i: i+csze] for i in range(0, len(a), csze)]
        a_0 = list(zl(*a_0, fillvalue=""))
    else:
        a_0 = [a[i: i+cols] for i in range(0, len(a), cols)]
    if hasattr(obj, '__module__'):
        args = ["-"*70, obj.__module__, obj.__class__]
    else:
        args = ["-"*70, type(obj), "py version"]
    txt_out = "\n{}\n| dir({}) ...\n|    {}\n-------".format(*args)
    cnt = 0
    for i in a_0:
        cnt += 1
        txt = "\n  ({:>03.0f})  ".format(cnt)
        frmt = (("{{!s:<{}}} ".format(w)))*len(i)
        txt += frmt.format(*i)
        txt_out += txt
    if prn:
        print(txt_out)
    else:
        return txt_out
# ===========================================================================
#
if __name__ == "__main__":
    """optional location for parameters"""
    # print("\n...npGeo imported")
    diff = sorted(list(set(dir(Geo)).difference(set(dir(np.ndarray)))))
    """
    ['__dict__', '__module__', 'aoi_extent', 'aoi_rectangle', 'areas',
     'bit_ids', 'bits', 'bounding_circles', 'centers', 'centroids',
     'close_polylines', 'common_segments', 'convex_hulls',
     'densify_by_distance', 'densify_by_percent', 'extent_rectangles',
     'extents', 'fill_holes', 'get', 'holes_to_shape', 'info', 'is_clockwise',
     'is_convex', 'is_multipart', 'lengths', 'maxs', 'means', 'min_area_rect',
     'mins', 'moveto_origin', 'multipart_to_singlepart', 'od_pairs',
     'outer_rings', 'part_cnt', 'parts', 'pnt_cnt', 'pnt_on_poly',
     'point_info', 'polygon_angles', 'polygons_to_polylines',
     'polyline_angles', 'polylines_to_polygons', 'polys_to_points',
     'polys_to_segments', 'pull', 'rotate', 'shapes', 'shift', 'split_by',
     'translate', 'triangulate', 'unique_segments']
    """
