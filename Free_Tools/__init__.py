# -*- coding: utf-8 -*-
"""
==================
npgeom.__init__.py
==================

Author :
    Dan_Patterson@carleton.ca

Modified : 2019-09-06
    Creation date during 2019 as part of ``arraytools``.

Purpose : Tools for working with point and poly features as an array class
    Requires npGeo to implement the array geometry class.

Notes
-----

Import suggestion and package properties and methods.  The Geo class in npGeo
provides the base class for this package.  It is based on the numpy ndarray.

>>> import npgeom as npg

>>> npg.npg_io.__all__
... ['poly2array', 'load_geojson', 'arrays_to_Geo', 'Geo_to_arrays',
...  'array_ift', '_make_nulls_', 'getSR', 'fc_composition', 'fc_data',
...  'fc_geometry', 'fc_shapes', 'getSR', 'shape_to_K', 'array_poly',
...  'geometry_fc', 'prn_q', '_check', 'prn_tbl', 'prn_geo']

>>> npg.npGeo.__all__
... ['Geo', 'Update_Geo']

>>> npg.npg_helpers.__all__
... ['_angles_', '_area_centroid_', '_ch_', '_ch_scipy',
...  '_ch_simple_', '_nan_split_', '_o_ring_', '_pnts_on_line_',
...  '_polys_to_segments_', '_polys_to_unique_pnts_', '_simplify_lines_']

**Import options for arcpy functions**

>>> import arcgisscripting as ags
... ['ContingentFieldValue', 'ContingentValue', 'DatabaseSequence', 'Describe',
... 'Domain', 'Editor', 'ExtendTable', 'FeatureClassToNumPyArray',
... 'InsertCursor', 'ListContingentValues', 'ListDatabaseSequences',
... 'ListDomains', 'ListFieldConflictFilters', 'ListReplicas', 'ListSubtypes',
... 'ListVersions', 'NumPyArrayToFeatureClass', 'NumPyArrayToTable', 'Replica',
... 'SearchCursor', 'TableToNumPyArray', 'UpdateCursor', 'Version', 'Walk'...]

>>> ags.da.FeatureClassToNumPyArray(...)  # useage

Arcpy methods and properties needed::

    arcpy.Point, arcpy.Polyline, arcpy.Polygon, arcpy.Array
    arcpy.ListFields
    arcpy.management.CopyFeatures
    arcpy.da.Describe
    arcpy.da.InsertCursor
    arcpy.da.SearchCursor
    arcpy.da.FeatureClassToNumPyArray
"""
# pylint: disable=unused-import
# pylint: disable=W0611
import numpy as np

from . import (
        npGeo, npg_io, npg_geom, npg_table, npg_create, npg_analysis,
        smallest_circle, _tests_
        )  # noqa

from .npGeo import (Geo, Update_Geo, dirr, geo_info)  # noqa

from .npg_io import (
        poly2array, load_geojson, geojson_Geo, fc_json, arrays_to_Geo,
        Geo_to_arrays, array_ift, _make_nulls_, getSR, shape_K,
        fc_composition, fc_data, fc_geometry, fc_shapes, array_poly,
        geometry_fc, prn_q, _check, prn_tbl, prn_geo,
        shape_properties, flatten_to_points
        )  # noqa

from .npg_geom import (
        _area_centroid_, _angles_, _rotate_, _ch_scipy_,
        _ch_simple_, _ch_, _dist_along_, _percent_along_, _pnts_on_line_,
        _pnt_on_segment_, _polys_to_unique_pnts_,
        _simplify_lines_, _tri_pnts_,
        )  # noqa

from .npg_table import (
        col_stats, crosstab_tbl, crosstab_rc, crosstab_array
        )  # noqa

from .npg_analysis import (
        closest_n, distances, not_closer, n_check, n_near, n_spaced,
        intersects, knn, knn0, mst, connect, concave
        )  # noqa

__all_io__ = [
        '__all_io__',
        'arrays_to_Geo', 'Geo_to_arrays', '_check', '_make_nulls_',
        'array_ift', 'array_poly', 'fc_composition', 'fc_data',
        'fc_geometry', 'fc_shapes', 'geometry_fc', 'getSR', 'getSR',
        'load_geojson', 'poly2array', 'prn_geo', 'prn_q', 'prn_tbl',
        'shape_to_K'
        ]  # noqa
__all_geo__ = [
        '__all_geo__',
        'Geo', 'Update_Geo', 'dirr', 'geo_info'
        ]  # noqa

__all_geom__ = [
        '__all_geom__',
        '_angles_', '_area_centroid_', '_ch_', '_ch_scipy_', '_ch_simple_',
        '_dist_along_', '_percent_along_', '_pnt_on_poly_', '_pnt_on_segment_',
        '_pnts_in_poly_', '_pnts_on_line_', '_polys_to_unique_pnts_',
        '_rotate_', '_simplify_lines_', '_tri_pnts_', 'ft', 'np', 'p_o_p'
        ]  # noqa

__all_analysis__ = [
        '__all_analysis__',
        'closest_n', 'distances', 'not_closer', 'n_check', 'n_near',
        'n_spaced', 'intersects', 'knn', 'knn0', '_dist_arr_', '_e_dist_',
        'mst', 'connect', 'concave'
        ]  # noqa

__all_table__ = [
        '__all_table__',
        'crosstab_tbl', 'crosstab_rc', 'crosstab_array', 'col_stats',
        'group_stats'
        ]  # noqa
args = [__all_io__, __all_geo__, __all_geom__, __all_analysis__,
        __all_table__]
__all__ = np.concatenate([np.asarray(a) for a in args]).tolist()

# __all__.sort()


print("\nUsage...\n  import npgeom as npg")
