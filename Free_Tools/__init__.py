# -*- coding: utf-8 -*-
"""
==================
npgeom.__init__.py
==================

Author :
    Dan_Patterson@carleton.ca

Modified : 2019-08-14
    Creation date during 2019 as part of ``arraytools``.

Purpose : Tools for working with point and poly features as an array class
    Requires npGeo to implement the array geometry class.

Notes
-----

Import suggestion and package properties and methods.  The Geo class in npGeo
provides the base class for this package.  It is based on the numpy ndarray.

>>> import npgeom as npg

>>> npg.npg_io.__all__
... ['poly2array', 'load_geojson', 'Arrays_to_Geo', 'Geo_to_arrays',
...  'array_ift', '_make_nulls_', 'getSR', 'fc_composition', 'fc_data',
...  'fc_geometry', 'fc_shapes', 'getSR', 'shape_to_K', 'array_poly',
...  'geometry_fc', 'prn_q', '_check', 'prn_tbl', 'prn_geo']

>>> npg.npGeo.__all__
... ['Geo', 'Update_Geo']

>>> npg.npg_helpers.__all__
... ['_angles_', '_area_centroid_', '_area_part_', '_ch_', '_ch_scipy',
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
# import numpy as np
# pylint: disable=W0611   # unused import
# pyflake: disable=F401

from . import (
        npGeo, npg_io, npg_geom, npg_table, smallest_circle,  _tests_
        )

from .npGeo import (Geo, Update_Geo)

from .npg_io import (
        poly2array, load_geojson, geojson_Geo, fc_json, Arrays_to_Geo,
        Geo_to_arrays, array_ift, _make_nulls_, getSR, shape_to_K,
        fc_composition, fc_data, fc_geometry, fc_shapes,
        array_poly, geometry_fc, prn_q, _check, prn_tbl, prn_geo
        )

from .npg_geom import (
        _area_centroid_, _area_part_, _o_ring_, _angles_, _ch_scipy_,
        _ch_simple_, _ch_, _dist_along_, _percent_along_, _pnts_on_line_,
        _densify_by_dist_,  _polys_to_segments_, _polys_to_unique_pnts_,
        _simplify_lines_, _tri_pnts_
        )

from .npg_table import (col_stats, crosstab_tbl, crosstab_rc, crosstab_array)


__all_io__ = [
        'Arrays_to_Geo', 'Geo_to_arrays', '_check', '_make_nulls_',
        'array_ift', 'array_poly', 'fc_composition', 'fc_data',
        'fc_geometry', 'fc_shapes', 'geometry_fc', 'getSR', 'getSR',
        'load_geojson', 'poly2array', 'prn_geo', 'prn_q', 'prn_tbl',
        'shape_to_K'
        ]
__all_geo__ = [
        'Geo', 'Update_Geo', '_angles_', '_area_centroid_', '_area_part_',
        '_ch_', '_ch_scipy_', '_ch_simple_', '_dist_along_', '_nan_split_',
        '_o_ring_', '_percent_along_', '_pnts_on_line_', '_densify_by_dist_',
        '_polys_to_segments_', '_polys_to_unique_pnts_', '_tri_pnts_',
        'unique_attributes'
        ]

__all_helpers__ = [
        '_angles_', '_area_centroid_', '_area_part_', '_ch_', '_ch_scipy_',
        '_ch_simple_', '_nan_split_', '_o_ring_', '_pnts_on_line_',
        '_polys_to_segments_', '_polys_to_unique_pnts_', '_simplify_lines_'
        ]

__all__ = __all_io__ + __all_geo__ + __all_helpers__
# __all__.sort()
print("\nUsage...\n  import npgeom as npg")
