# -*- coding: utf-8 -*-
"""
npgeom.__init__.py
"""

import numpy as np
from . import fc_geo_io
from . import npGeo
from .fc_geo_io import (
        poly2array, array_ift, _make_nulls_, getSR, fc_composition, fc_data,
        fc_geometry, fc_shapes, array_poly, geometry_fc, prn_q, _check,
        prn_tbl, prn_geo
        )
from .npGeo import (
        Geo, Arrays_to_Geo, Geo_to_arrays, Update_Geo, _angles_,
        _area_centroid_, _area_part_, _ch_, _o_ring_, _pnts_on_line_,
        _poly_segments_, _polys_to_unique_pnts_, _simplify_lines_
        )

__all__ = fc_geo_io.__all__ + npGeo.__all__
__all__.sort()
print("\nUsage  import npgeom as npg")
