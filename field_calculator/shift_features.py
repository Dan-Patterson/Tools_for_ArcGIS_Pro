# -*- coding: utf-8 -*-
"""
shift_features
==============

Script :   shift_features.py

Author :   Dan_Patterson@carleton.ca

Modified : 2018-07-25
"""

import arcpy

def shift_features(in_features, x_shift=None, y_shift=None):
    """
    Shifts features by an x and/or y value. The shift values are in
    the units of the in_features coordinate system.

    Parameters:
    in_features: string
        An existing feature class or feature layer.  If using a
        feature layer with a selection, only the selected features
        will be modified.

    x_shift: float
        The distance the x coordinates will be shifted.

    y_shift: float
        The distance the y coordinates will be shifted.
    """

    with arcpy.da.UpdateCursor(in_features, ['SHAPE@XY']) as cursor:
        for row in cursor:
            cursor.updateRow([[row[0][0] + (x_shift or 0),
                               row[0][1] + (y_shift or 0)]])

    return