# -*- coding: UTF-8 -*-
"""
:Script:  movepnts.py
:Author:  Dan.Patterson@carleton.ca
:Modified: 2017-04-06
:Notes:
:- arcpy.da.FeatureClassToNumPyArray(in_table, field_names, {where_clause},
:                                    {spatial_reference}, {explode_to_points},
:                                    {skip_nulls}, {null_value})
:- arcpy.da.NumPyArrayToFeatureClass(in_array, out_table, shape_fields,
:                                    {spatial_reference})
:- create multipart polygons: https://geonet.esri.com/message/461451
: our house relative to 0,0 in MTM9
: xy_shift = [341886,5023462]
:
:Spatial reference
: NAD_1983_CSRS_MTM_9
: WKID: 2951 Authority: EPSG
: in_fc = r'C:\GIS\Table_tools\Table_tools.gdb\polygon_demo'
: dx = 2
: dy = 2
: out_fc = r'C:\GIS\Table_tools\Table_tools.gdb\bb'
"""
import sys
import numpy as np
import arcpy

arcpy.env.overwriteOutput = True


def fc_info(in_fc):
    """basic feature class information"""
    desc = arcpy.Describe(in_fc)
    SR = desc.spatialReference
    shp_field = desc.ShapeFieldName
    OIDField = desc.OIDFieldName
    return shp_field, OIDField, SR

# ---- input parameters ----
in_fc = sys.argv[1]
dx = float(sys.argv[2])
dy = float(sys.argv[3])
out_fc = sys.argv[4]
xy_shift = np.array([dx, dy], dtype="<f8")
shp_field, OIDField, SR = fc_info(in_fc)
# ---- convert to array, shift and return ----
# Apparently, there can be problems writing directly to a featureclass
# so, write to in_memory changing the required field names, then copy out
arr = arcpy.da.FeatureClassToNumPyArray(in_fc, "*", "", SR, True)
arr[shp_field] = arr[shp_field] + xy_shift
nms = ['Feat_id', 'XYs'] + [i for i in arr.dtype.names[2:]]
arr.dtype.names = nms
temp_out = "in_memory/temp2"
arcpy.da.NumPyArrayToFeatureClass(arr, temp_out, ['XYs'])
arcpy.CopyFeatures_management(temp_out, out_fc)

# ---- the end ----
