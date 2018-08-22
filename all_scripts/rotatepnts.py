# -*- coding: UTF-8 -*-
"""
Script :  movepnts.py

Author :  Dan.Patterson@carleton.ca

Modified : 2018-07-21

Notes:
-----
- arcpy.da.FeatureClassToNumPyArray(in_table, field_names, {where_clause},
                                    {spatial_reference}, {explode_to_points},
                                    {skip_nulls}, {null_value})
- arcpy.da.NumPyArrayToFeatureClass(in_array, out_table, shape_fields,
                                    {spatial_reference})

- create multipart polygons: https://geonet.esri.com/message/461451
 our house relative to 0,0 in MTM9
 xy_shift = [341886,5023462]
:
:Spatial reference
: NAD_1983_CSRS_MTM_9
: WKID: 2951 Authority: EPSG
: in_fc = r'C:\GIS\Table_tools\Table_tools.gdb\polygon_demo'
: dx = 2
: dy = 2
: out_fc = r'C:\GIS\Table_tools\Table_tools.gdb\bb'
C:\GIS\A_Tools_scripts\PointTools\Point_tools.gdb\std_dist_center
"""
import sys
import numpy as np
import arcpy
from arcpytools import fc_info, tweet

script = sys.argv[0]

arcpy.env.overwriteOutput = True

def trans_rot(a, angle):
    """Translate and rotate and array of points about the point cloud origin.

    Requires:
    ---------
    a : array
        2d array of x,y coordinates
    theta : double
        angle in degrees in the range -180. to 180        
    Returns:
    --------
    Points rotated about the origin and translated back.
    
    Notes:
    ------
    np.einsum('ij,kj->ik', a - cent, R).T  =  np.dot(a - cent, R.T).T
    ik does the rotation in einsum
    
    R = np.array(((c, s), (-s,  c)))  - clockwise about the origin
    """
    cent = a.mean(axis=0)
    angle = np.radians(angle)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array(((c, s), (-s,  c)))
    return  np.einsum('ij,kj->ik', a - cent, R) + cent


# ---- main section ----
#
if len(sys.argv) == 1:
    testing = True
    fc = "/Point_tools.gdb/std_dist_center"
    flder = "/".join(script.split("/")[:-2])
    in_fc = flder + fc
    angle = 30.0
    out_fc = flder + "/Point_tools.gdb/rot_std_dist2"

else:
    testing = False
    in_fc = sys.argv[1]
    angle = float(sys.argv[2])
    out_fc = sys.argv[3]
    

# ---- convert to array, shift and return ----
# Apparently, there can be problems writing directly to a featureclass
# so, write to in_memory changing the required field names, then copy out
#
shp_fld, oid_fld, shp_type, SR = fc_info(in_fc)
arr = arcpy.da.FeatureClassToNumPyArray(in_fc, "*", "", SR, True)
a = arr[shp_fld]
new_pnts = trans_rot(a, angle)
nms = ['Feat_id', 'XYs'] + [i for i in arr.dtype.names[2:]]
arr.dtype.names = nms
arr['XYs'] = new_pnts
arcpy.da.NumPyArrayToFeatureClass(arr, out_fc, ['XYs'])

msg = """
-------------------------------------
Input points..... {}
Rotation angle... {}
Output points.... {}
"""
tweet(msg.format(in_fc, angle, out_fc))
# ---- the end ----
