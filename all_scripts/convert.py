# -*- coding: utf-8 -*-
"""
:Script:   convert.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-02-22
:Purpose:

:Requires:
:
:Notes:
:
:Functions:

:References:

"""

import sys
import numpy as np
import arcpy
from arcpytools import fc_info, tweet, fc_array

'''
def tweet(msg):
    """Produce a message for both arcpy and python
    : msg - a text message
    """
    m = "\n{}\n".format(msg)
    arcpy.AddMessage(m)
    print(m)
    return None


def fc_info(in_fc):
    """basic feature class information"""
    desc = arcpy.Describe(in_fc)
    SR = desc.spatialReference      # spatial reference object
    shp_fld = desc.shapeFieldName   # FID or OIDName, normally
    oid_fld = desc.OIDFieldName     # Shapefield ...
    return shp_fld, oid_fld, SR



def fc_array(in_fc, flds, allpnts):
    """Convert featureclass to an ndarray...with optional fields besides the
    :FID/OIDName and Shape fields.
    :Syntax: read_shp(input_FC,other_flds, explode_to_points)
    :   input_FC    shapefile
    :   other_flds   "*", or specific fields ['FID','Shape','SomeClass', etc]
    :   see:  FeatureClassToNumPyArray, ListFields for more information
    """
    out_flds = []
    shp_fld, oid_fld, SR = fc_info(in_fc)  # get the base information
    fields = arcpy.ListFields(in_fc)       # all fields in the shapefile
    if flds == "":                   # return just OID and Shape field
        out_flds = [oid_fld, shp_fld]     # FID and Shape field required
    elif flds == "*":                # all fields
        out_flds = [f.name for f in fields]
    else:
        out_flds = [oid_fld, shp_fld]
        for f in fields:
            if f.name in flds:
                out_flds.append(f.name)
    frmt = "\nRunning 'fc_array' with...\n{}\nFields...{}\nSR...{}"
    args = [in_fc, out_flds, SR.name]
    msg = frmt.format(*args)
    tweet(msg)
    a = arcpy.da.FeatureClassToNumPyArray(in_fc, out_flds, "", SR, allpnts)
    # out it goes in array format
    return a, out_flds, SR
'''

in_fc = sys.argv[1]
fld_names = sys.argv[2]
pnt_type = sys.argv[3]
p4 = sys.argv[4]
if pnt_type == 'all':
    allpnts = True
else:
    allpnts = False
a, out_flds, SR = fc_array(in_fc, fld_names, allpnts)  # , fld_names, SR
frmt = """
Input featureclass...\n{}\nSR...{}\nFields...\n{}\nAll points? {}
Array...\n{!r:}\n
"""
msg = frmt.format(in_fc, SR.name, out_flds, allpnts, a)
tweet(msg)
arcpy.GetMessages()
# del in_fc, fld_names, a, out_flds, SR


# -------------------------------------------------------------------------
if __name__ == "__main__":
    """ x  """
    pass
#    a, fld_names, SR = _demo()
#    a, out_flds, SR = _demo()
