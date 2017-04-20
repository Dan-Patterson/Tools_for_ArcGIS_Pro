# -*- coding: UTF-8 -*-
"""
:Script:   .py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-02-27
:Purpose:  tools for working with numpy arrays
:Useage:
:
:References:
:
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
import numpy as np
import arcpy
# from arcpytools import array_fc, array_struct, tweet

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}

np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


def tweet(msg):
    """Produce a message for both arcpy and python
    : msg - a text message
    """
    m = "\n{}\n".format(msg)
    arcpy.AddMessage(m)
    print(m)
    print(arcpy.GetMessages())
#    return None


def fc_info(in_fc):
    """basic feature class information"""
    desc = arcpy.Describe(in_fc)
    SR = desc.spatialReference      # spatial reference object
    shp_fld = desc.shapeFieldName   # FID or OIDName, normally
    oid_fld = desc.OIDFieldName     # Shapefield ...
    return shp_fld, oid_fld, SR


def array_struct(a, fld_names=['X', 'Y'], dt=['<f8', '<f8']):
    """Convert an array to a structured array
    :Requires:
    :--------
    :  a - an ndarray with shape at least (N, 2)
    :  dt = dtype class
    :  names - names for the fields
    """
    dts = [(fld_names[i], dt[i]) for i in range(len(fld_names))]
    z = np.zeros((a.shape[0],), dtype=dts)
    names = z.dtype.names
    for i in range(a.shape[1]):
        z[names[i]] = a[:, i]
    return z


def array_fc(a, out_fc, fld_names, SR):
    """array to featureclass/shapefile...optionally including all fields
    :  syntax: array_fc(a, out_fc, fld_names, SR)
    :  see: NumpyArrayToFeatureClass, ListFields for information and options
    :  out_fc:   featureclass/shapefile... complete path
    :  fld_names:  is the Shapefield name ie ['Shape'] or ['X', 'Y's]
    """
    if arcpy.Exists(out_fc):
        arcpy.Delete_management(out_fc)
    arcpy.da.NumPyArrayToFeatureClass(a, out_fc, fld_names, SR)
    return out_fc


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
    frmt = """
    Inside....
    Running 'fc_array' with...\n{}\nFields...{}\nAll pnts...{}\nSR...{}
    """
    args = [in_fc, out_flds, allpnts, SR.name]
    msg = frmt.format(*args)
    tweet(msg)
    a = arcpy.da.FeatureClassToNumPyArray(in_fc, out_flds, "", SR, allpnts)
    # out it goes in array format
    return a, out_flds, SR


def arr2pnts(in_fc, as_struct=True, shp_fld=None, SR=None):
    """Create points from an array.
    :  in_fc - input featureclass
    :  as_struct - if True, returns a structured array with X, Y fields,
    :            - if False, returns an ndarray with dtype='<f8'
    :Notes: calls fc_info to return featureclass information
    """
    if shp_fld is None or SR is None:
        shp_fld, oid_fld, SR = fc_info(in_fc)
    a = arcpy.da.FeatureClassToNumPyArray(in_fc, "*", "", SR)
    dt = [('X', '<f8'), ('Y', '<f8')]
    if as_struct:
        shps = np.array([tuple(i) for i in a[shp_fld]], dtype=dt)
    else:
        shps = a[shp_fld]
    return shps, shp_fld, SR


def arr2line(a, out_fc, SR=None):
    """create lines from an array"""
    pass


def shapes2fc(shps, out_fc):
    """Create a featureclass/shapefile from poly* shapes.
    :  out_fc - full path and name to the output container (gdb or folder)
    """
    msg = "\nCan't overwrite the {}... rename".format(out_fc)
    try:
        if arcpy.Exists(out_fc):
            arcpy.Delete_management(out_fc)
        arcpy.CopyFeatures_management(shps, out_fc)
    except:
        tweet(msg)


def arr2polys(a, out_fc, oid_fld, SR):
    """Make poly* features from a structured array.
    :  a - structured array
    :  out_fc: a featureclass path and name, or None
    :  oid_fld - object id field, used to partition the shapes into groups
    :  SR - spatial reference object, or name
    :Returns:
    :-------
    :  Produces the featureclass optionally, but returns the polygons anyway.
    """
    arcpy.overwriteOutput = True
    pts = []
    keys = np.unique(a[oid_fld])
    for k in keys:
        w = np.where(a[oid_fld] == k)[0]
        v = a['Shape'][w[0]:w[-1] + 1]
        pts.append(v)
    # Create a Polygon from an Array of Points, save to featueclass if needed
    s = []
    for pt in pts:
        s.append(arcpy.Polygon(arcpy.Array([arcpy.Point(*p) for p in pt]), SR))
    return s
# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
#    _demo()
    in_fc = r"C:\GIS\Pro_base\scripts\testfiles\testdata.gdb\Carp_5x5km"
    result = fc_array(in_fc, flds="", allpnts=True)  # a, out_flds, SR
    #del in_fc, result
