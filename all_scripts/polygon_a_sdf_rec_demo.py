# -*- coding: UTF-8 -*-
"""
:Script:   .py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-xx-xx
:Purpose:  tools for working with numpy arrays
:Useage:
:
:References:
:
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
from textwrap import dedent
import numpy as np
import arraytools as art
import arcpy
from arcgis.geometry import _types
from arcgis.features._data.geodataset import SpatialDataFrame as SDF
import pandas as pd


# import json

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.2f}'.format}

np.set_printoptions(edgeitems=3, linewidth=80, precision=2, suppress=True,
                    threshold=10, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


def time_deco(func):  # timing originally
    """timing decorator function
    :print("\n  print results inside wrapper or use <return> ... ")
    """
    import time
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()        # start time
        result = func(*args, **kwargs)  # ... run the function ...
        t1 = time.perf_counter()        # end time
        dt = t1-t0
        print("\nTiming function for... {}\n".format(func.__name__))
        print("  Time: {: <8.2e}s for {:,} objects".format(dt, len(result)))
        return result                   # return the result of the function
        return dt                       # return delta time
    return wrapper


def fc_info(in_fc):
    """Return basic featureclass information, including...
    : SR - spatial reference object (use SR.name to get the name)
    : shp_fld - field name which contains the geometry object
    : oid_fld - the object index/id field name
    : - others: 'areaFieldName', 'baseName', 'catalogPath','featureType',
    :   'fields', 'hasOID', 'hasM', 'hasZ', 'path'
    : - all_flds =[i.name for i in desc['fields']]
    """
    desc = arcpy.da.Describe(in_fc)
    args = ['shapeFieldName', 'OIDFieldName', 'spatialReference', 'shapeType']
    shp_fld, oid_fld, SR, shp_type = [desc[i] for i in args]
    return shp_fld, oid_fld, SR, shp_type


def tweet(msg):
    """Print a message for both arcpy and python.
    : msg - a text message
    """
    m = "\n{}\n".format(msg)
    arcpy.AddMessage(m)
    print(m)
    print(arcpy.GetMessages())


def flatten(a_list, flat_list=None):
    """Change the isinstance as appropriate
    :  Flatten an object using recursion
    :  see: itertools.chain() for an alternate method of flattening.
    """
    if flat_list is None:
        flat_list = []
    for item in a_list:
        if isinstance(item, list):
            flatten(item, flat_list)
        else:
            flat_list.append(item)
    return flat_list



def unpack(iterable, param='__iter__'):
    """Unpack an iterable based on the param(eter) condition using recursion.
    :Notes:
    : ---- see main docs for more information and options ----
    : To produce an array from this, use the following after this is done.
    :   out = np.array(xy).reshape(len(xy)//2, 2)
    """
    xy = []
    for x in iterable:
        if hasattr(x, '__iter__'):
            xy.extend(unpack(x))
        else:
            xy.append(x)
    return xy


@time_deco
def get_geom(in_fc):
    """just get the geometry object"""
    coords = [np.asarray(row[0].__geo_interface__['coordinates'])
              for row in arcpy.da.SearchCursor(in_fc, ['SHAPE@'])]  # shape@
    # coords = [i.__geo_interface__['coordinates'] for i in geoms]
    return coords  # , g2

@time_deco
def cursor_to_dicts(in_fc):  #cursor, field_names):
    """use a searchcursor to get a list of a shape's attributes
    :
    :Reference:
    :---------
    :https://stackoverflow.com/questions/11869473/
    :      loading-a-feature-class-in-a-list-using-arcpy-strange-behaviour
    :      -of-searchcurso
    def cursor_to_dicts(cursor, field_names):
        for row in cursor:
            row_dict = {}
            for field in field_names:
                val = row.getValue(field)
                row_dict[field] = getattr(val, '__geo_interface__', val)
            yield row_dict

    fc = '/path/to/fc'
    fields = [f.name for f in arcpy.ListFields(fc)]   # get field list
    features = list(cursor_to_dicts(arcpy.SearchCursor(fc), fields))
    :Useage:
    :------
    : flds = [f.name for f in arcpy.ListFields(fc)]  # get field list
    : dct = list(cursor_to_dicts(arcpy.SearchCursor(fc), flds))
    : 'Shape': {'coordinates': [[[(300020.0, 5000000.0), ...snip...
    :      (300020.0, 5000000.0)]]],
    : 'type': 'MultiPolygon'}
    : c[0]['Shape']['coordinates']
    :
    : shps = [np.array(c[i]['Shape']['coordinates']) for i in range(len(c))]
    :---------------------------------------------------------
    """
    def yld(cursor, flds):
        for row in cursor:
            row_dict = {}
            idx = row[0]
            shp = row[1]
            #if hasattr(val, '__geo_interface__'):
            v = np.asarray(shp.__geo_interface__['coordinates'])
            row_dict['ID'] = idx
            row_dict['Shape'] = v
            yield row_dict

    #flds = [f.name for f in arcpy.ListFields(in_fc)]   # get field list
    flds = ['OID@', 'SHAPE@']
    cursor = arcpy.da.SearchCursor(in_fc, flds)
    features = list(yld(cursor, flds))
    return features




@time_deco
def to_arr0(in_fc):
    """Just get the geometry and id field
    """
    shp_fld, oid_fld, SR, shp_type = fc_info(in_fc)
    flds = ['OID@', 'SHAPE@X', 'SHAPE@Y']
    a = arcpy.da.FeatureClassToNumPyArray(in_fc,
                                          field_names=flds,
                                          spatial_reference=SR,
                                          explode_to_points=True)
    dt = [('Idx', '<i4'), ('Xs', '<f8'), ('Ys', '<f8')]
    a.dtype = dt
    return a


@time_deco
def to_arr1(in_fc, split_geom=True):
    """Pulls out the geometry and index values from a featureclass.  Optionally
    :  the full array can be split into an object array containing an array
    :  of individual geometries.  Note, this is only useful for poly* objects
    :
    :Requires: numpy, arcpy and fc_info
    :--------
    :  in_fc - featureclass
    :  split-geom - True, separate arrays are created for each geometry object
    :
    :Notes:
    : - arcpy.da.SearchCursor(
    :     in_fc, field_names, where_clause, spatial_reference,
    :     explode_to_points, sql_clause=(None, None))
    :-------------------------------------------------------------------------
    """
    shp_fld, oid_fld, SR, shp_type = fc_info(in_fc)
#    flds = arcpy.ListFields(in_fc)
#    fnames = [f.name for f in flds if f.type not in ['OID', 'Geometry']]
#    flds = [oid_fld, shp_fld]
    g_flds = ['OID@', 'SHAPE@X', 'SHAPE@Y']  # 'SHAPE@', SHAPE@Z etc
    vals = []
    with arcpy.da.SearchCursor(in_fc, g_flds, None, SR, True) as rows:
        for row in rows:
            vals.append(row)
    del row, rows
    # ---- construct the array ----
    dt = [('Idx', '<i4'), ('Xs', '<f8'), ('Ys', '<f8')]
    a = np.array(vals, dtype=dt)
    # ---- split out into an object array containing arrays of dt ----
    if split_geom:
        ids = np.unique(a['Idx'])
        w = np.where(np.diff(a['Idx']))[0] + 1
        a = np.split(a, w)  # a[['Xs', 'Ys']], w)
        # dt = [('Idx', '<i4'), ('Shp', 'O')]
        a = np.array([[ids[i], a[i][['Xs', 'Ys']]] for i in range(len(ids))])
    return a


@time_deco
def to_arr(in_fc, use_geo=False):
    """Convert a featureclass to a structured or recarray using a searchcursor.
    :
    :Requires: import arcpy, numpy as np
    :--------
    : in_fc - featureclass
    : use_geo - True .__geo_interface__
    :         - list comprehension
    :get the row information
    : cycle through all geometries and get xy pairs
    :
    :References:
    :----------
    : - see the polygon, polyline etc classes in
    :   C:\ArcPro\Resources\ArcPy\arcpy\arcobjects\geometry.py
    """
    shp_fld, oid_fld, SR, shp_type = fc_info(in_fc)  # get the base information
    flds = arcpy.ListFields(in_fc)
    fnames = [f.name for f in flds if f.type not in ['OID', 'Geometry']]
    geom_flds = ['SHAPE@', oid_fld] + fnames
    flds = [shp_fld, oid_fld] + fnames
    vals = []
    geoms = []
    coords = []
    idx = flds.index(shp_fld)
    with arcpy.da.SearchCursor(in_fc,
                               field_names=geom_flds, where_clause=None,
                               spatial_reference=SR,  explode_to_points=False,
                               sql_clause=(None, None)) as rows:
        for row in rows:
            row = list(row)
            geom = row.pop(idx)
            vals.append(row)
            geoms.append(geom)  # pop the geometry out
            if use_geo:
                xy = geom.__geo_interface__['coordinates']
            else:
                xy = [np.array([(pt.X, pt.Y) for pt in arr if pt])
                      for arr in geom]  # if pt else None
            coords.append(np.asarray(xy))  # maybe dump the last as np.asarray
            del row, geom, xy
        del rows
    return vals, coords, geoms


@time_deco
def fc_sdf(in_fc, fields=None, sr=None, where_clause=None, sql_clause=None):
    """Abbreviated version of featureclass to SpatialDataFrame
    : in fileops.py... from_featureclass(filename, **kwargs)
    : sdf - SpatialDataFrame is a class in
    :     - arcgis.features._data.geodataset.geodataframe
    :     - C:\ArcPro\bin\Python\envs\arcgispro-py3\lib\site-packages\
            arcgis\features\_data\geodataset.py
    :
    :  all_vals, vals, geoms = fc_sdf(in_fc)
    :  p = all_vals[0][-1]
    :  p0 = p.getPart(0)  # returns an array which you can cycle through
    :
    :Note:  These imports are at the top of the script
    :  from arcgis.geometry import _types
    :  from arcgis.features._data.geodataset import SpatialDataFrame as SDF
    :  import pandas as pd
    """
    if not fields:
        fields = [field.name for field in arcpy.ListFields(in_fc)
                  if field.type not in ['Geometry']]
    geom_fields = ['SHAPE@'] + fields
    flds = ['SHAPE'] + fields
    vals = []
    all_vals = []
    geoms = []
    geom_idx = flds.index('SHAPE')
    with arcpy.da.SearchCursor(in_fc,
                               field_names=geom_fields,
                               where_clause=where_clause,
                               sql_clause=sql_clause,
                               spatial_reference=sr) as rows:

        for row in rows:
            all_vals.append(row)
            row = list(row)
            geoms.append(_types.Geometry(row.pop(geom_idx)))
            vals.append(row)
            del row
        del rows
    df = pd.DataFrame(data=vals, columns=fields)
    sdf = SDF(data=df, geometry=geoms)
    sdf.reset_index(drop=True, inplace=True)
    del df
    if sr is None:
        sdf.sr = sr
    else:
        sdf.sr = sdf.geometry[0].spatialReference
    rec = sdf.to_records()
    shps = np.asarray([np.asarray(i['rings']) for i in a['SHAPE']])
    return rec, shps, sdf, geoms  # df


def fc_array(in_fc, flds="", allpnts=True):
    """Convert a featureclass to an ndarray of attribute, with the geometry
    :  removed.
    :
    :Requires:
    :--------
    : input_fc - featureclass/shapefile complete path
    : flds     - ""  just oid and shape fields
    :          - "*" all fields or
    :          - ['Field1', 'Field2', etc] for specific fields
    : allpnts  - True/False
    :          - True to explode the geometry to individual points
    :          - False for the centroid of the geometry
    :References:
    :----------
    :  FeatureClassToNumPyArray, ListFields for more information
    """
    out_flds = []
    shp_fld, oid_fld, SR, shp_type = fc_info(in_fc)  # get the base information
    fields = arcpy.ListFields(in_fc)       # all fields in the shapefile
    if flds == "":                         # return just OID and Shape field
        out_flds = [oid_fld, shp_fld]      # FID and Shape field required
    elif flds == "*":                      # all fields
        out_flds = [f.name for f in fields]
    else:                                  # oid, shape and specific fields
        out_flds = [oid_fld, shp_fld]
        for f in fields:
            if f.name in flds:
                out_flds.append(f.name)
    frmt = """Creating array from featureclass using fc_array...with...
    {}\nFields...{}\nAll pnts...{}\nSR...{}
    """
    frmt = dedent(frmt)
    args = [in_fc, out_flds, allpnts, SR.name]
    msg = frmt.format(*args)
    tweet(msg)
    a = arcpy.da.FeatureClassToNumPyArray(in_fc, out_flds, "", SR, allpnts)
    # out it goes in array format
    return a

# =============================================================================
# @time_deco
# def to_arr1(geom):
#     """making parts and rings
#     """
#     xy = []
#     for pts in geom:
#         sub = []
#         for pt in pts:
#             if pt:  # is not None:
#                 sub.append([pt.X, pt.Y])
#             else:
#                 xy.append(sub)  #np.asarray(sub))
#                 sub = []
#         xy.append(sub)  #np.asarray(sub))
#     return np.asarray(xy)
# =============================================================================

def _demo():
    """
    : -
    """
    in_fc = r"C:\Git_Dan\a_Data\testdata.gdb\polygon"

    vals, a0, geoms = to_arr(in_fc, use_geo=False)
    _, a1, _ = to_arr(in_fc, use_geo=True)
    # fc_array(in_fc, flds="*", allpnts=True)
    a2 = to_arr0(in_fc)
    a3 = np.array([to_arr1(g) for g in geoms])
    sdf = SDF.from_featureclass(in_fc)
    a4_rec = SDF.to_records(sdf)
    a4_s = a4_rec['SHAPE']
    a4 = np.asarray([np.array(i['rings']) for i in a4_s])
    # a_rec1 = a_nd.view(np.recarray)
    # print("\ndarray.... \n{!r:}\n\nsdf..... \n{!r:}\n".format(a_nd, sdf))
    # print("\nrec0...... \n{!r:}\n\nrec1.... \n{!r:}\n".format(a_rec0, a_rec1))
    return a0, a1, a2, a3, a4 #, a5
# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
#    a_nd, sdf, a_rec0, a_rec1 = _demo()
#    print(art.frmt_struct(a_nd, 1, True))
#    print(art.frmt_struct(a_rec1, 1, True))

    # pd.DataFrame.from_records(data, index=None, exclude=None, columns=None,
    #                         coerce_float=False, nrows=None)

    """
    flds = a_nd.dtype.names
    vals = a_nd.tolist()
    df = pd.DataFrame(data=vals, columns=flds)
    g = a_rec1.Shape
    gg = [arcpy.Point(x,y) for x,y in g]
    sdf2 = SDF(data=df, geometry=gg)  # SpatialDataFrame
    #


#    in_fc = r"C:\Git_Dan\a_Data\testdata.gdb\polygon"

    geom_type = 'Polygon'
    row = None
    geo = [json.loads(u[i][0].JSON)['rings']  for i in range(len(u))]
    attr = [u[i][1:]  for i in range(len(u))]
    #geo = [json.loads(u[i][0].JSON)['rings']  for i in range(len(u))]
    # zz = {"type" : geom_type, "geometry" : json.loads(z), "attributes":row}
    # json.loads(u[0][0])
    """

    in_fc = r"C:\Git_Dan\a_Data\testdata.gdb\polygon"
    #in_fc = r"C:\Git_Dan\a_Data\arcpytools_demo.gdb\Can_0_big_3"
    #in_fc = r"C:\Data\Canada\CAN_adm0.gdb\CAN_0_sp"
    #a, b, c = to_arr(in_fc)
    """
    in_fc
Out[4]: 'C:\\Data\\Canada\\CAN_adm.gdb\\CAN_adm1'
%timeit to_arr(in_fc)
2min 5s ± 2.33 s per loop (mean ± std. dev. of 7 runs, 1 loop each)

%timeit fc_sdf(in_fc)
5min 53s ± 7.99 s per loop (mean ± std. dev. of 7 runs, 1 loop each)

%timeit -n1 -r1 to_arr(in_fc)  # with conversion to points
1min 54s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)

%timeit -n1 -r1 to_arr(in_fc)  # without conversion to points
5.1 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)
"""
    """ taken out of to_fc
            desc = arcpy.da.Describe(in_fc)  # use the new da.Describe method
        if hasattr(desc, 'areaFieldName'):
            afn = desc.areaFieldName
            if afn in fields:
                fields.remove(afn)
        if hasattr(desc, 'lengthFieldName'):
            lfn = desc.lengthFieldName
            if lfn in fields:
                fields.remove(lfn)
        del desc
    """