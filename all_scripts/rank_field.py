# -*- coding: UTF-8 -*-
"""
:Script:   rank_field.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-07-12
:Purpose:  tools for working with arcpy and numpy arrays
:  - sort a table based on a field or fields
:References:
:(1) FeatureClassToNumPyArray (in_table, field_names, {where_clause},
:                           {spatial_reference}, {explode_to_points},
:                           {skip_nulls}, {null_value})
: - http://pro.arcgis.com/en/pro-app/arcpy/data-access/
:        featureclasstonumpyarray.htm
:   -SHAPE@TRUECENTROID —A tuple of the feature's true centroid coordinates
:   -SHAPE@X — A double of the feature's x-coordinate.
:   -SHAPE@Y — A double of the feature's y-coordinate.
:
:(2) TableToNumPyArray (in_table, field_names, {where_clause},
:                   {skip_nulls}, {null_value})
: - http://pro.arcgis.com/en/pro-app/arcpy/data-access/tabletonumpyarray.htm
:
:(3) ExtendTable(in_table, table_match_field, in_array,
:              array_match_field, {append_only})
: - http://pro.arcgis.com/en/pro-app/arcpy/data-access/extendtable.htm
:Notes:
:-----
:
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
import numpy as np
import arcpy

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}

np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


# -----------------------------------------------------------------------------
# functions from arraytools
def tweet(msg):
    """Produce a message (msg)for both arcpy and python
    """
    m = "{}".format(msg)
    arcpy.AddMessage(m)
    print(m)
    print(arcpy.GetMessages())


def fc_info(in_fc):
    """basic feature class information"""
    desc = arcpy.Describe(in_fc)    # fix to use da.Describe
    SR = desc.spatialReference      # spatial reference object
    shp_fld = desc.shapeFieldName   # FID or OIDName, normally
    oid_fld = desc.OIDFieldName     # Shapefield ...
    return shp_fld, oid_fld, SR


# -----------------------------------------------------------------------------
# ---- Other defs ----
def rankmin(x):
    """Returns a rank accounting for duplicates
    :  The array must be sorted first
    :  Warren W. solution at
    : https://stackoverflow.com/questions/39059371/
    :       can-numpys-argsort-give-equal-element-the-same-rank
    """
    u, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    csum = np.zeros_like(counts)
    csum[1:] = counts[:-1].cumsum()
    return csum[inv]


# ------------------------------------------------------------------------
# ---- Checks to see if running in test mode or from a tool
if len(sys.argv) == 1:
    in_tbl = r'C:\GIS\Tools_scripts\Statistics\Stats_demo_01.gdb\pnts_2K_normal'
    desc = arcpy.da.Describe(in_tbl)
    oid_fld = desc['OIDFieldName']
    flds = arcpy.ListFields(in_tbl)
    # fld_names = [fld.name for fld in flds]
    fld_names = ['Rand_1_100', oid_fld]
    testing = True
    rank_fld = 'Rand_1_100'
    rank_min = True
else:
    in_tbl = sys.argv[1]
    fld_names = sys.argv[2]
    rank_fld = sys.argv[3]
    rank_min = sys.argv[4]
    #
    desc = arcpy.da.Describe(in_tbl)
    oid_fld = desc['OIDFieldName']
    testing = False

# ------------------------------------------------------------------------
# ---- Create the array, sort extend/join to the input table ----
if rank_fld == '':
    rank_fld = 'Rank'
else:
    no_good = ' !"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~'
    rank_fld = "".join([i for i in rank_fld if i not in no_good])

if isinstance(fld_names, (list, tuple)):
    order_by = fld_names
elif isinstance(fld_names, (str)):
    if str(fld_names).find(";") == -1:
        order_by = [fld_names, oid_fld]
    else:
        order_by = fld_names.split(";") + [oid_fld]

a = arcpy.da.TableToNumPyArray(in_tbl, field_names=order_by)

a_s = a[order_by]
srted = np.argsort(a_s, order=order_by)

dt = [(oid_fld, '<i4'), (rank_fld, '<i4')]
j_a = np.zeros(a.shape, dtype=dt)
j_a[oid_fld] = a_s[srted][oid_fld]

if rank_min in ('true', True):  # use regular or rankmin ranking method
    r = a_s[srted][fld_names]
    r = rankmin(r)
    j_a[rank_fld] = r
else:
    j_a[rank_fld] = np.arange(1, a.shape[0]+1)
#
if not testing:
    arcpy.da.ExtendTable(in_table=in_tbl,
                         table_match_field=oid_fld,
                         in_array=j_a,
                         array_match_field=oid_fld)

frmt = """
{}
:Script....{}
:Ranking... {}
:Using fields...
:   {}
{}
"""
args = ["-"*70, script, in_tbl, order_by, "-"*70]
msg = frmt.format(*args)
tweet(msg)

# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
    pass
