# -*- coding: UTF-8 -*-
"""
:Script:   query_reclass.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-11-16
:Purpose:  tools for working with numpy arrays
:Useage:  Used in arctoolbox for 'like' sub-query style queries
: - create a field in a geodatabase table and calculates values based on
:   conditions being met.  Essentiall enables a reclassification into a new
:   field.
:
:References:
:----------
: - SQL
:   http://pro.arcgis.com/en/pro-app/help/mapping/navigation/
:        sql-reference-for-elements-used-in-query-expressions.
:        htm#GUID-68D21843-5274-4AF4-B7F3-165892232A43
: - TableSelect
:   http://pro.arcgis.com/en/pro-app/tool-reference/analysis/table-select.htm
:   arcpy.analysis.TableSelect("xy1000", path_to_table,
:                              "Address_ LIKE '%Street%'")
: - CalculateField
:   http://pro.arcgis.com/en/pro-app/tool-reference/data-management/
:        calculate-field.htm
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
from textwrap import dedent
import numpy as np
import arcpy

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}

np.set_printoptions(edgeitems=5, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


# ------------------------------------------------------------------------
# ---- functions ----
def tweet(msg):
    """Print a message for both arcpy and python.
    : msg - a text message
    """
    m = "{}".format(msg)
    arcpy.AddMessage(m)
    print(m)
    print(arcpy.GetMessages())


def _demo():
    """run when script is standalone"""
    in_fc = r'C:\Git_Dan\a_Data\testdata.gdb\xy1000'
    flds = ['OID@', 'Address_']
    in_fld = 'Address_'
    out_fld = 'Str_class'
    from_s = ['Street', 'Lane', 'Court']  # partial set
    to_s = [10, 20, 30]
    dt = [('IDs', '<i4'), ('Str_class', '<i4')]
    testing = True
    return in_fc, flds, in_fld, out_fld, from_s, to_s, dt, testing


def _tool():
    """run when script is from a tool"""
    in_fc = sys.argv[1]
    in_fld = sys.argv[2]
    out_fld = sys.argv[3]
    from_s = sys.argv[4].split(",")
    to_s = sys.argv[5].split(",")
    testing = False
    return in_fc, in_fld, out_fld, from_s, to_s, testing


# ------------------------------------------------------------------------
# (1) ---- Checks to see if running in test mode or from a tool ----------
if len(sys.argv) == 1:
    in_fc, flds, in_fld, out_fld, from_s, to_s, dt, testing = _demo()
else:
    in_fc, in_fld, out_fld, from_s, to_s, testing = _tool()

#
# ------------------------------------------------------------------------
# (2) ---- Create the array from the cursor, print inputs
#
desc = arcpy.da.Describe(in_fc)
tbl_path = desc['path']
fnames = [i.name for i in arcpy.ListFields(in_fc)]
if out_fld in fnames:
    out_fld += '_dupl'
out_fld = arcpy.ValidateFieldName(out_fld)
flds = ['OBJECTID', in_fld]
args = [in_fc, flds, None, None, False, (None, None)]
cur = arcpy.da.SearchCursor(*args)
a = cur._as_narray()
# ----
args = ["-"*60, in_fc, in_fld, out_fld, from_s, to_s, a]
frmt = """
{}\nInput table:  {}\nin_fld:   {}\nout_fld:  {}\n
From values  {}\nTo values    {}
Input array...
{!r:}"""
msg = frmt.format(*args)
tweet(msg)
#
# ------------------------------------------------------------------------
# (3) ----  Check the output dtype and produce the empty array -----------
#
to_s = np.asarray(to_s)
dt_2s = to_s.dtype.str
dt_k = to_s.dtype.kind
dt = [('IDs', '<i4'), (out_fld, dt_2s)]
out = np.zeros((len(a),), dtype=dt)
if dt_k in ('i', 'I', 'l', 'L'):
    fill_v = -9
elif dt_k in ('U', 'S'):
    fill_v = None
else:
    fill_v = np.nan
out[out_fld].fill(fill_v)
out['IDs'] = np.arange(1, len(a) + 1, dtype='int32')
cnt = 0
for f in from_s:
    idx = np.array([i for i, item in enumerate(a[in_fld]) if f in item])
    out[out_fld][idx] = to_s[cnt]
    cnt += 1
#
# ------------------------------------------------------------------------
# (4) ---- Do the table joining ------------------------------------------
if testing:
    tweet("Output array...\n{!r:}".format(out.reshape(out.shape[0], -1)))
else:
    arcpy.da.ExtendTable(in_fc, 'OBJECTID', out, 'OBJECTID')

# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
#    in_fc, flds, in_fld, out_fld, from_s, to_s, dt, testing = _demo()
