# -*- coding: UTF-8 -*-
"""
========
vincenty
========

Script : vincenty.py

Author : Dan.Patterson@carleton.ca

Created : 2014-10

Modified : 2019-02-27

Purpose :
    Calculates the Vincenty Inverse distance solution for 2 long/lat pairs

References
----------
`<http://www.movable-type.co.uk/scripts/latlong-vincenty.html>`_.  java code

T Vincenty, 1975 "Direct and Inverse Solutions of Geodesics on the
Ellipsoid with application of nested equations", Survey Review,
vol XXIII, no 176, 1975

`<http://www.ngs.noaa.gov/PUBS_LIB/inverse.pdf>`_.

`<https://github.com/geopy/geopy/blob/master/geopy/distance.py>`_.

`<http://www.onlineconversion.com/map_greatcircle_bearings.htm>`_.

Notes
-----
>>> atan2(y, x) # or atan2(sin, cos) not like Excel

used fmod(x, y) to get the modulous as per python

`<http://stackoverflow.com/questions/34552284/vectorize-haversine-distance-
computation-along-path-given-by-list-of-coordinates>`_.

Returns
-------
Distance in meters, initial and final bearings (as an azimuth from N)

Examples::

    d = [[-78.0, 44.0, -78.0, 48.0], [-78.0, 48.0, -72.0, 48.0],
         [-72.0, 48.0, -72.0, 44.0], [-72.0, 44.0, -78.0, 44.0],
         [-75.0, 45.0, -76.0, 46.0], [-75.0, 46.0, -76.0, 45.0],
         [-75.0, 45.0, -75.0, 46.0], [-78.0, 45.0, -72.0, 45.0],
         [-90.0, 0.0, 0.0, 0.0], [-75.0, 0.0, -75.0, 90.0]]
    orig = data[:, :2]
    dest = data[:, -2:]
    #
          Orig_  Orig_  Dest_  Dest_              Orig_    Dest_
    id    long   lat    long   lat    Distance_m  Bearing  Bearing  
    --------------------------------------------------------------
    000 -78.00  44.00  -78.00  48.00    444605.23    0.00     0.00
    001 -78.00  48.00  -72.00  48.00    447639.09   87.77    92.23
    002 -72.00  48.00  -72.00  44.00    444605.23  180.00   180.00
    003 -72.00  44.00  -78.00  44.00    481131.00  272.08   267.92
    004 -75.00  45.00  -76.00  46.00    135869.09  325.24   324.53
    005 -75.00  46.00  -76.00  45.00    135869.09  215.47   214.76
    006 -75.00  45.00  -75.00  46.00    111141.55    0.00     0.00
    007 -78.00  45.00  -72.00  45.00    472972.88   87.88    92.12
    008 -90.00   0.00    0.00   0.00  10018754.17   90.00    90.00
    009 -75.00   0.00  -75.00  90.00  10001965.73    0.00     0.00

Notes
-----
Using arcpy

>>> p0 = arcpy.Point(-78.0, 48.0)
>>> p1 = arcpy.Point(-75.0, 44.0)
>>> SR = arcpy.SpatialReference(4326)
>>> pg0 = arcpy.PointGeometry(p0, SR)
>>> pg1 = arcpy.PointGeometry(p1, SR)
>>> pg0.angleAndDistanceTo(p1, 'GEODESIC')  
... (151.31467831801586, 501577.31743303017)
>>> pg0.angleAndDistanceTo(p1, 'GREAT_ELLIPTIC')
... (151.31821854020157, 501577.3177450169)
>>> pg0.angleAndDistanceTo(p1, 'LOXODROME')
... (152.4195622173897, 501606.98677004984)
>>> pg0.angleAndDistanceTo(p1, 'PRESERVE_SHAPE')
... (151.31467831801586, 501577.31743303017)


"""
# pylint: disable=C0103  # invalid-name
# pylint: disable=R0914  # Too many local variables
# pylint: disable=R1710  # inconsistent-return-statements
# pylint: disable=W0105  # string statement has no effect

# ---- imports, formats, constants ----

import sys
from textwrap import dedent
import math
import numpy as np
from arcpytools_plt import tweet
import arcpy

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2,
                    suppress=True, threshold=100,
                    formatter=ft)
np.ma.masked_print_option.set_display('-')

script = sys.argv[0]

__all__ = ['vincenty_cal',
           'vincenty']

arcpy.overwriteOutputs = True

# ---- functions ----

def vincenty_cal(long0, lat0, long1, lat1):
    """return the distance on the ellipsoid between two points using
    Vincenty's Inverse method

    Notes
    -----
    `a`, `b` : numbers
        semi-major and minor axes WGS84 model (EPSG 4326) or other
    `f` : number
        inverse flattening
    `L`, `dL` : number
        delta longitude, initial and subsequent
    `u0`, `u1` : number
        reduced latitude
    `s_sig`, `c_sig` : number
        sine and cosine sigma
    """
    a = 6378137.0         # GCS WGS84
    b = 6356752.314245
    ab_b = (a**2 - b**2)/b**2
    f = 1.0/298.257223563
    twoPI = 2*math.pi
    dL = L = math.radians(long1 - long0)
    u0 = math.atan((1 - f) * math.tan(math.radians(lat0)))  # reduced latitudes
    u1 = math.atan((1 - f) * math.tan(math.radians(lat1)))
    s_u0 = math.sin(u0)
    c_u0 = math.cos(u0)
    s_u1 = math.sin(u1)
    c_u1 = math.cos(u1)
    # ---- combine repetitive terms ----
    sc_01 = s_u0*c_u1
    cs_01 = c_u0*s_u1
    cc_01 = c_u0*c_u1
    ss_01 = s_u0*s_u1
    #
    lambdaP = float()
    max_iter = 20
    # first approximation
    cnt = 0
    while (cnt < max_iter):
        s_dL = math.sin(dL)
        c_dL = math.cos(dL)
        s_sig = math.sqrt((c_u1*s_dL)**2 + (cs_01 - sc_01*c_dL)**2)  # eq14
        if (s_sig == 0):
            return 0
        c_sig = ss_01 + cc_01*c_dL                      # eq 15
        sigma = math.atan2(s_sig, c_sig)                # eq 16
        s_alpha = cc_01*s_dL/s_sig                      # eq 17
        c_alpha2 = 1.0 - s_alpha**2
        if c_alpha2 != 0.0:
            c_sigM2 = c_sig - 2.0*s_u0*s_u1/c_alpha2    # eq 18
        else:
            c_sigM2 = c_sig
        C = f/16.0 * c_alpha2*(4 + f*(4 - 3*c_alpha2))  # eq 10
        lambdaP = dL
        # dL => equation 11
        dL = L + (1 - C)*f*s_alpha*(sigma +
                                    C*s_sig*(c_sigM2 +
                                             C*c_sig*(-1.0 + 2*c_sigM2**2)))
        #
        if (cnt == max_iter):          # is it time to bail?
            return 0.0
        elif((math.fabs(dL - lambdaP) > 1.0e-12) and (cnt < max_iter)):
            cnt += 1
        else:
            break
    # ---- end of while ----
    uSq = c_alpha2 * ab_b
    A = 1 + uSq/16384.0 * (4096 + uSq*(-768 + uSq*(320 - 175*uSq)))  # eq 3
    B = uSq/1024.0 * (256 +  uSq*(-128 + uSq*(74 - 47*uSq)))         # eq 4
    d_sigma = B*s_sig*(c_sigM2 +
                      (B/4.0)*(c_sig*(-1 + 2*c_sigM2**2) -
                      (B/6.0)*c_sigM2*(-3 + 4*s_sig**2)*(-3 +
                      4*c_sigM2**2)))
   # removed **2 from c_sigM2 
    # d_sigma => eq 6
    dist = b*A*(sigma - d_sigma)                                     # eq 19
    alpha1 = math.atan2(c_u1*s_dL,  cs_01 - sc_01*c_dL)
    alpha2 = math.atan2(c_u0*s_dL, -sc_01 + cs_01*c_dL)
    # normalize to 0...360  degrees
    alpha1 = math.degrees(math.fmod((alpha1 + twoPI), twoPI))      # eq 20
    alpha2 = math.degrees(math.fmod((alpha2 + twoPI), twoPI)) # eq 21
    return dist, alpha1, alpha2


def _vincenty_(data, as_array=True):
    """Calculate vincenty distance and bearings for single or multiple
    orig-dest pairs of longitude/latitude coordinates

    Parameters
    ----------
    data : array-like
        long0, lat0, long1, lat1 as either a single entry or a Nx4 array
    as_array : boolean
        True, returns a structured array. False, returns an ndarray.
    """
    result = []
    for i, coords in enumerate(data):
        long0, lat0, long1, lat1 = data[i]
        r = vincenty_cal(long0, lat0, long1, lat1)
        row = np.concatenate((data[i], r))
        result.append(row)
    if as_array:
        result = np.array(result)
        dt = [('Orig_long', '<f8'), ('Orig_lat', '<f8'), ('Dest_long', '<f8'),
              ('Dest_lat', '<f8'), ('Distance_m', '<f8'),
              ('Orig_Bearing', '<f8'), ('Dest_Bearing', '<f8')]
        return result.view(dt).squeeze()
    return np.array(result)

# ---- demos -----------------------------------------------------------------
#
def _demo_data():
    """some demo data
    d[:4] UTM18N bounding box
    """
    d = [[-78.0, 44.0, -78.0, 48.0],
         [-78.0, 48.0, -72.0, 48.0],
         [-72.0, 48.0, -72.0, 44.0],
         [-72.0, 44.0, -78.0, 44.0],
         [-78.0, 48.0, -75.0, 48.0],
         [-78.0, 44.0, -75.0, 44.0],
         [-78.0, 48.0, -75.0, 44.0],
         [-78.0, 44.0, -75.0, 48.0],
         [-78.0, 44.0, -72.0, 48.0],
         [-72.0, 44.0, -78.0, 48.0]]
    o = np.tile([-75., 45.], 25).reshape(25,2)
    dlong = np.linspace(-78., -72., 25, True)
    dlat = np.linspace(42., 48., 25, True)
    od = np.vstack((o[:,0], o[:,1], dlong, dlat)).T
    return od

# ===========================================================================
# ---- main section: testing or tool run ------------------------------------
# transect_lines(N=5, orig=None, dist=1, x_offset=0, y_offset=0,
#                   bearing=0, as_ndarray=True)
if len(sys.argv) == 1:
    create_output = False
    in_pth = script.split("/")[:-2] + ["Polygon_lineTools.gdb"]
    frum_too = _demo_data()
    arr = _vincenty_(frum_too, True)
    out_fc = "/".join(in_pth) + "/od_circ"  # sys.argv[9]
    SR = 4326
    #print(arr)
else:
    create_output = True
    in_tbl =  sys.argv[1]
    long0 = sys.argv[2]
    lat0 = sys.argv[3]
    long1 = sys.argv[4]
    lat1 = sys.argv[5]
    out_tbl = sys.argv[6]
    SR = 4326
    flds = [long0, lat0, long1, lat1]
    tweet("fields {}".format(flds))
    a = arcpy.da.TableToNumPyArray(in_tbl, flds)
    frum_too = a.view((a.dtype[0], len(a.dtype.names)))
    arr = _vincenty_(frum_too, True)
# ---- Create the table and the lines
if create_output:
    if arcpy.Exists(out_tbl):
        arcpy.Delete_management(out_tbl)
    arcpy.da.NumPyArrayToTable(arr, out_tbl)
    out_fc = out_tbl + "_line"

    arcpy.XYToLine_management(out_tbl, out_fc, 'Orig_long', 'Orig_lat',
                              'Dest_long', 'Dest_lat', "GEODESIC",
                              spatial_reference=SR)
    #       startx_field, starty_field, endx_field, endy_field,
    #       {line_type}, {id_field}, {spatial_reference}

# ==== Processing finished ====
    
# ---------------------------------------------------------------------
if __name__ == "__main__":
    """Main section...   """
#    #print("Script... {}".format(script))
#     ----- uncomment one of the  below  -------------------
#    _demo_single()
#    _demo_batch()
