# -*- coding: utf-8 -*-
"""

=======

Script :   .py

Author :   Dan_Patterson@carleton.ca

Modified : 2018-

Purpose :  tools for working with numpy arrays and geometry

Notes:

References:

"""
import sys
import numpy as np


ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

import arcpy
import timeit

poly_rings = [
    [[15,0], [25,0], [25,10], [15,10], [15,0]],
    [[18,13], [24,13], [24,18], [18,18], [18,13]
]]

def FromArcPyArray():
    aarr = arcpy.Array(
        arcpy.Array(arcpy.Point(*xy) for xy in ring) for ring in poly_rings
    )
    return arcpy.Polygon(aarr)

def FromEsriJSON():
    esri_json = {"type":"Polygon", "rings":poly_rings}
    return arcpy.AsShape(esri_json, True)

def FromGeoJSON():
    geojson = {"type":"Polygon", "coordinates":poly_rings}
    return arcpy.AsShape(geojson)

def FromWKT():
    wkt = "MULTIPOLYGON({})".format(
        ",".join("(({}))".format(
            ", ".join("{} {}".format(*xy) for xy in ring)
        ) for ring in poly_rings)
    )
    return arcpy.FromWKT(wkt)

for ctor in [FromArcPyArray, FromEsriJSON, FromGeoJSON, FromWKT]:
    pg = ctor()
    print("\n".join(
        str(i) for i in [ctor.__name__, timeit.timeit(ctor, number=10000), ""]
    ))

# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
'''
poly_rings = [
    [[15,0], [25,0], [25,10], [15,10], [15,0]],
    [[18,13], [24,13], [24,18], [18,18], [18,13]],
    [[10,0], [5, 0]]]

z = np.asarray(poly_rings)

z
Out[240]: 
array([list([[15, 0], [25, 0], [25, 10], [15, 10], [15, 0]]),
       list([[18, 13], [24, 13], [24, 18], [18, 18], [18, 13]]),
       list([[10, 0], [5, 0]])], dtype=object)

wkt = "MULTIPOLYGON({})".format(
    ",".join("(({}))".format(
        ", ".join("{} {}".format(*xy) for xy in ring)
    ) for ring in z))

arcpy.FromWKT(wkt)
Out[242]: <Polygon object at 0x2a6592b6630[0x2a6580a4c88]>

esri_json = {"type":"Polygon", "rings":z.tolist()}

arcpy.AsShape(esri_json, True)
Out[244]: <Polygon object at 0x2a6581abc50[0x2a658115d00]>



z = np.asarray([np.asarray(i) for i in poly_rings])

z
Out[246]: 
array([array([[15,  0],
       [25,  0],
       [25, 10],
       [15, 10],
       [15,  0]]),
       array([[18, 13],
       [24, 13],
       [24, 18],
       [18, 18],
       [18, 13]]),
       array([[10,  0],
       [ 5,  0]])], dtype=object)

zl = [i.tolist() for i in z]; esri_json = {"type":"Polygon", "rings":zl}

arcpy.AsShape(esri_json, True)
Out[251]: <Polygon object at 0x2a6564607f0[0x2a658115d28]>
'''
