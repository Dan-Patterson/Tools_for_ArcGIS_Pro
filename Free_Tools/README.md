## Free Tools ##

Developed in ArcGIS Pro 2.4, python 3.6.8 and numpy 1.16.4.

These demo scripts and the toolbox show how numpy and arcpy can play nice together and generate geometries that are normally only available at the ArcGIS Pro Advanced level.  The tools are already provided to do this, but less attention is paid to the attributes.  Usually a spatial and/or attribute join enables one to bring the attributes from the input class to the output class.  This can be done after the geometry is created, or I may have done so during script construction (depending on how bored I was).

In some cases, the outputs are only one option of what the Esri tool provides, for example #6 Polygons to Polylines, is just that... a simple conversion of the geometry type, no fancy intersection and overlap stuff... you get what you pay for, but the widest use is probably the simplest.


## Last update : 2019-07-03 ##

The toolbox and scripts allow one to determine:

**Feature Envelope to Polygon**

Using the Geo array class, the extent of polyline and polygon features are created from their constituent points.
    https://pro.arcgis.com/en/pro-app/tool-reference/data-management/feature-envelope-to-polygon.htm>

**Convex hulls**

    https://pro.arcgis.com/en/pro-app/tool-reference/data-management/minimum-bounding-geometry.htm

**Feature to Point**

    https://pro.arcgis.com/en/pro-app/tool-reference/data-management/feature-to-point.htm

**Split Line at Vertices**

    https://pro.arcgis.com/en/pro-app/tool-reference/data-management/split-line-at-vertices.htm

**Feature Vertices to Points**

    https://pro.arcgis.com/en/pro-app/tool-reference/data-management/feature-vertices-to-points.htm

**Polygons to Polylines**

    https://pro.arcgis.com/en/pro-app/tool-reference/data-management/feature-to-polygon.htm

**Frequency**

    https://pro.arcgis.com/en/pro-app/tool-reference/analysis/frequency.htm


