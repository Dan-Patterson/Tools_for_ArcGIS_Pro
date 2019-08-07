## Free Tools ##

Developed in ArcGIS Pro 2.4, python 3.6.8 and numpy 1.16.4.

These demo scripts and the toolbox show how numpy and arcpy can play nice together and generate geometries that are normally only available at the ArcGIS Pro Advanced level.  The tools are already provided to do this, but less attention is paid to the attributes.  Usually a spatial and/or attribute join enables one to bring the attributes from the input class to the output class.  This can be done after the geometry is created, or I may have done so during script construction (depending on how bored I was).

In some cases, the outputs are only one option of what the Esri tool provides, for example #6 Polygons to Polylines, is just that... a simple conversion of the geometry type, no fancy intersection and overlap stuff... you get what you pay for, but the widest use is probably the simplest.


## Last update : 2019-08-06 ##

The toolbox and scripts allow one to determine:

**Feature Envelope to Polygon**

Using the Geo array class, the extent of polyline and polygon features are created from their constituent points.
    https://pro.arcgis.com/en/pro-app/tool-reference/data-management/feature-envelope-to-polygon.htm

**Convex hulls**

Simple convex hull implementation in python, or scipy (feature points > than a threshold)
    https://pro.arcgis.com/en/pro-app/tool-reference/data-management/minimum-bounding-geometry.htm

**Feature to Point**

For polygon features.  Reduces the points to a representative centroid.
    https://pro.arcgis.com/en/pro-app/tool-reference/data-management/feature-to-point.htm

**Split Line at Vertices**

As it says.  I chose to keep the order of the resultant line segments as they were and not remove apparent `duplicates` for line segments that occur on shared borders.  In such cases, the shared segments will have the same points, but their from-to order is reversed.  There are cases to be made for keeping them or removing them... however, if they are removed, then they are harder to add back in should one want to recreate polygon geometry from the line segments.

    https://pro.arcgis.com/en/pro-app/tool-reference/data-management/split-line-at-vertices.htm

**Feature Vertices to Points**

Convert polygon features to a centroid.  One point is returned for multipart shapes, but this could be altered if someone has a use-case that might be relevant.

    https://pro.arcgis.com/en/pro-app/tool-reference/data-management/feature-vertices-to-points.htm

**Polygons to Polylines**

Really... You are just changing from polygons to polylines.  They are still a closed geometry, nothing fancy geometry-wise.  Should definitely be **Freed** from its shackles.

    https://pro.arcgis.com/en/pro-app/tool-reference/data-management/feature-to-polygon.htm

**Bounding Circles**

Another container that has been around for a long time in a variety of formats and readily implemented in python.  Sort-of ported this over from an old toolbox for ArcMap, but redone for the new geom class.  Speedy and accurate.

    https://pro.arcgis.com/en/pro-app/tool-reference/data-management/minimum-bounding-geometry.htm

**Frequency**

Another tool that should be free for such basic functionality.
    https://community.esri.com/blogs/dan_patterson/2016/03/03/create-classes-from-multiple-columns
    https://pro.arcgis.com/en/pro-app/tool-reference/analysis/frequency.htm


