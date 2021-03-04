
## Tools for ArcGIS Pro

----

This folder contains various toolboxes for working with feature data in ArcGIS Pro.

**NOTE**

The main testing repository for this package is *numpy_geometry* on the main landing page.

I move finished materials here when I have completed updates or additions to it.

----

### [**Free tools**](/Free_Tools/README.md)

Free tools are sample tools that use numpy and arcpy to implement tools that normally require a Standard or Advanced ArcGIS Pro license.
It should be noted, that in some cases, the implementation is partial because either a full implementation couldn't be done or I didn't feel like it.

In any event, the principles of the tools should be available at the Basic license level... like Split Layer by Attribute ... which took about 15 years to do so.  

The script can be examined for details in thes old links as well.

 - [Frequency and Statistics](https://community.esri.com/t5/python-blog/free-advanced-tools-frequency-and-statistics/ba-p/902835)
 - [Feature extent to poly features](https://community.esri.com/t5/python-blog/free-advanced-tools-feature-extent-to-poly-features/ba-p/893420)
 - [Feature to points](https://community.esri.com/t5/python-blog/free-advanced-tools-feature-to-point/ba-p/893553)
 - [Convex hulls](https://community.esri.com/t5/python-blog/free-advanced-tools-convex-hulls/ba-p/893549)
 - [Polygons to lines or segments](https://community.esri.com/t5/python-blog/free-advanced-tools-polygons-to-lines-or-segments/ba-p/902811)
 - [Minimum area bounding circles](https://community.esri.com/t5/python-blog/free-advanced-tools-bounding-circles/ba-p/902820)

----
### [**Polygon_line tools**](/PolygonLineTools/README.md)

A collection of tools directed towards work with poly* features.

 - [Polyline tools for ArcGIS Pro](https://community.esri.com/t5/python-blog/polygon-polyline-tools-for-pro/ba-p/904067)
 - [Polygon Polyline tools](https://community.esri.com/t5/python-documents/free-tools-for-arcgis-pro-polygon-polyline-tools/ta-p/917751)

----
### [**PointTools**](/PointTools/README.md)

**Update 2020-05-11**

This folder contains a zip file (PointTools_pro.zip).  The zip contains a toolbox and associated script(s) for working with point features in ArcGIS Pro.

-------------------------
### [**TableTools**](/TableTools/README.md)

**Update 2020-05-17**

The toolbox and script(s) contained here are for working with tabular data in ArcGIS Pro.  Also available at

 - [Table tools for ArcGIS Pro](https://community.esri.com/t5/python-blog/table-tools-for-pro/ba-p/904042)
 - [Table tools](https://community.esri.com/t5/python-documents/free-tools-for-arcgis-pro-table-tools/ta-p/916415)

The foundations is the ability of NumPy to perform rudimentary array operations simply and efficiently on a vectorized basis. This small toolbox implements

1.  Concatenate fields

*  Concatenate fields together regardless of the data type and stripping 'null' values from the fields.  A good demonstration on how to use TableToNumPy array and ExtendTable 

2. Frequency analysis

* I have always thought that the Frequency tool should be available at all license levels.  Give it a try.

3. Table to Text

* This tool facilitates getting a textual representation of table data in a .txt format for presentation and documentation processes.



----
### **Concave hulls**

Contains an implementation for concave hull determination based on k-nearest neighbors (included there).  Also an implementation of convex hulls in case you want to compare differences between the hulls.


-------------------------
### **field_calculator**

This folder is a collection of scripts which contain defs which can be used in other modules or specifically, they can be used in the field calculator in ArcGIS Pro or ArcMap.  The header contains useage information, and a python parser is assumed in all cases.  Check the respective help documentation on how to use the field calculator from within a table or the Calculate Field tool in Pro.


### Feature Extent to Poly\* Features

Produce axis-aligned extent rectangles from polygon/poline features.

### **Frequency and Statistics**

This toolbox provides frequency counts for combinations of tabular data... a combination of the classes in two or more fields.
An option ability to determine basic statistics for another numeric field based on the new classes.

-------------------------
### **Python Code Samples**

A repository for Python Code Samples, largely associated with ArcGIS software but some standalone applications.

Toolboxes are contained in separate folders.  Each folder has a scripts folder containing the scripts associated with the toolbox.

To use, all you need to do is download scripts folder and the toolbox and install in your working path or in a path accessible to all projects.

*Background*

- [Point tools for ArcGIS Pro](https://community.esri.com/t5/python-blog/point-tools-for-pro/ba-p/904043)
- [Point tools](https://community.esri.com/t5/python-documents/free-tools-for-arcgis-pro-point-tools/ta-p/916006)

*General blog posts*

[Geometry ... Stuff to do with points](https://community.esri.com/t5/python-blog/geometry-stuff-to-do-with-points/ba-p/902633)


(1)  Concave hull  [Concave hulls: the elusive container](https://community.esri.com/t5/python-blog/concave-hulls-the-elusive-container/ba-p/902545)

(2)  Convex hull

(3)  Near as table equivalent

(4)  Modify points ... move ... sort

(5)  Create points

(6)  Space points.
      Produce point patterns ensuring a certain minimum spacing between them.

(7)  Minimum Spanning Trees of point sets

(8)  Whatever else I forgot to mention.


-------------------------
### **DATA**

The DATA folder contains a number of numpy arrays (structured arrays specifically) that contain the same fixed proportions of data types but contain a variety of record lengths.  These arrays can be used to standard your testing when you need to work with tabular data with known properties and you need to assess the affect of table size on processing times. 

The arrays can be brought into ArcMap or ArcGIS Pro using the arcpy.da module's **NumPyArrayToTable** and returned to array format using **TableToNumPyArray**

More data constructs will be added as needed.

