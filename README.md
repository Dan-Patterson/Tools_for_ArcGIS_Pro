
## Tools for ArcGIS Pro

----

This folder contains various toolboxes for working with feature data in ArcGIS Pro.

**NOTE**

The main testing repository for this package is *npgeom* on the main landing page.  I move finished materials here when I have completed updates or additions to it.

----

### [**Free tools**](/Free_Tools/README.md)

Free tools are sample tools that use numpy and arcpy to implement tools that normally require a Standard or Advanced ArcGIS Pro license.
It should be noted, that in some cases, the implementation is partial because either a full implementation couldn't be done or I didn't feel like it.  In any event, the principles of the tools should be available at the Basic license level... like Split Layer by Attribute ... which took about 15 years to do so.  



The script can be examined for details in thes old links as well.

https://community.esri.com/blogs/dan_patterson/2019/06/24/free-tools-frequency-and-statistics

https://community.esri.com/blogs/dan_patterson/2019/06/26/free-tools-feature-extent-to-poly-features

https://community.esri.com/blogs/dan_patterson/2019/07/26/free-tools-feature-to-point

https://community.esri.com/blogs/dan_patterson/2019/07/23/free-tools-convex-hulls

https://community.esri.com/blogs/dan_patterson/2019/08/08/free-tools-polygons-to-lines-or-segments

https://community.esri.com/blogs/dan_patterson/2019/08/08/free-tools-bounding-circles


### **Concave hulls**

Contains an implementation for concave hull determination based on k-nearest neighbors (included there).  Also an implementation of convex hulls in case you want to compare differences between the hulls.

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

-------------------------
### **Polygon_line tools**

A collection of tools directed towards work with poly* features.

https://community.esri.com/people/danretired/blog/2020/05/19/polygonpolyline-tools-for-pro
https://community.esri.com/docs/DOC-14930-free-tools-for-arcgis-pro-polygon-polyline-tools

-------------------------
### **field_calculator**

This folder is a collection of scripts which contain defs which can be used in other modules or specifically, they can be used in the field calculator in ArcGIS Pro or ArcMap.  The header contains useage information, and a python parser is assumed in all cases.  Check the respective help documentation on how to use the field calculator from within a table or the Calculate Field tool in Pro.

-------------------------
### **[PointTools](PointTools/README.md)**

**Update 2020-05-11**

This folder contains a zip file (PointTools_pro.zip).  The zip contains a toolbox and associated script(s) for working with point features in ArcGIS Pro.

*Background*

https://community.esri.com/people/danretired/blog/2020/05/15/point-tools-for-pro
https://community.esri.com/docs/DOC-14926-free-tools-for-arcgis-pro-point-tools

*General blog posts*

https://community.esri.com/blogs/dan_patterson/2018/03/01/geometry-stuff-to-do-with-points


(1)  Concave hull  https://community.esri.com/blogs/dan_patterson/2018/03/11/concave-hulls-the-elusive-container

(2)  Convex hull

(3)  Near as table equivalent

(4)  Modify points ... move ... sort

(5)  Create points

(6)  Space points.
      Produce point patterns ensuring a certain minimum spacing between them.

(7)  Minimum Spanning Trees of point sets

(8)  Whatever else I forgot to mention.

-------------------------
### **TableTools**

**Update 2020-05-17**

The toolbox and script(s) contained here are for working with tabular data in ArcGIS Pro.  Also available at

https://community.esri.com/people/danretired/blog/2020/05/18/free-tools-for-arcgis-pro-table-tools
https://community.esri.com/docs/DOC-14928-free-tools-for-arcgis-pro-table-tools

The foundations is the ability of NumPy to perform rudimentary array operations simply and efficiently on a vectorized basis. This small toolbox implements

1.  Concatenate fields

*  Concatenate fields together regardless of the data type and stripping 'null' values from the fields.  A good demonstration on how to use TableToNumPy array and ExtendTable 

2. Frequency analysis

* I have always thought that the Frequency tool should be available at all license levels.  Give it a try.

3. Table to Text

* This tool facilitates getting a textual representation of table data in a .txt format for presentation and documentation processes.


-------------------------
### **DATA**

The DATA folder contains a number of numpy arrays (structured arrays specifically) that contain the same fixed proportions of data types but contain a variety of record lengths.  These arrays can be used to standard your testing when you need to work with tabular data with known properties and you need to assess the affect of table size on processing times. 

The arrays can be brought into ArcMap or ArcGIS Pro using the arcpy.da module's **NumPyArrayToTable** and returned to array format using **TableToNumPyArray**

More data constructs will be added as needed.


### **all_scripts**

The scripts for the toolboxes as separate entities for those that just need the script for other purposes.  Be aware of any dependencies in the scripts if they are specified.

