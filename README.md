
## __***Tools for ArcGIS Pro***__


### **Python Code Samples**

A repository for Python Code Samples, largely associated with ArcGIS software but some standalone applications.

Toolboxes are contained in separate folders.  Each folder has a scripts folder containing the scripts associated with the toolbox.

To use, all you need to do is download scripts folder and the toolbox and install in your working path or in a path accessible to all projects.


### **field_calculator**

This folder is a collection of scripts which contain defs which can be used in other modules or specifically, they can be used in the field calculator in ArcGIS Pro or ArcMap.  The header contains useage information, and a python parser is assumed in all cases.  Check the respective help documentation on how to use the field calculator from within a table or the Calculate Field tool in Pro.


### **point_tools**

This folder contains a toolbox and associated script(s) for ArcGIS Pro to work with points.  Also found at
http://www.arcgis.com/home/item.html?id=f96ede37dcd04c2e96dc903a4ce26244

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

### **table_tools**

The toolbox and script(s) contained here are for working with tabular data in ArcGIS Pro.  Also available at
http://www.arcgis.com/home/item.html?id=90d9ca933e8c4b96bf341a20ae1f2514

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

