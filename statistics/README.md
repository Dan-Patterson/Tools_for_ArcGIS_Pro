## Statistics for ArcGIS Pro ##

Just download the zip file and unzip it where you want your toolbox to appear.  The script will be contained in a folder.

The toolbox was created to provide the capability to calculate descriptive statistics for 'double' fields in tables (geodatabase or dbase).  It essentially enables you to select one, more than one or all fields.  The sum, min, max, median, mean, std dev and var are determined.  

I have intentially left out integer fields from the available options since integers can be used to represent nominal categories or ranks and these statistics are not appropriate.

The base code and the method for calculating values is for a column basis, but the code can quickly be exploited to provide information on a row basis.  I will supplement the capabilities as need arises.
