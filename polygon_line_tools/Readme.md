# polygon_line_tools

Tools for working with polygon and polyline tools

#Update 2018-09-12#

Added split by area to the split poly* features tool.  I will update with images and explanation when I get a chance.

## Densify by Factor and Features to Points

Densify poly`*` features by a factor between the existing nodes.  The points on the image show the result when converted to points.

<a href="url"><img src="https://github.com/Dan-Patterson/tools_pro/blob/master/Polygon_lineTools/Images/Densify.png" align="center" width="400" ></a>

The poly* to points retains the original poly feature ID then adds a new column with the point ID for that feature.  The labelling was simply a concatenation of the two bits of information.

<a href="url"><img src="https://github.com/Dan-Patterson/tools_pro/blob/master/Polygon_lineTools/Images/poly_pnts.png" align="center" width="400" ></a>

## Split Poly`*` Features`*`

Split poly`*` features based on a width or height factor.

<a href="url"><img src="https://github.com/Dan-Patterson/tools_pro/blob/master/polygon_line_tools/images/Split_poly_features.png" align="center" width="200" ></a>

## Sampling Grids

Produce sampling grids as rectangles, hexagons (two types) and one triangle form.
Rotation is supported and the features are given a spreadsheet-line labeling code.

<a href="url"><img src="https://github.com/Dan-Patterson/tools_pro/blob/master/Polygon_lineTools/Images/sampling_grid_results.png" align="center" width="400" ></a>


<a href="url"><img src="https://github.com/Dan-Patterson/tools_pro/blob/master/Polygon_lineTools/Images/sampling_grids.png" align="center" width="400" ></a>
