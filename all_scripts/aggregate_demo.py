"""
Script:  aggregate_demo.py
Modified: 2016-01-30
Author:   Dan.Patterson@carleton.ca
Purpose:  To demonstrate aggregation of raster data without the spatial analyst
    extension.  A sample raster is created and methods to convert an array to
    a raster and vice versa are shown.
Notes:
- RasterToNumPyArray(in_raster, {lower_left_corner},
                         {ncols}, {nrows}, {nodata_to_value})
    arr = arcpy.RasterToNumPyArray(rast) #,arcpy.Point(300000,5025000),10)

- NumPyArrayToRaster(in_array, {lower_left_corner}, {x_cell_size},
                     {y_cell_size}, {value_to_nodata})
    rast = arcpy.NumPyArrayToRaster(a, arcpy.Point(300000,5025000),10,10,-9999)
    rast.save(r"F:\Demos\raster_ops\test_agg") # esri grid, or add tif, jpg etc

"""
import numpy as np
from numpy.lib.stride_tricks import as_strided
from textwrap import dedent
import arcpy

np.set_printoptions(edgeitems=3, linewidth=80, precision=2,
                    suppress=True, threshold=200)

arcpy.env.overwriteOutput = True


def block_a(a, block=(3, 3)):
    """Provide a 2D block view of a 2D array. No error checking made. Columns
    and rows outside of the block are truncated.
    """
    a = np.ascontiguousarray(a)
    r, c = block
    shape = (a.shape[0]//r, a.shape[1]//c) + block
    strides = (r*a.strides[0], c*a.strides[1]) + a.strides
    b_a = as_strided(a, shape=shape, strides=strides)
    return b_a


def agg_demo(n):
    """Run the demo with a preset array shape and content.  See the header
    """
    a = np.random.randint(0, high=5, size=n*n).reshape((n, n))
    rast = arcpy.NumPyArrayToRaster(a, x_cell_size=10)
    agg_rast = arcpy.sa.Aggregate(rast, 2, "MAXIMUM")
    agg_arr = arcpy.RasterToNumPyArray(agg_rast)
    # ,arcpy.Point(300000,5025000),10)
    # --- a_s is the strided array, a_agg_max is the strided array max
    a_s = block_a(a, block=(2, 2))
    a_agg_max = a_s.max(axis=(2, 3))
    # ---
    frmt = """
    Input array... shape {} rows/cols
    {}\n
    Arcpy.sa aggregate..
    {}\n
    Numpy aggregate..
    {}\n
    All close? {}
    """
    yup = np.allclose(agg_arr, a_agg_max)
    print(dedent(frmt).format(a.shape, a, agg_arr, a_agg_max, yup))
    return a, agg_arr, a_s, a_agg_max


if __name__ == "__main__":
    """ Returns the input array, it's aggregation raster from arcpy.sa,
    the raster representation of the raster and the block representation
    and the aggregation array.
    """
    n = 5000
#    a, agg_arr, a_s, a_agg_max = agg_demo(n)




