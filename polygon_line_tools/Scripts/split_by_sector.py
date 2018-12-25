# -*- coding: UTF-8 -*-
"""
split_by_sector
===============

Script   :  split_by_sector.py

Author   :  Dan.Patterson@carleton.ca

Modified :  2018-08-30

Purpose  :  tools for working with numpy arrays

Source :

References:
----------

`<https://stackoverflow.com/questions/3252194/numpy-and-line-intersections>`_.
`<https://community.esri.com/message/627051?commentID=627051#comment-627051>`
`<https://community.esri.com/message/779043-re-how-to-divide-irregular-
polygon-into-equal-areas-using-arcgis-105?commentID=779043#comment-779043>`

This is a good one
`<https://tereshenkov.wordpress.com/2017/09/10/dividing-a-polygon-into-a-given
-number-of-equal-areas-with-arcpy/>`
"""
# ---- imports, formats, constants ----
import sys
import numpy as np
from arcpytools_plt import(tweet, fc_info, get_polys)
import arcpy


ft={'bool': lambda x: repr(x.astype('int32')),
    'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2,
                    suppress=True, threshold=100,
                    formatter=ft)

script = sys.argv[0]

__all__ = ["plot_",
           "to_polygon",
           "cal_sect",
           "sectors",
           "process"
           ]
#---- functions ----

def plot_(pnts):
    """plot a circle, arc sector etc
    """
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    #x_min = pnts[:,0].min()
    #x_max = pnts[:,0].max()
    #y_min = pnts[:,1].min()
    #y_max = pnts[:,1].max()
    fig, ax = plt.subplots()
    patches = []
    # Points need to form a closed loopset closed to True if your 1st and
    # last pnt aren't equal.
    for i in pnts:
        polygon = Polygon(i, closed=False)
        patches.append(polygon)
    p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=1.0)
    colors = 100*np.random.rand(len(patches))
    p.set_array(np.array(colors))
    #ax.set_xlim(x_min-0.5, x_max+0.5)  # (x_min, x_max)
    #ax.set_ylim(y_min-0.5, y_max+0.5)  # y_min, y_max)
    ax.add_collection(p)
    plt.axis('equal')
    plt.show()
#    plt.close()
    #return fig, ax


def to_polygon(pnts):
    """create polygons from list or array pairs.  pass [pnts] if you are only
    creating one polygon.  multiple polygons will already be represented as a
    list of lists of points.
    In short, it expects a 3d array or its list equivalent
        """
    polygons = []
    for pair in pnts:
        pl = arcpy.Polygon(arcpy.Array([arcpy.Point(*xy) for xy in pair]))
        polygons.append(pl)
    return polygons


def cal_sect(poly, cutters, pnts, factor):
    """Calculate the areas
    """
    # ---- have to intersect here
    cuts = [cutters[i].intersect(poly, 4) for i in range(len(cutters))]
    tot_area = poly.area
    fract_areas = np.array([c.area/tot_area for c in cuts])
    c_sum = np.cumsum(fract_areas)
    f_list = np.linspace(0.0, 1.0, factor, endpoint=False)
    idxs = [np.argwhere(c_sum <= i)[-1][0] for i in f_list[1:]]
    splits = np.split(cuts, idxs, axis=0)
    return idxs, splits


def sectors(radius=100, theta=0.5, xc=0.0, yc=0.0):
    """Create sectors radiating out from the geometry center.  A circle is
    created and the geometry is parsed adding the center point to each set
    of points to form a polygon.  The first and last point can be duplicated
    if needed.
    """
    def xy(x_s, y_s):
        z = np.zeros((len(x_s), 2), 'float')
        z[:, 0] = x_s
        z[:, 1] = y_s
        return z

    # ---- Make a circle first ----
    angles = np.deg2rad(np.arange(180.0, -180.0-theta, step=-theta))
    x_s = radius*np.cos(angles)     # X values
    y_s = radius*np.sin(angles)     # Y values
    pnts = xy(x_s, y_s)
    # ----
    fr = pnts[:-1]
    too = pnts[1:]
    cent = np.array([[xc, yc]])
    z = np.array([[0., 0.]])
    zs = z.repeat(len(fr), axis=0)
    sect = np.array(list(zip(zs, fr, too))) + cent
    pnts = pnts + cent
    return sect, pnts


def process(in_polys):
    """Process the splits

    Parameters:
    -----------
    in_fc: text
        input featureclass
    out_fc: text
        output featureclass
    s_fac: integer
        split factor

    Requires:
    ---------
        sectors, to_polygon, cal_sect

    Notes:
    ------
    You can fine-tune the analysis by changing the theta value from 1.0 to a
    smaller value 360 circle sectors result when theta = 1, 720 when it equal
    0.5.  Processing time changes minimally on a per polygon basis.
    """
    result_ = []
    for i in range(len(in_polys)):
        poly = in_polys[i]
        ext = max(poly.extent.width, poly.extent.height)
        xc, yc = cent = [poly.centroid.X, poly.centroid.Y]
        sect, pnts = sectors(radius=ext, theta=0.5, xc=xc, yc=yc)  # theta=1
        cutters = to_polygon(sect)
        idxs, splits = cal_sect(poly, cutters, pnts, s_fac)
        ps = np.split(pnts, np.array(idxs)+1)
        new_polys = [np.vstack((cent, ps[i-1], ps[i][0], cent))
                     for i in range(0, len(ps))]
        r = to_polygon(new_polys)
        rs = [i.intersect(poly, 4) for i in r]
        #p = arcpy.Polygon(r[0])
        result_.extend(rs)
    return result_


# ---- demo and tool section -------------------------------------------------
# large Canada  Can_0_sp_lcc
if len(sys.argv) == 1:
    testing = True
    in_pth = script.split("/")[:-2] + ["Polygon_lineTools.gdb"]
    in_fc = "/".join(in_pth) + "/shapes_mtm9"  # "/Big"  #
    out_fc = "/".join(in_pth) + "/s1"
    s_fac = 4
else:
    testing = False
    in_fc = sys.argv[1]
    out_fc = sys.argv[2]
    s_fac = int(sys.argv[3])

# ---- for both
#
shp_fld, oid_fld, shp_type, SR = fc_info(in_fc)

# ---- instant bail if not projected
if SR.type == 'Projected':
    in_polys, out_ids = get_polys(in_fc)
    out_polys = process(in_polys)
    if not testing:
        if arcpy.Exists(out_fc):
            arcpy.Delete_management(out_fc)
        arcpy.CopyFeatures_management(out_polys, out_fc)
        out_ids = np.repeat(out_ids, s_fac)
        id_fld = np.zeros((len(out_polys),),
                          dtype=[("key", "<i4"), ("Old_ID", "<i4")])
        id_fld["key"] = np.arange(1, len(out_polys) + 1)
        id_fld["Old_ID"] = out_ids
        arcpy.da.ExtendTable(out_fc, oid_fld, id_fld, "key")
else:
    msg = """
    -----------------------------------------------------------------
    Input data is not in a projected coordinate system....
    bailing...
    -----------------------------------------------------------------
    """
    tweet(msg)

#----------------------
if __name__=="__main__":
    """Uncomment what you want to see"""
    #print("Script... {}".format(script))
    #circ_pnts = _circle(radius=1, theta=30, xc=5, yc=5)
    #print("\ncircle points...\n{}".format(circ_pnts))
    #arc_pnts = _arc(radius=10, start=0, stop=90.5, step=5, xc=0.0, yc=0.0)
    #print("\narc points...\n{}".format(arc_pnts))
    #pnts = arc_sector()
    #pnts = buffer_ring()
    #multi_sector_demo()
    #multiring_buffer_demo()

