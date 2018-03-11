# -*- coding: UTF-8 -*-
"""
:Script:   random_data_demo.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-07-07
:Purpose:
:  Generate an array containing random data.  Optional fields include:
:  ID,Shape, text, integer and float fields
:Functions:
:  colrow_txt, rowcol_txt, rand_text, rand_str, rand_case, rand_int,
:  rand_float, pnts_IdShape
:Requires:
:  required imports
:  required constants
:Shape dtype options:
:  dt_sub = np.dtype([('X','<f8'),('Y','<f8')]) # data type for X,Y fields
:  dt = np.dtype([('ID','<i4'),('Shape',dt_sub)])
:
:-To delete namespace use the following after unwrapping...
: del [ __name__, __doc__, a, blog_post, colrow_txt, func_run, main,
: pnts_IdShape, rand_case, rand_float, rand_int, rand_str, rand_text,
: rfn, rowcol_txt, str_opt, str_opt, wraps]
"""

# -----------------------------------------------------------------------------
# Required imports
import arcpy
from functools import wraps
import numpy as np
import numpy.lib.recfunctions as rfn
np.set_printoptions(edgeitems=5, linewidth=75, precision=2,
                    suppress=True, threshold=5,
                    formatter={'bool': lambda x: repr(x.astype('int32')),
                               'float': '{: 0.2f}'.format})

# -----------------------------------------------------------------------------
# Required constants  ... see string module for others
str_opt = ['0123456789',
           '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~',
           'abcdefghijklmnopqrstuvwxyz',
           'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
           'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
           ]
# -----------------------------------------------------------------------------
# decorator

def func_run(func):
    """Prints basic function information and the results of a run.
    :Required:  from functools import wraps
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("\nFunction... {}".format(func.__name__))
        print("  args.... {}\n  kwargs.. {}".format(args, kwargs))
        print("  docs.... \n{}".format(func.__doc__))
        result = func(*args, **kwargs)
        print("{!r:}\n".format(result))  # comment out if results not needed
        return result  # for optional use outside.
    return wrapper

# -----------------------------------------------------------------------------
# functions


@func_run
def colrow_txt(N=10, cols=2, rows=2, zero_based=True):
    """  Produce spreadsheet like labels either 0- or 1-based.
    :N  - number of records/rows to produce.
    :cols/rows - this combination will control the output of the values
    :  cols=2, rows=2 - yields (A0,A1,B0,B1)
    :  as optional classes regardless of the number of records being produced
    :zero-based - True for conventional array structure,
    :             False for spreadsheed-style classes
    """
    if zero_based:
        start = 0
    else:
        start = 1
        rows = rows + 1
    UC = (list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))[:cols]  # see constants
    dig = (list('0123456789'))[start:rows]
    cr_vals = [c + r for r in dig for c in UC]
    colrow = np.random.choice(cr_vals, N)
    return colrow


@func_run
def rowcol_txt(N=10,rows=2,cols=2):
    """  Produce array-like labels in a tuple format.
    """
    rc_vals = ["({},{})".format(r, c)
               for c in range(cols)
               for r in range(rows)]
    rowcol = np.random.choice(rc_vals, N)
    return rowcol


@func_run
def rand_text(N=10, cases=3, vals=str_opt[3]):
    """  Generate N samples from the letters of the alphabet denoted by the
    :number of cases.  If you want greater control on the text and
    :probability, see rand_case or rand_str.
    :vals:  see str_opt in required constants section
    """
    vals = list(vals)
    txt_vals = np.random.choice(vals[:cases], N)
    return txt_vals


@func_run
def rand_str(N=10,low=1,high=10,vals=str_opt[3]):
    """  Returns N strings from 'size' random letters.
    :- create the cases as a list:  string.ascii_lowercase or ascii_uppercase
    :- determine how many letters.
    :  Ensure min <= max. Add 1 to max alleviate low==high
    :- shuffle the case list each time through loop
    """
    vals = list(vals)
    letts = np.arange(min([low, high]), max([low, high]) + 1)  # num letters
    result = []
    for i in range(N):
        np.random.shuffle(vals)
        size = np.random.choice(letts, 1)
        result.append("".join(vals[:size]))
    result = np.array(result)
    return result

@func_run
def rand_case(N=10,cases=["Aa","Bb"],p_vals=[0.8,0.2]):
    """  Generate N samples from a list of classes with an associated probability
    ensure: len(cases)==len(p_vals) and  sum(p_values) == 1
    small sample sizes will probably not yield the desired p-values
    """
    p = (np.array(p_vals))*N   # convert to integer
    kludge = [np.repeat(cases[i],p[i]).tolist() for i in range(len(cases))]
    case_vals = np.array([ val for i in range(len(kludge)) for val in kludge[i]])
    np.random.shuffle(case_vals)
    return case_vals


@func_run
def rand_int(N=10, begin=0, end=10):
    """  Generate N random integers within the range begin - end
    """
    int_vals = np.random.random_integers(begin, end, size=(N))
    return int_vals


@func_run
def rand_float(N=10, begin=0, end=10, deci=2):
    """  Generate N random floats within the range begin - end
    :Technically, N random integers are produced then a random
    :amount within 0-1 is added to the value
    """
    float_vals = np.random.randint(begin, end, size=(N))
    float_vals = np.around(float_vals + np.random.rand(N), deci)
    return float_vals


def rand_norm(N=10, avg_=10, st_dev=1, deci=2):
    """  Generate N random floats within the range begin - end
    :Technically, N random integers are produced then a random
    :amount within 0-1 is added to the value
    """
    float_vals = np.random.normal(avg_, st_dev, size=(N))
    float_vals = np.around(float_vals + np.random.rand(N), deci)
    return float_vals



@func_run
def pnts_IdShape(N=10, x_min=0, x_max=10, y_min=0, y_max=10, simple=True):
    """  Create an array with a nested dtype which emulates a shapefile's
    :data structure.  This array is used to append other arrays to enable
    :import of the resultant into ArcMap.  Array construction, after hpaulj
    :http://stackoverflow.com/questions/32224220/
    :     methods-of-creating-a-structured-array
    """
    Xs = np.random.randint(x_min, x_max + 1, size=N)
    Ys = np.random.randint(y_min, y_max + 1, size=N)
    IDs = np.arange(0, N)
    c_stack = np.column_stack((IDs, Xs, Ys))
    if simple:     # version 1
        dt = [('ID', '<i4'), ('Shape', '<f8', (2,))]  # short version,
        a = np.ones(N, dtype=dt)
        a['ID'] = c_stack[:, 0]
        a['Shape'] = c_stack[:, 1:]                   # this line too
    else:          # version 2
        dt = [('ID', '<i4'), ('Shape', ([('X', '<f8'), ('Y', '<f8')]))]
        a = np.ones(N, dtype=dt)
        a['Shape']['X'] = c_stack[:, 1]
        a['Shape']['Y'] = c_stack[:, 2]
        a['ID'] = c_stack[:, 0]
    return a  # IDs, Xs, Ys, a, dt


def main_demo():
    """Run all the functions with their defaults
    :  To make your own run func, copy and paste the function itself into
    :   a func list.  The decorator handles the printing of results
    """
    N = 10
    id_shape = pnts_IdShape(N, x_min=0, x_max=10, y_min=0, y_max=10)
    colrow = colrow_txt(N, cols=5, rows=1, zero_based=True),
    rowcol = rowcol_txt(N, rows=5, cols=1),
    txt_fld = rand_text(N, cases=3, vals=str_opt[3]),
    str_fld = rand_str(N, low=1, high=10, vals=str_opt[3]),
    case1_fld = rand_case(N, cases=['cat', 'dog', 'fish'],
                          p_vals=[0.6, 0.3, 0.1]),
    case2_fld = rand_case(N, cases=["Aa", "Bb"], p_vals=[0.8, 0.2])
    int_fld = rand_int(N, begin=0, end=10),
    float_fld = rand_float(N, begin=0, end=10, deci=2)
    #
    print(("\n" + "-"*60 + "\n"))
    fld_names = ['Colrow', 'Rowcol', 'txt_fld', 'str_fld',
                 'case1_fld', 'case2_fld', 'int_fld', 'float_fld']
    fld_data = [colrow, rowcol, txt_fld, str_fld,
                case1_fld, case2_fld, int_fld, float_fld]
    arr = rfn.append_fields(id_shape, fld_names, fld_data, usemask=False)
    print("\nArray generated....\n{!r:}".format(arr))
    return fld_data


def blog_post():
    """Sample run
    |  import arcpy
    |  a = blog_post()  # do the run if it isn't done
    |  # ..... snip ..... the output
    |  # ..... snip ..... now create the featureclass
    |  SR_name = 32189  # u'NAD_1983_CSRS_MTM_9'
    |  SR = arcpy.SpatialReference(SR_name)
    |  output_shp ='F:/Writing_Projects/NumPy_Lessons/Shapefiles/out.shp'
    |  arcpy.da.NumPyArrayToFeatureClass(a, output_shp, 'Shape', SR)"""
    N = 10
    #id_shape = pnts_IdShape(N, x_min=300000, x_max=300500,y_min=5000000,y_max=5000500)
    IDs = np.arange(0,N)
    a = np.ones(N, dtype=[('ID', '<i4')])
    a=IDs
    case1_fld = rand_case(N, cases=["A", "B", "C", "D"],
                          p_vals=[0.4, 0.2, 0.2, 0.2])
    case3_fld = rand_case(N, cases=["wet ","dry "],
                          p_vals=[0.5, 0.5])
    case4_fld = rand_case(N, cases=['cat', 'dog', 'fish'],
                          p_vals=[0.6, 0.3, 0.1])
    case2_fld = rand_case(N, cases=["Aa", "Bb", "Cc", "Dd"],
                          p_vals=[0.4, 0.2, 0.2, 0.2])
    int_fld = rand_int(N, begin=0, end=10)
    fld_names = ['Place', 'State', 'Case', 'Pet', 'Number']
    fld_data = [case1_fld, case2_fld, case3_fld, case4_fld, int_fld]
    arr = rfn.append_fields(a, fld_names, fld_data, usemask=False)
    return arr


def run_samples():
    """sample run"""
    N = 1000
    # id_shape = pnts_IdShape(N, x_min=300000, x_max=300500,
    #                         y_min=5000000, y_max=5000500)
    IDs = np.arange(0,N)
    a = np.ones(N, dtype=[('ID', '<i4')])
    a=IDs
    case1_fld = rand_case(N,cases=["A", "B", "C", "D"],
                          p_vals=[0.3, 0.3, 0.3, 0.1])
    case2_fld = rand_case(N,cases=["A_ ", "B_", "C_"],
                          p_vals=[0.4, 0.3, 0.3])
    case3_fld = rand_case(N,cases=['Hosp', 'Hall'],
                          p_vals=[0.5, 0.5])
    int_fld = rand_int(N, begin=1, end=60)
    int_fld2 = rand_int(N, begin=1, end=300)
    fld_names = ['County', 'Town', 'Facility', 'Time', 'People']
    fld_data = [case1_fld, case2_fld, case3_fld, int_fld, int_fld2]
    arr = rfn.append_fields(a, fld_names, fld_data, usemask=False)
    return arr


def stats_demo(N=10, cols=10):
    """Create points with 12 columns of random float values.
    :  N - number of points
    :  SR - GCS North American 1983 CSRS, WKID 4617
    :  my link....
    :  https://stackoverflow.com/questions/43442415/
    :    cannot-perform-reduce-with-flexible-type
    """
    a = pnts_IdShape(N, x_min=300000, x_max=310000,
                     y_min=5025000, y_max=5035000, simple=True)
    fld_dt = a.dtype.descr
    col_names = tuple([('C_' + str(i)) for i in range(cols)])
    xtra_dt = [(i, '<f8') for i in col_names]  # modify dtype if desired
    #fld_dt.extend(xtra_dt)
    fld_data = []
    for i in range(cols):
        vals = rand_norm(N, avg_=10, st_dev=1, deci=2)  # normal
        #vals = rand_float(N, begin=0, end=10)  # uncomment for random
        #vals = np.zeros(N)
        #vals.fill(i)
        vals[:100] = np.nan
        np.random.shuffle(vals)
        fld_data.append(vals)
    b = rfn.append_fields(a, names=col_names, data=fld_data, usemask=False)
    return a, b, col_names, fld_data


def cal_stats(in_fc, col_names):
    """Calculate stats for an array with nodata (nan, None) in the columns
    :in_fc - input featureclass or table
    :col_names - the columns... numeric (floating point, double)
    """
    a = arcpy.da.FeatureClassToNumPyArray(in_fc, col_names)  # "*")
    b = a.view(np.float).reshape(len(a), -1)
    r_sum = np.nansum(b, axis=1)   # by row
    c_sum = np.nansum(b, axis=0)   # by column
    r_avg = np.nanmean(b, axis=1)  # by row
    c_avg = np.nanmean(b, axis=0)  # by column
    stck = np.vstack((col_names, c_sum, c_avg)).T
    print(stck)
    return stck, col_names, c_sum, c_avg

if __name__ == '__main__':
    """random_data_demo
    Create a point shapefile containing fields:
      ID, Shape, {txt_fld,int_fld...of any number}
    """
    print("\nRunning... {}\n".format(__file__))
    # uncomment an option below
    N = 2000
    cols = 3
    a, b, col_names, fld_data = stats_demo(N=N, cols=cols)
    out_fc = r'C:\GIS\Tools_scripts\Statistics\Stats_demo_01.gdb\pnts_2000_norm'
    SR = arcpy.SpatialReference('NAD 1983 CSRS MTM  9')
    arcpy.da.NumPyArrayToFeatureClass(b, out_fc, ['Shape'], SR)
#    stck, col_names, c_sum, c_avg = cal_stats(out_fc, col_names)
#    dts = [('Column', '<U15'), ('Col_sum', '<f8'), ('Col_avg', '<f8')]
#    args = [col_names, c_sum, c_avg]
#    z = np.empty(shape=(cols,), dtype=dts)
#    for i in range(len(args)):
#        z[z.dtype.names[i]] = args[i]
#    c = arcpy.da.FeatureClassToNumPyArray(out_fc, col_names) # "*")
#    good_flds = [i for i in c.dtype.names if c[i].dtype.kind in ['f']]
#    by_col = [(i, np.nanmean(c[i])) for i in good_flds
#              if c[i].dtype.kind in ('i', 'f')]
#    bc2 = c.view(np.float).reshape(len(c), -1)
#    np.nanmean(bc2, axis=1)  # by row
#    np.nanmean(bc2, axis=0)  # by column


#    np.vstack((c['C_0'], c['C_1'])).T
    #arr = run_samples()
    #a = blog_post()
    #returned = main_demo()  # print a complete function run...or specify one as in below
    print(("\n" + "-"*60 +"\n"))
