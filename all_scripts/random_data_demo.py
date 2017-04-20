 # -*- coding: UTF-8 -*-
"""
:Script:   random_data_demo.py
:Author:   Dan.Patterson@carleton.ca
:Original: 2015-08-29
:Modified: 2017-04-04
:
:Purpose:
:  Generate an array containing random data.  Optional fields include:
:  ID,Shape, text, integer and float fields
:
:Functions:
:  colrow_txt, rowcol_txt, rand_text, rand_str, rand_case, rand_int,
:  rand_float, pnts_IdShape
:
:Requires:
:  required imports
:  required constants
:  Shape dtype options:
:    dt_sub = np.dtype([('X','<f8'),('Y','<f8')]) # data type for X,Y fields
:    dt = np.dtype([('ID','<i4'),('Shape',dt_sub)])
:    dt_shp = np.dtype([('ID','<i4'),('Shape','<f8',(2,))])  # short version
:
:Reference:
:  https://geonet.esri.com/blogs/dan_patterson/2016/04/04/
:        numpy-lessons-6-creating-data-for-testing-purposes
:
:Notes:
:-To delete namespace use the following after unwrapping...
:  del [ __name__, __doc__, a, blog_post, colrow_txt, func_run, main,
:  pnts_IdShape, rand_case, rand_float, rand_int, rand_str, rand_text,
:  rfn, rowcol_txt, str_opt, str_opt, wraps]
"""
# ----------------------------------------------------------------------------
# Required imports
import sys
from functools import wraps
import numpy as np
import numpy.lib.recfunctions as rfn

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}

np.set_printoptions(edgeitems=3, linewidth=80, precision=2,
                    suppress=True, threshold=100,
                    formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

# ----------------------------------------------------------------------------
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
    Required:  from functools import wraps
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
    """Produce spreadsheet like labels either 0- or 1-based.
    :N  - number of records/rows to produce.
    :cols/rows - this combination will control the output of the values
    :    cols=2, rows=2 - yields (A0,A1,B0,B1)
    :    as optional classes regardless of the number of records being produced
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
def rowcol_txt(N=10, rows=2, cols=2):
    """Produce array-like labels in a tuple format.
    """
    rc_vals = ["({},{})".format(r, c)
               for c in range(cols) for r in range(rows)]
    rowcol = np.random.choice(rc_vals, N)
    return rowcol


@func_run
def rand_text(N=10, cases=3, vals=str_opt[3]):
    """Generate N samples from the letters of the alphabet denoted by the
    :  number of cases.  If you want greater control on the text and
    :  probability, see rand_case or rand_str.
    :N - samples to produce
    :vals - see str_opt in required constants section
    :cases - number of cases to draw from the vals list
    """
    vals = list(vals)
    txt_vals = np.random.choice(vals[:cases], N)
    return txt_vals


@func_run
def rand_str(N=10, low=1, high=10, vals=str_opt[3]):
    """Returns N strings constructed from 'size' random letters.
    :- create the cases as a list:
    :    string.ascii_lowercase or ascii_uppercase etc
    :- determine how many letters:
    :    Ensure min <= max. Add 1 to max alleviate low==high
    :- shuffle the case list each time through loop
    :vals - see str_opt in required constants section
    """
    vals = list(vals)
    letts = np.arange(min([low, high]), max([low, high]) + 1)  # num letters
    result = []
    for i in range(N):
        np.random.shuffle(vals)
        size = np.random.choice(letts, 1).squeeze()
        result.append("".join(vals[:size]))
    result = np.array(result)
    return result


@func_run
def rand_case(N=10, cases=["Aa", "Bb"], p_vals=[0.8, 0.2]):
    """Generate N samples from a list of classes with an associated probability
    :Ensure that len(cases)==len(p_vals) and  sum(p_values) == 1
    :Small sample sizes will probably not yield the desired p-values
    """
    p = (np.array(p_vals))*N   # convert to integer
    a = [np.repeat(cases[i], p[i]).tolist() for i in range(len(cases))]
    aa = [val for i in range(len(a)) for val in a[i]]
    case_vals = np.array(aa)
    np.random.shuffle(case_vals)
    return case_vals


@func_run
def rand_int(N=10, begin=0, end=10):
    """Generate N random integers within the range begin - end
    """
    int_vals = np.random.randint(begin, end, size=(N))
    return int_vals


@func_run
def rand_float(N=10, begin=0, end=10):
    """Generate N random floats within the range begin - end
    :Technically, N random integers are produced then a random
    :amount within 0-1 is added to the value
    """
    float_vals = np.random.randint(begin, end-1, size=(N))
    float_vals = float_vals + np.random.rand(N)
    return float_vals


@func_run
def pnts_IdShape(N=10, x_min=0, x_max=10, y_min=0, y_max=10, simple=True):
    """Create an array with a nested dtype which emulates a shapefile's
    :  data structure.  This array is used to append other arrays to enable
    :  import of the resultant into ArcMap.  Array construction, after hpaulj
    :- http://stackoverflow.com/questions/32224220/
    :       methods-of-creating-a-structured-array
    """
    Xs = np.random.randint(x_min, x_max, size=N)
    Ys = np.random.randint(y_min, y_max, size=N)
    IDs = np.arange(0, N)
    c_stack = np.column_stack((IDs, Xs, Ys))
    if simple:  # version 1
        dt = [('ID', '<i4'), ('Shape', '<f8', (2,))]  # optional form
        a = np.ones(N, dtype=dt)
        a['ID'] = c_stack[:, 0]
        a['Shape'] = c_stack[:, 1:]                   # this line too
    else:       # version 2
        dt = [('ID', '<i4'), ('Shape', ([('X', '<f8'), ('Y', '<f8')]))]
        a = np.ones(N, dtype=dt)
        a['Shape']['X'] = c_stack[:, 1]
        a['Shape']['Y'] = c_stack[:, 2]
        a['ID'] = c_stack[:, 0]
    return a  # IDs,Xs,Ys,a,dt


def main_demo():
    """Run all the functions with their defaults
    :  To make your own run func, copy and paste the function itself into a
    :  func list.  The decorator handles the printing of results.
    """
    N = 10
    id_shape = pnts_IdShape(N, x_min=0, x_max=10, y_min=0, y_max=10)
    colrow = colrow_txt(N, cols=5, rows=1, zero_based=True)
    rowcol = rowcol_txt(N, rows=5, cols=1)
    txt_fld = rand_text(N, cases=3, vals=str_opt[3])
    str_fld = rand_str(N, low=1, high=10, vals=str_opt[3])
    case1_fld = rand_case(N, cases=['cat', 'dog', 'fish'],
                          p_vals=[0.6, 0.3, 0.1])
    case2_fld = rand_case(N, cases=["Aa", "Bb"], p_vals=[0.8, 0.2])
    int_fld = rand_int(N, begin=0, end=10)
    float_fld = rand_float(N, begin=0, end=10)
    #
    print(("-")*60, "\n")
    fld_names = ['Colrow', 'Rowcol', 'txt_fld', 'str_fld',
                 'case1_fld', 'case2_fld', 'int_fld', 'float_fld']
    fld_data = [colrow, rowcol, txt_fld, str_fld,
                case1_fld, case2_fld, int_fld, float_fld]
    arr = rfn.append_fields(id_shape, fld_names, fld_data, usemask=False)
    print("\nArray generated....\n{!r:}".format(arr))
    return fld_data


def blog_post():
    """sample run"""
    N = 10
    # id_shape = pnts_IdShape(N, x_min=300000, x_max=300500,
    #                          y_min=5000000, y_max=5000500)
    IDs = np.arange(0, N)
    a = np.ones(N, dtype=[('ID', '<i4')])
    a['ID'] = IDs
    case1_fld = rand_case(N, cases=["A", "B", "C", "D"],
                          p_vals=[0.4, 0.2, 0.2, 0.2])
    case3_fld = rand_case(N, cases=["wet ", "dry "], p_vals=[0.5, 0.5])
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
    # id_shape = pnts_IdShape(N, x_min=300000, x_max=301000,
    #                          y_min=5000000, y_max=5001000)
    IDs = np.arange(0, N)
    a = np.ones(N, dtype=[('ID', '<i4')])
    a = IDs
    case1_fld = rand_case(N, cases=["A", "B", "C", "D"],
                          p_vals=[0.3, 0.3, 0.3, 0.1])
    case2_fld = rand_case(N, cases=["A_ ", "B_", "C_"],
                          p_vals=[0.4, 0.3, 0.3])
    case3_fld = rand_case(N, cases=['Hosp', 'Hall'], p_vals=[0.5, 0.5])
    int_fld = rand_int(N, begin=1, end=60)
    int_fld2 = rand_int(N, begin=1, end=300)
    fld_names = ['County', 'Town', 'Facility', 'Time', 'People']
    fld_data = [case1_fld, case2_fld, case3_fld, int_fld, int_fld2]
    arr = rfn.append_fields(a, fld_names, fld_data, usemask=False)
    return arr


# ----------------------------------------------------------------------------
if __name__ == '__main__':
    """create ID,Shape,{txt_fld,int_fld...of any number}
    """
#    arr = run_samples()
    a = blog_post()
#    print a complete function run...or specify one as in below
#    returned = main_demo()
#    print(("-")*60,"\n")
# import arcpy
# a = blog_post()  # do the run if it isn't done
# ..... snip ..... the output
# ..... snip ..... now create the featureclass
# SR_name = 32189  # u'NAD_1983_CSRS_MTM_9'
# SR = arcpy.SpatialReference(SR_name)
# output_shp ='F:/Writing_Projects/NumPy_Lessons/Shapefiles/out.shp'
# arcpy.da.NumPyArrayToFeatureClass(a, output_shp, 'Shape', SR)
#  t2 = np.asarray(t0, dtype=[('F0', '<U2')])
# t1 = np.array([('A', 'a'), ('B', 'b'), ('C', 'c')],
#       dtype=[('F0', '<U2'), ('F1', '<U2')])