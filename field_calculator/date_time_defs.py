# -*- coding: UTF-8 -*-
"""
:Script:   date_time_defs.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-06-17
:Purpose:  date_time defs
:  Can be used in the field calculator in ArcMap or ArcGIS Pro
:---------------------------------------------------------------------:
"""
from datetime import datetime

# -*- coding: utf-8 -*-
"""
Script:  DDddd_DMS_convert.py
Author:  Dan.Patterson@carleton.ca

Purpose: Formatting stuff
Notes:
- to use in the field calculator, set the parser to Python and
  use ' !fieldname! ' in the expression box
- degree sign, ° ... ALT 248 on the numeric keypad
- utf-8  u'\N{DEGREE SIGN}' or u'\xb0'
"""

def ddd_dms(a):
    """Decimal degree to DMS format"""
    sign = [-1, 1][a > 0]
    DD, dd = divmod(a, sign)
    MM, ss = divmod(dd*60, sign)
    SS, ssss = divmod(ss*60, 1)
    frmt = "{:0= 4}" + u'\xb0' + " {:=2}' {:0=7.4f}\" "
    DMS = frmt.format(int(sign*DD), int(MM), sign*ss*60)
    return DMS


def get_date(fld):
    """input a date field, strip off the time and format
    :Useage  - get_date(!FieldName!)
    :From    - 2017-06-17 20:35:58.777353 ... 2017-06-17
    :Returns -2017-06-17
    """
    if fld is not None:
        lst = [int(i) for i in (str(fld).split(" ")[0]).split("-")]
        return "{}-{:02.0f}-{:02.0f}".format(*lst)
    else:
        return None


def get_time(fld):
    """input a date field, strip off the date and format
    :From    - 2017-06-17 20:35:58.777353 ... 2017-06-17
    :Returns - 20 h 35 m  58.78 s
    """
    if fld is not None:
        lst = [float(i) for i in (str(fld).split(" ")[1]).split(":")]
        return "{:02.0f} h {:02.0f} m {: 5.2f} s".format(*lst)
    else:
        return None

def _demo():
    """
    :other format options
    :mess with the order of line 44 for different outputs
    """
    today = datetime.today()
    print('\n_demo def...\nISO     :', today)
    print('format(): {:%a %b %d %H:%M:%S %Y}'.format(today))
    #return today

#--------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """ A simple test """
    n = str(datetime.now())

    print("\nget_date def...\nDate from {} ... {}".format(n, get_date(n)))
    print("\nget_time def...\nDate from {} ... {}".format(n, get_time(n)))

    _demo()

    vals = [45.501234567890, -45.501234567890,
            145.501234567890, -145.501234567890]
    print("\nddd_dms def...\nTest run with a in vals")
    for a in vals:
        print("{:> 22.12f} ... {!s:>20}".format(a, ddd_dms(a)))
