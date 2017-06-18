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

# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """ A simple test """
    n = str(datetime.now())
    print("\nDate from {} ... {}".format(n, get_date(n)))
    print("\nDate from {} ... {}".format(n, get_time(n)))
