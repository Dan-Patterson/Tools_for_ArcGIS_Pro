# -*- coding: UTF-8 -*-
"""
:Script:   .py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-xx-xx
:Purpose:  tools for working with numpy arrays
:Useage:
:
:References:
:
:---------------------------------------------------------------------:
"""
def pad_date(fld):
    """input a date field, strip off the time and format"""
    if fld is not None:
        lst = [int(i) for i in (str(fld).split(" ")[0]).split("-")]
        return "{}-{:02.0f}-{:02.0f}".format(*lst)
    else:
        return None

# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """ A simple test """
    from datetime import datetime
    n = str(datetime.now())
    print(pad_date(n))