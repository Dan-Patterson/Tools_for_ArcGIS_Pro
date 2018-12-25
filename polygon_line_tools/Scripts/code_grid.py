# -*- coding: utf-8 -*-
"""
code_grid
=========

Script :   code_grid.py

Author :   Dan_Patterson@carleton.ca

Modified : 2018-08-19

Purpose:  produce a spreadsheet-like numbering system for 'grid' cells

This use padding A01 to facilitate sorting.
If you want a different system change
>>> >>> "{}{}".format(UC[c], r)    # A1 to whatever, no padding
>>> "{}{:02.0f}".format(UC[c], r)  # A01 to ..99
>>> "{}{:03.0f}".format(UC[c], r)  # A001 to A999
>>> # etc
>>> c0 = code_grid(cols=5, rows=3, zero_based=False, shaped=True, bottom_up=False)
[['A01' 'B01' 'C01' 'D01' 'E01']
 ['A02' 'B02' 'C02' 'D02' 'E02']
 ['A03' 'B03' 'C03' 'D03' 'E03']]

---------------------------------------------------------------------
"""

import sys
import numpy as np

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

def code_grid(cols=1, rows=1, zero_based=False, shaped=True, bottom_up=False):
    """produce spreadsheet like labelling, either zero or 1 based
    :  zero - A0,A1  or ones - A1, A2..
    :  dig = list('0123456789')  # string.digits
    : import string .... string.ascii_uppercase
    """
    alph = list(" ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    UC = [("{}{}").format(alph[i], alph[j]).strip()
          for i in range(27)
          for j in range(1,27)]
    z = [1, 0][zero_based]
    rc = [1, 0][zero_based]
    c = ["{}{:02.0f}".format(UC[c], r) # pull in the column heading
         for r in range(z, rows + rc)  # label in the row letter
         for c in range(cols)]         # label in the row number
    c = np.asarray(c)
    if shaped:
        c = c.reshape(rows, cols)
        if bottom_up:
            c = np.flipud(c)
    return c

# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
    c0 = code_grid(cols=3, rows=100, zero_based=False, shaped=True, bottom_up=False)
    print(c0)
