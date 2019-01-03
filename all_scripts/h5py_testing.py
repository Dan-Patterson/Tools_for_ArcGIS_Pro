# -*- coding: UTF-8 -*-
"""
:Script:   .py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2018-xx-xx
:Purpose:  tools for working with numpy arrays
:Useage:
:
:References:
:
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
import numpy as np
import h5py

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

v = r'C:\Git_Dan\arraytools\Data\sample_1000.npy'
a = np.load(v)
v_h = r'C:\Git_Dan\arraytools\Data\sample_100K.hdf5'
a.dtype = [('OBJECTID', '<i4'), ('f0', '<i4'), ('County', '<S2'),
           ('Town', '<S6'), ('Facility', '<S8'), ('Time', '<i4')]
shp = a.shape
with h5py.File(v_h, "w") as f:
    dset = f.create_dataset("sample100K", shp, dtype=dt, data=a)


def _demo():
    """
    : -
    """
    pass


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
    _demo()

