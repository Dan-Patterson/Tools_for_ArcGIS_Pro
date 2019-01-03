# -*- coding: UTF-8 -*-
"""
:Script:   .py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2018-xx-xx
:Purpose:  tools for working with numpy arrays
:Useage:
:
:References:
: https://www.lfd.uci.edu/~gohlke/code/tifffile.py.html
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----
import sys
from textwrap import dedent
import numpy as np
import io


ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script


def save_as_txt(fname, a):
    """Save a numpy array as a text file determining the format from the
    :  data type
    :
    :Reference:  from numpy savetxt
    :----------
    : savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='',
    :         footer='', comments='# ')
    : fmt - '%[flag]width[.precision]specifier'
    : fmt='%.18e'
    """
    dt_kind = a.dtype.kind
    l_sze = max(len(str(a.max())), len(str(a.min())))
    frmt = '%{}{}'.format(l_sze, dt_kind)
    hdr = "dtype: {} shape: {}".format(a.dtype.str, a.shape)
    np.savetxt(fname, a, fmt=frmt, delimiter=' ',
               newline='\n', header=hdr, footer='', comments='# ')


def _read_bytes(fp, size, error_template="ran out of data"):
    """
    Read from file-like object until size bytes are read.
    Raises ValueError if not EOF is encountered before size bytes are read.
    Non-blocking objects only supported if they derive from io objects.

    Required as e.g. ZipExtFile in python 2.6 can return less data than
    requested.
    Requires:  import io #####

    _read_bytes(open(fp, 'rb'), 100)
    b"\x93NUMPY\x01\x00F\x00{'descr': '<i4',
    'fortran_order': False, 'shape': (10, 10), }        \n
\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00"
    """
    data = bytes()
    while True:
        # io files (default in python3) return None or raise on
        # would-block, python2 file will truncate, probably nothing can be
        # done about that.  note that regular files can't be non-blocking
        try:
            r = fp.read(size - len(data))
            data += r
            if len(r) == 0 or len(data) == size:
                break
        except io.BlockingIOError:
            pass
    if len(data) != size:
        msg = "EOF: reading %s, expected %d bytes got %d"
        raise ValueError(msg % (error_template, size, len(data)))
    else:
        return data


def read_npy(fp, prn=False):
    """ read an npy file quickly
    : fp = file path
    :
    : file = "c:/temp/a01.npy"
    """
    frmt = """
    Magic {}
    Shape {},  C-contig {}, dtype {}
    """
    from numpy.lib import format as format_
    with open(fp, 'rb') as f:
        major, minor = format_.read_magic(f)
        mag = format_.magic(major, minor)
        shp, is_fortran, dt = format_.read_array_header_1_0(f)
        count = np.multiply.reduce(shp, dtype=np.int64)
        #data = f.readlines()

        BUFFER_SIZE = 2**18
        max_read_count = BUFFER_SIZE // min(BUFFER_SIZE, dt.itemsize)
        array = np.ndarray(count, dtype=dt)
        for i in range(0, count, max_read_count):
            read_count = min(max_read_count, count - i)
            read_size = int(read_count * dt.itemsize)
            data = format_._read_bytes(f, read_size, "array data")
            array[i:i+read_count] = np.frombuffer(data, dtype=dt,
                                                  count=read_count)
        array.shape = shp
    if prn:
        print(dedent(frmt).format(mag, shp, (not is_fortran), dt))
    return array


# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """

# not used
def arr_tif_PIL(a, fname):
    """convert an array to a tif using PIL
    :  img = Image.fromarray(a, mode='F') works for only 2D
    """
    from PIL import Image
    imgs = []
    for i in a:
        imgs.append(Image.fromarray(i))
    imgs[0].save(fname, compression="tiff_deflate", save_all=True,
                 append_images=imgs[1:])
    # ---- done


#    print("Script... {}".format(script))
#    fname = "c:/temp/a01.txt"
    a = np.arange(100).reshape(10,10)
#    _demo()
#    save_as_txt(fname, a)
#    import pickle
#
#    file = "c:/temp/a01.npy"
#    fid = open(file, "rb")
##    pk = pickle.load(fid)
#    fid.close()
#    ps = pickle.dumps(a)
#    back = pickle.loads(ps)

"""
This works

file = "c:/temp/a01.npy"
from numpy.lib import format
with open(file, 'rb') as f:
    a = format.read_array(f)

from numpy.lib import format
with open(file, 'rb') as f:
    major, minor = format.read_magic(f)
    mag = format.magic(major, minor)
    shp, is_fortran, dt = format.read_array_header_1_0(f)
    count = np.multiply.reduce(shp, dtype=np.int64)
    data = f.readlines()

    BUFFER_SIZE = 2**18
    max_read_count = BUFFER_SIZE // min(BUFFER_SIZE, dt.itemsize)
    for i in range(0, count, max_read_count):
        read_count = min(max_read_count, count - i)
        read_size = int(read_count * dtype.itemsize)
        data = _read_bytes(fp, read_size, "array data")
        array[i:i+read_count] = np.frombuffer(data, dtype=dtype, count=read_count)

for i in data:
    print("\n{}\n".format(i))


pickle stuff
-------------
np.load(fp, mmap_mode=None, allow_pickle=True, fix_imports=True,
        encoding='ASCII'):

own_fid = False
if isinstance(file, basestring):
    fid = open(file, "rb")
    own_fid = True
elif is_pathlib_path(file):
    fid = file.open("rb")
    own_fid = True
else:
    fid = file
...
if it is an npy file, then the prefix will equal the magicprefix

elif magic == format.MAGIC_PREFIX:  then
format.read_array(fid, allow_pickle=allow_pickle,
                  pickle_kwargs=pickle_kwargs)

else... it isn't an nparray and you will have to read a pickle.

try: return pickle.load(fid, **pickle_kwargs)


encoding='ASCII'
encoding = 'ASCII' # 'latin1' or 'bytes'
pickle_kwargs = dict(encoding=encoding, fix_imports=fix_imports)

"""

"""
from https://github.com/numpy/numpy/blob/master/numpy/lib/format.py

MAGIC_PREFIX = b'\x93NUMPY'

MAGIC_LEN = len(MAGIC_PREFIX) + 2

ARRAY_ALIGN = 64 # plausible values are powers of 2 between 16 and 4096

BUFFER_SIZE = 2**18  # size of buffer for reading npz files in bytes

import struct
hlength_str = _read_bytes(open(fp, 'rb'), struct.calcsize(hlength_type), "array header length")
# b'\x93NUM'

header_length = struct.unpack(hlength_type, hlength_str)[0]
# 1297436307

"""
