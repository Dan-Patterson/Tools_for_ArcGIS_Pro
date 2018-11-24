# -*- coding: UTF-8 -*-
"""
:Script:   coordinate_defs.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-06-17
:Purpose:  convert string formatted coordinates in some variant of degrees
:
:---------------------------------------------------------------------:
"""
def dd_mm_ss(dd, cal_long=True, use_sign=False, use_quad=False):
    """decimal degrees to deg dec min"""
    deg_sign = u'\N{DEGREE SIGN}'
    deg = int(dd)
    if deg < 0:
        quad = ['S', 'W'][cal_long]
        deg = abs(deg)
    else:
        quad = ['N', 'E'][cal_long]
    if not use_quad:
        quad = ""
    if not use_sign:
        deg_sign = ""
    mins, secs = divmod(dd*3600, 60)
    degs, mins = divmod(mins, 60)
    frmt = "{}{}-{:0.0f}-{:05.2f}{}".format(deg, deg_sign, mins, secs, quad)
    return frmt


def dd_dmm(dd, cal_long=True):
    """decimal degrees to deg dec min"""
    deg_sign = u'\N{DEGREE SIGN}'
    deg = int(dd)
    if deg < 0:
        quad = ['S', 'W'][cal_long]
        deg = abs(deg)
    else:
        quad = ['N', 'E'][cal_long]
    minsec = divmod((deg - dd)*60, 60)[-1]
    frmt = "{}{} {:0.2f}' {}".format(deg, deg_sign, minsec, quad)
    return frmt


def ddm_ddd(a, sep=" "):
    """ convert degree, decimal minute string to decimal degrees
    : a - degree, decimal minute string
    : sep - usually a space, but check
    : Useage - ddm_ddd(!SourceField!, sep=" ")
    :    python parser, sourcefield is the input string field, destination
    :    field is type double
    """
    d, m = [float(i) for i in a.split(sep)]
    sign = [-1, 1][d > 0]
    dd = sign*(abs(d) + m/60.)
    return dd


def dms_ddd(a, sep=" "):
    """ convert degree, minute, decimal second string to decimal degrees
    : a - degree, minute, decimal second string
    : sep - usually a space, but check
    : Useage - dms_ddd(!SourceField!, sep=" ")
    :    python parser, sourcefield is the input string field, destination
    :    field is type double
    """
    d, m, s = [float(i) for i in a.split(sep)]
    sign = [-1, 1][d > 0]
    dd = sign*(abs(d) + (m + s/60.)/60.)
    return dd
# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Samples
    """
    a = '45 30.30'
    b = '-75 45.45'
    c = '45 30 30.30'
    d = '-75 45 45.45'
    print('Input... {:>12s} to... {:> 12.8f}'.format(a, ddm_ddd(a, sep=" ")))
    print('Input... {:>12s} to... {:> 12.8f}'.format(b, ddm_ddd(b, sep=" ")))
    print('Input... {:>12s} to... {:> 12.8f}'.format(c, dms_ddd(c, sep=" ")))
    print('Input... {:>12s} to... {:> 12.8f}'.format(d, dms_ddd(d, sep=" ")))
