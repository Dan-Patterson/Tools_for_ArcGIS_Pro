# -*- coding: UTF-8 -*-
"""
:Script:   compass_angles.py
:
:Author:   Dan.Patterson@carleton.ca
:
:Modified: 2016-09-25
:
:Purpose:  Convert angles, in degrees, to compass designations.
:
:Functions:  help(<function name>) for help
:  compass - the conversion function
:  _demo  -  demo function ...
:
:Notes:
:  Cardinal direction and degree ranges.........
:  N 348.75- 11.25  NNE  11.25- 33.75  NE  33.75- 56.2   ENE  56.25- 78.75
:  E  78.75-101.25  ESE 101.25-123.75  SE 123.75-146.25  SSE 146.25-168.75
:  S 168.75-191.25  SSW 191.25-213.75  SW 213.75-236.25  WSW 236.25-258.75
:  W 258.75-281.25  WNW 281.25-303.75  NW 303.75-326.25  NNW 326.25-348.75
:
:  np.arange(11.25, 360., 22.5)  #  Generator which yields...
:  array([ 11.250,  33.750,  56.250 ... 303.750, 326.250,  348.750])
:
:References
:
"""
# ---- imports, formats, constants ----

import sys
import numpy as np
from textwrap import dedent

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2,
                    suppress=True, threshold=100, formatter=ft)

script = sys.argv[0]

# ---- functions ----


def compass(angle):
    """Return the compass direction based on supplied angle.

    Requires:
    --------
    `angle` : number
        Angle(s) in degrees, no check made for other formats.  a single value,
        list or np.ndarray can be used as input.

        Angles are assumed to be from compass north, alter to suit.

    Returns:
    -------
        The compass direction.
    Notes:
    -----
        Compass ranges can be altered to suit the desired format.
        See various wiki's for other options.

        This incarnation uses 22.5 degree ranges with the compass centered
        on the range.

        ie. N  between 348.75 and 11.25 degrees, range equals 22.5)

    """
    c = np.array(['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                  'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 'N'])
    a = np.arange(11.25, 360., 22.5)
    if isinstance(angle, (float, int, list, np.ndarray)):
        angle = np.atleast_1d(angle)
    comp_dir = c[np.digitize(angle, a)]
    if len(comp_dir) == 1:
        comp_dir[0]
    return comp_dir


def run_demo():
    """A sample run of compass returning compass notations for 20 deg
    increments.  Change to suit
    """
    angles = np.arange(0, 360, 20)
    rose = compass(angles)
    dt = [("Angle", "<f8"), ("Code", "U5")]
    comp_rose = np.asarray(list(zip(angles, rose)), dtype=dt)
    comp_rose['Angle'] = angles
    comp_rose['Code'] = rose
    print("\nCompass rose examples\n{}".format(comp_rose))
    return comp_rose

# n = np.arange(-10, 10.)
# v = np.where(n >= 0, np.PZERO, np.NZERO)

# a = "Hello there my friend"
# b = "".join([[i, "\n"][i == " "] for i in a])
# print(b)


# ----------------------
if __name__ == "__main__":
    """Main section...   """
    # print("Script... {}".format(script))
    comp_rose = run_demo()
