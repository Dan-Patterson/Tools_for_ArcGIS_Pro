# -*- coding: UTF-8 -*-
"""
:Script:   natural_sort.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-08-23
:Purpose:  Returns a mixed text-number list accounting for numeric values
:  ['a1', 'a20', 'a2', 'a10'] should yield ['a1', 'a2', 'a10', 'a20']
:Note:
: C:\Git_Dan\JupyterNoteBooks\Short_Samples\Natural_sort.ipynb
"""

import re


def natsort(text_lst):
    """natural sort returns text containing numbers sorted considering the
    :  number in the sequence.
    :originals used lambda expressions
    :  convert = lambda text: int(text) if text.isdigit() else text
    :  a_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    """
    def convert(text):
        return int(text) if text.isdigit() else text

    def a_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(text_lst, key=a_key)


# --------------------------------------------------------------------------
if __name__ == '__main__':
    """run with sample"""
    a = ['a1', 'a20', 'a2', 'a10']
    vals = natsort(a)
#    print("natural sort - \n{}".format(vals))
