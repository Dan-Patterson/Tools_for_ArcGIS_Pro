# -*- coding: UTF-8 -*-
"""
:Script:   string_defs.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-06-17
:Purpose:  tools for working strings
:Useage:
:  These are mini-onliners or so

:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----

a = 'A string with numbers 10   20 in it'

keep_text = "".join([i for i in a if i.isalpha() or i == " "]).strip()

strip_spaces = " ".join([i.strip() for i in a.split(" ") if i != ""])

keep_numb = "".join([i for i in a if i.isdigit() or i == " "]).strip()

num_csv = ", ".join([i for i in a.split() if i.isdigit() ]).strip()

frmt = """
Input string......... {}

Just text ........... {}
Strip extra spaces .. {}
Just numbers ........ {}
Numbers to csv ...... {}

"""
args = [a, keep_text, strip_spaces, keep_numb, num_csv]
print(frmt.format(*args))
