# -*- coding: UTF-8 -*-
# (1) ... field calculator code block
#     Save as a *.cal file for loading directly into the code block.
#     Uncomment the last 2 lines to have the seq_dup line split out
#
fld = ""
def seq_dup(val):
    """sequential duplicate checks"""
    global fld
    if val == fld:
        ret = 1
    else:
        ret = 0
    fld = val
    return ret
#__esri_field_calculator_splitter__  # used by *.cal files
#seq_dup(!Test!)   # copy to expression section

# (2) ... CalculateField format for use in scripts.
#     Uncomment in your IDE, insert the following lines into your script----
#
#import arcpy
#expr = '''
#fld = ""
#def seq_dup(val):
#    """sequential duplicate checks"""
#    global fld
#    if val == fld:
#        ret = 1
#    else:
#        ret = 0
#    fld = val
#    return ret
#'''
#arcpy.management.CalculateField("f1", "IndFld", "seq_dup(!Test!),
#                                 "PYTHON_9.3",
#                                 expression=expr)
# ---- End of CalculateField section

if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    """
    import inspect
    s = "".join([i for i in inspect.getsourcelines(seq_dup)[0]])
    s = "{}\n{}".format('fld = ""', s)
    print(s)
