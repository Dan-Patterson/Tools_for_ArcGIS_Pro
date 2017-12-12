import re
def nat_pad(val, pad='0000'):
    """natural sort... put the import re outside of the function
    :if using the field calculator
    : calculator expression- nat_pad(!data_field!, pad='a bunch of 0s')
    """
    txt = re.split('([0-9]+)', val)
    l_val = len(str(val))
    txt_out = "{}{}{}".format(txt[0], pad[:-l_val], txt[1])
    return txt_out
if __name__ == '__main__':
    a = ['r1', 'r1', 'r1', 'r4', 'r4', 'r7', 'r7', 'r7', 'r10', 'r10']
    print("input - \n{}".format(a))
    vals = [nat_pad(i) for i in a]
    print("output - \n{}".format(vals))