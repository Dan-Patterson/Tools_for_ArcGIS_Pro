import re
def natsort(lst):
    """natural sort"""
    import re
    convert = lambda text: int(text) if text.isdigit() else text
    a_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(lst, key=a_key)
if __name__ == '__main__':
    a = ['r1', 'r1', 'r1', 'r4', 'r4', 'r7', 'r7', 'r7', 'r10', 'r10']
    b = sorted(a)
    print("input - \n{}".format(a))
    print("text sort - \n{}".format(b))
    vals = natsort(a)
    print("natural sort - \n{}".format(vals))