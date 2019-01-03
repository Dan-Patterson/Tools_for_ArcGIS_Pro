import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines

def scatter(pts,show=True, label=False, connect=False, pairs=None,pos=1,row=1, col=1, new=False):
    if(connect==True and pairs is None):
        print("ERROR: There must be a valid input for the 'pairs' parameter to connect the points.")
        return
    if new==True:
        plt.figure().add_subplot(row, col, pos)
    else:
        plt.subplot(row, col, pos)

#    if show==True:
#        plt.scatter(pts[:, 0], pts[:, 1])

    if label==True:
        lbl = np.arange(len(pts))
        for label, xpt, ypt in zip(lbl, pts[:, 0], pts[:, 1]):
            plt.annotate(label, xy=(xpt, ypt), xytext=(2, 2), size=8, textcoords='offset points', ha='left', va='bottom')

    if connect==True:
        for pair in pairs:
            i, j = pair
            plt.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], c='r')


def subplts(plots=1, by_col=True, max_rc=4):
    """specify the num(ber) of subplots desired and return the rows
    :  and columns to produce the subplots.
    :  by_col - True for column oriented, False for row
    :  max_rc - maximum number of rows or columns depending on by_col
    """
    row_col = (1, 1)
    if by_col:
        if plots <= max_rc:
            row_col = (1, plots)
        else:
            row_col = (plots - max_rc, max_rc)
    else:
        if plots <= max_rc:
            row_col = (plots, 1)
        else:
            row_col = (max_rc, plots - max_rc)
    return row_col


def generate():
    plt.show()
    #plt.close()
