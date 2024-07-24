"""
    This module contains a method for defining error bar plotting scheme.
    The function was adapted from matlab EMTF.
"""
import numpy as np


def err_log(x: np.ndarray, y: np.ndarray, yerr: np.ndarray, ll: str, lims: list):
    """
    err_log : used for plotting error bars with a y-axis log scale
    takes VECTORS x and y and outputs matrices (one row per data point) for
    plotting error bars ll = 'XLOG' for log X axis

    Development Notes:
     This function returns 6 numbers per data point.
     There is no documentation for what it does.
     A reasonable guess would be that the six numbers define 3 line segments.
     One line segment for the error bar, and one line segment at the top of the error bar, and one at the bottom.

    Parameters
    ----------
    x : numpy vector
    y : numpy vector
    yerr
    ll : str
        'XLOG' for log X axis
    lims

    Returns
    -------
    [xb,yb]: matrices, one row per data point
    """
    num_observations = len(x)
    xb = np.zeros((6, num_observations))
    yb = np.zeros((6, num_observations))
    barsize = 0.0075
    if ll.lower() == "xlog":
        dx = np.log(lims[1] / lims[0]) * barsize  # natural log in matlab & python
        xb[2, :] = np.log(x)
    else:
        dx = (lims[1] - lims[0]) * barsize
        xb[2, :] = x
    xb[3, :] = xb[2, :]
    xb[0, :] = xb[2, :] - dx
    xb[1, :] = xb[2, :] + dx
    xb[4, :] = xb[2, :] - dx
    xb[5, :] = xb[2, :] + dx

    if ll.lower() == "xlog":
        xb = np.exp(xb)

    yb[0, :] = (y - yerr).T
    yb[1, :] = (y - yerr).T
    yb[2, :] = (y - yerr).T
    yb[3, :] = (y + yerr).T
    yb[4, :] = (y + yerr).T
    yb[5, :] = (y + yerr).T

    return xb, yb
