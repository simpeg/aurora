import numpy as np


def err_log(x, y, yerr, ll, lims):
    """
    err_log : used for plotting error bars with a y-axis log scale
    takes VECTORS x and y and outputs matrices (one row per data point) for
    plotting error bars ll = 'XLOG' for log X axis

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
