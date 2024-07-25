"""
    This module contains a method for defining error bar plotting scheme.
    The function was adapted from matlab EMTF.
"""
import numpy as np
from typing import Optional


def err_log(
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    x_axis_limits: list,
    log_x_axis: Optional[bool] = True,
    barsize: float = 0.0075,
):
    """
        Returns the coordinates for the line segments that make up the error bars.

    Development Notes:
     This function returns 6 numbers per data point.
     There is no documentation for what it does.
     A reasonable guess would be that the six numbers define 3 line segments.
     One line segment for the error bar, and one line segment at the top of the error bar, and one at the bottom.
     The vectors xb and yb each have six elements per data point assigned as follows
     xb = [x-dx, x+dx, x, x, x-dx, x+dx,]
     yb = [y-dy, y-dy, y-dy, y+dy, y+dy, y+dy,]
     and if log_x_axis is True
     [log(x)-dx, log(x)+dx, log(x), log(x), log(x)-dx, log(x)+dx,]

    Matlab Documentation
    err_log : used for plotting error bars with a y-axis log scale
    takes VECTORS x and y and outputs matrices (one row per data point) for
    plotting error bars ll = 'XLOG' for log X axis

    Parameters
    ----------
    x : np.ndarray
        The x-axis values.  Usually these are periods with units of seconds
    y : np.ndarray
        The x-axis values.  Usually apparent resistivity or phase
    yerr: np.ndarray
        A value associated with the error in the y measurement.
        It seems that this is the "half height" of the error bar.
    log_x_axis : bool
        If True the xaxis is logarithmic
        Not tested for False
    x_axis_limits: list
        The lower and upper limits for the xaxis in position 0, 1 respectively.
    barsize: float
        The width of the top and bottom horizontal error bar lines.

    Returns
    -------
    xb, yb: tuple
        Each is np.ndarray, 6 rows and one column per data point
        These are the six points needed to draw the error bars.
    """
    num_observations = len(x)
    xb = np.zeros((6, num_observations))
    yb = np.zeros((6, num_observations))
    if log_x_axis:
        dx = (
            np.log(x_axis_limits[1] / x_axis_limits[0]) * barsize
        )  # natural log in matlab & python
        xb[2, :] = np.log(x)
    else:
        dx = (x_axis_limits[1] - x_axis_limits[0]) * barsize
        xb[2, :] = x
    xb[3, :] = xb[2, :]
    xb[0, :] = xb[2, :] - dx
    xb[1, :] = xb[2, :] + dx
    xb[4, :] = xb[2, :] - dx
    xb[5, :] = xb[2, :] + dx

    if log_x_axis:
        xb = np.exp(xb)

    yb[0, :] = (y - yerr).T
    yb[1, :] = (y - yerr).T
    yb[2, :] = (y - yerr).T
    yb[3, :] = (y + yerr).T
    yb[4, :] = (y + yerr).T
    yb[5, :] = (y + yerr).T

    return xb, yb
