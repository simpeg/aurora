import numpy as np
import xarray as xr


def covariance_xr(X):
    """
    Compute the covariance matrix.  Uses

    Parameters
    ----------
    X: xarray.core.dataarray.DataArray
        Multivariate time series as an xarray

    Returns
    -------
    S: xarray.core.dataarray.DataArray
        The covariance matrix of the data in xarray form.
    """
    channels = list(X.coords["variable"].values)

    S = xr.DataArray(
        np.cov(X),
        dims=["channel_1", "channel_2"],
        coords={"channel_1": channels, "channel_2": channels},
    )
    return S
