"""
Placeholder module for methods manipulating xarray time series
"""

import numpy as np
import xarray as xr
from loguru import logger
from typing import Optional, Union


def handle_nan(X, Y, RR, drop_dim=""):
    """
    Drops Nan from multiple channel series'.

    Initial use case is for Fourier coefficients, but could be more general.

    Idea is to merge X,Y,RR together, and then call dropna.  We have to be careful
    with merging because there can be namespace clashes in the channel labels.
    Currently handling this by relabelling the remote reference channels from for
    example "hx"--> "remote_hx", "hy"-->"remote_hy".  If needed we could add "local" to
    local the other channels in X, Y.

    It would be nice to maintain an index of what was dropped.

    TODO: We can probably eliminate the config argument by replacing
    config.reference_channels with list(R.data_vars) and setting a variable
    input_channels to X.data_vars.  In general, this method could be robustified by
    renaming all the data_vars with a prefix, not just the reference channels

    Parameters
    ----------
    X : xr.Dataset
    Y : xr.Dataset or None
    RR : xr.Dataset or None
    drop_dim: string
        specifies the dimension on which dropna is happening.  For 3D STFT arrays
        this is "time", for 2D stacked STFT this is "observation"
    Returns
    -------
    X : xr.Dataset
    Y : xr.Dataset
    RR : xr.Dataset or None

    """
    if Y is None:
        Y = xr.Dataset()
    if RR is None:
        RR = xr.Dataset()

    input_channels = list(X.data_vars)
    output_channels = list(Y.data_vars)
    reference_channels = list(RR.data_vars)
    data_var_add_label_mapper = {}
    data_var_rm_label_mapper = {}
    for ch in reference_channels:
        data_var_add_label_mapper[ch] = f"remote_{ch}"
        data_var_rm_label_mapper[f"remote_{ch}"] = ch
    RR = RR.rename(data_var_add_label_mapper)

    merged_xr = X.merge(Y, join="exact")
    # Workaround for issue #228
    # merged_xr = merged_xr.merge(RR, join="exact")
    try:
        merged_xr = merged_xr.merge(RR, join="exact")
    except ValueError:
        logger.error("Coordinate alignment mismatch -- see aurora issue #228 ")
        matches = X.time.values == RR.time.values
        logger.error(f"{matches.sum()}/{len(matches)} timestamps match exactly")
        deltas = X.time.values - RR.time.values
        logger.error(f"Maximum offset is {deltas.__abs__().max()}ns")
        #        print(f"X.time.[0]: {X.time[0].values}")
        #        print(f"RR.time.[0]: {RR.time[0].values}")
        merged_xr = merged_xr.merge(RR, join="left")
        for ch in list(RR.keys()):
            merged_xr[ch].values = RR[ch].values

    merged_xr = merged_xr.dropna(dim=drop_dim)
    merged_xr = merged_xr.to_array(dim="channel")
    X = merged_xr.sel(channel=input_channels)
    X = X.to_dataset(dim="channel")
    Y = merged_xr.sel(channel=output_channels)
    Y = Y.to_dataset(dim="channel")

    remote_channels = list(data_var_rm_label_mapper.keys())
    RR = merged_xr.sel(channel=remote_channels)
    RR = RR.to_dataset(dim="channel")
    RR = RR.rename(data_var_rm_label_mapper)

    return X, Y, RR


def covariance_xr(
    X: xr.DataArray, aweights: Optional[Union[np.ndarray, None]] = None
) -> xr.DataArray:
    """
    Compute the covariance matrix with numpy.cov.

    Parameters
    ----------
    X: xarray.core.dataarray.DataArray
        Multivariate time series as an xarray
    aweights: array_like, optional
        Doc taken from numpy cov follows:
        1-D array of observation vector weights. These relative weights are
        typically large for observations considered "important" and smaller for
        observations considered less "important". If ``ddof=0`` the array of
        weights can be used to assign probabilities to observation vectors.

    Returns
    -------
    S: xarray.DataArray
        The covariance matrix of the data in xarray form.
    """

    channels = list(X.coords["variable"].values)

    S = xr.DataArray(
        np.cov(X, aweights=aweights),
        dims=["channel_1", "channel_2"],
        coords={"channel_1": channels, "channel_2": channels},
    )
    return S


def initialize_xrda_1d(
    channels: list,
    dtype=Optional[type],
    value: Optional[Union[complex, float, bool]] = 0,
) -> xr.DataArray:
    """

    Returns a 1D xr.DataArray with variable "channel", having values channels named by the input list.

    Parameters
    ----------
    channels: list
        The channels in the multivariate array
    dtype: type
        The datatype to initialize the array.
        Common cases are complex, float, and bool
    value: Union[complex, float, bool]
        The default value to assign the array

    Returns
    -------
    xrda: xarray.core.dataarray.DataArray
        An xarray container for the channels, initialized to zeros.
    """
    k = len(channels)
    logger.debug(f"Initializing xarray with values {value}")
    xrda = xr.DataArray(
        np.zeros(k, dtype=dtype),
        dims=[
            "variable",
        ],
        coords={
            "variable": channels,
        },
    )
    if value != 0:
        data = value * np.ones(k, dtype=dtype)
        xrda.data = data
    return xrda


def initialize_xrda_2d(
    channels, dtype=complex, value: Optional[Union[complex, float, bool]] = 0, dims=None
):

    """
     TODO: consider merging with initialize_xrda_1d
     TODO: consider changing nomenclature from dims=["channel_1", "channel_2"],
     to dims=["variable_1", "variable_2"], to be consistent with initialize_xrda_1d

    Parameters
     ----------
     channels: list
         The channels in the multivariate array
     dtype: type
         The datatype to initialize the array.
         Common cases are complex, float, and bool
     value: Union[complex, float, bool]
         The default value to assign the array

    Returns
     -------
     xrda: xarray.core.dataarray.DataArray
         An xarray container for the channel variances etc., initialized to zeros.
    """
    if dims is None:
        dims = [channels, channels]

    K = len(channels)
    logger.debug(f"Initializing 2D xarray to {value}")
    xrda = xr.DataArray(
        np.zeros((K, K), dtype=dtype),
        dims=["channel_1", "channel_2"],
        coords={"channel_1": dims[0], "channel_2": dims[1]},
    )
    if value != 0:
        data = value * np.ones(xrda.shape, dtype=dtype)
        xrda.data = data

    return xrda
