"""
Placeholder module for methods manipulating xarray time series
"""

import numpy as np
import xarray as xr
from loguru import logger
from typing import Optional


def handle_nan(
    X: xr.Dataset,
    Y: Optional[xr.Dataset],
    RR: Optional[xr.Dataset],
    drop_dim: Optional[str] = "",
) -> tuple:
    """
    Drops Nan from multiple channel series.

    Initial use case is for Fourier coefficients, but could be more general.

    Idea is to merge X,Y,RR together, and then call dropna.  We have to be careful
    with merging because there can be namespace clashes in the channel labels.
    Currently handling this by relabelling the remote reference channels from for
    example "hx"--> "remote_hx", "hy"-->"remote_hy".  If needed we could add "local" to
    local the other channels in X, Y.

    It would be nice to maintain an index of what was dropped.

    Development Note: Even if there are no NaNs in the data, this function can still encounter issues,
    in particular if the time coordinates of X and RR do not match exactly.
    In this case, the merge will fail, and we will log an error message.

    TODO: This should be handled more gracefully, perhaps by aligning the time coordinates
    before merging, or by using a different merge strategy that allows for slight mismatches.
    Alternatively, a validator could be added to ensure that the time coordinates of X and RR match before calling this function.
    That validation could be done by tfk, or at least immediately following the construction / access of the FCs.

    TODO: Merge this with drop_nan in transfer_function_helpers.py

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
    time_axis_problem = False
    if not time_axis_match(X, Y, RR):
        time_axis_problem = True
        logger.error("Time axes do not match, merging datasets with left join")
        logger.error("This may lead to unexpected results, please check your data.")

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
    if time_axis_problem:
        # If the time axes do not match, we cannot merge them directly.
        # Instead, we will merge RR with a left join, which will keep all timestamps from
        # X and Y, and fill in the missing values from RR where available.
        merged_xr = merged_xr.merge(RR, join="left")
        for ch in list(RR.keys()):
            merged_xr[ch].values = RR[ch].values
    else:
        # If the time axes match, we can merge them directly.
        merged_xr = merged_xr.merge(RR, join="exact")
        logger.debug("Time axes match, merging datasets")

    # Drop NaN values across the specified dimension
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


def time_axis_match(
    X: xr.Dataset,
    Y: Optional[xr.Dataset] = None,
    RR: Optional[xr.Dataset] = None,
) -> bool:
    """
    Check if the time axes of X, Y, and RR match.

    TODO: consider raising an exception if they do not match, rather than returning False.

    Parameters
    ----------
    X : xr.Dataset
        Dataset containing time series data.
    Y : xr.Dataset
        Dataset containing time series data.
    RR : xr.Dataset or None
        Optional dataset for remote reference channels.

    Returns
    -------
    bool
        True if all time axes match, False otherwise.
    """
    if RR is None:
        ok = X.time.equals(Y.time)
    elif Y is None:
        ok = X.time.equals(RR.time)
    else:
        ok = X.time.equals(Y.time) and X.time.equals(RR.time)
    if not ok:
        logger.error("Coordinate alignment mismatch -- see aurora issue #228 ")
        matches = X.time.values == RR.time.values
        logger.error(f"{matches.sum()}/{len(matches)} timestamps match exactly")
        deltas = X.time.values - RR.time.values
        logger.error(f"Maximum offset is {deltas.__abs__().max()}ns")

    return ok


def nan_to_mean(xrds: xr.Dataset) -> xr.Dataset:
    """
    Set Nan values in xr.Dataset to the mean value per channel.

    xrds: xr.Dataset
        Time series data

    Returns
    -------
    run_xrds : xr.Dataset
        The same as the input time series but NaN values are replaced by the mean of the time series (per channel).

    """
    for ch in xrds.keys():
        null_values_present = xrds[ch].isnull().any()
        if null_values_present:
            msg = "Unexpected Null values in xrds -- this should be examined"
            logger.warning(msg)
            value = np.nan_to_num(np.nanmean(xrds[ch].data))
            xrds[ch] = xrds[ch].fillna(value)
    return xrds
