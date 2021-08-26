"""
Placeholder module for methods manipulating xarray time series
"""

import xarray as xr


def handle_nan(X, Y, RR, config, drop_dim=""):
    """
    !!! PROBABLY TO BE DEPRECATED
    !!! THis method works with 3D STFT arrays, but as of 26AUg2021 we are using
    xarray.stack() to cast FCs are 2D arrays merging time/frequency axes.

    Drops Nan from multiple channel series'.
    Initial use case is for Fourier coefficients, but could be more general.

    Idea is to merge X,Y,RR together, and then call dropna

    Parameters
    ----------
    X : xr.Dataset
    Y : xr.Dataset
    RR : xr.Dataset or None
    config : ProcessingConfig
    drop_dim: string
        specifies the dimension on which dropna is happening.  For 3D STFT arrays
        this is "time", for 2D stacked STFT this is "observation"
    Returns
    -------
    X : xr.Dataset
    Y : xr.Dataset
    RR : xr.Dataset or None

    """
    data_var_add_label_mapper = {}
    data_var_rm_label_mapper = {}
    for ch in config.reference_channels:
        data_var_add_label_mapper[ch] = f"remote_{ch}"
        data_var_rm_label_mapper[f"remote_{ch}"] = ch
    # if needed we could add local to local channels as well, or station label
    merged_xr = X.merge(Y, join="exact")
    if RR is not None:
        RR = RR.rename(data_var_add_label_mapper)
        merged_xr = merged_xr.merge(RR, join="exact")

    merged_xr = merged_xr.dropna(dim=drop_dim)
    merged_xr = merged_xr.to_array(dim="channel")
    X = merged_xr.sel(channel=config.input_channels)
    X = X.to_dataset(dim="channel")
    output_channels = list(Y.data_vars)
    Y = merged_xr.sel(channel=output_channels)
    Y = Y.to_dataset(dim="channel")
    if RR is not None:
        remote_channels = list(data_var_rm_label_mapper.keys())
        RR = merged_xr.sel(channel=remote_channels)
        RR = RR.to_dataset(dim="channel")
        RR = RR.rename(data_var_rm_label_mapper)

    return X, Y, RR


def cast_3d_stft_to_2d_observations(XY):
    """
    When the data for a frequency band are extracted from the STFT and
    passed to RegressionEstimator they have a typical STFT structure:
    One axis is time (the time of the window that was FFT-ed) and the
    other axis is frequency.  However we make no distinction between the
    harmonics (or bins) within a band.  We need to gather all the FCs for
        each channel into a 1D array.
    This method performs that reshaping (ravelling) operation.
    *It is not important how we unravel the FCs but it is important that
    we use the same scheme for X and Y.

    2021-08-25: Modified this method to use xarray's stack() method.

    Parameters
    ----------
    XY: either X or Y of the regression nomenclature.  Should be an
    xarray.Dataset already splitted on channel

    Returns
    -------
    output_array: numpy array of two dimensions (observations, channel)

    """
    if isinstance(XY, xr.Dataset):
        tmp = XY.to_array("channel")

    tmp = tmp.stack(observation=("frequency", "time"))
    return tmp
    # output_array = tmp.data.T
    # return output_array
