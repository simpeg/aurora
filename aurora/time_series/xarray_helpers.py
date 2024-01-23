"""
Placeholder module for methods manipulating xarray time series
"""

import xarray as xr
from loguru import logger


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
