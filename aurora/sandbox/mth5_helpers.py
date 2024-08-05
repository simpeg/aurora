"""
This module contains utility functions for working with MTH5 files.
TODO: Review and move to mth5 if relevant

Background: This module was inside of mth5/clients/helper_functions.py on branch issue_76_make_mth5_factoring

Some of these functions are handy, and should eventually be merged into mth5. I would also like to
use some of these functions from time-to-time, so I am putting them here for now, until we can
decide what to move to mth5 and what to keep in aurora (and what to throw out).
"""
import datetime
import pandas as pd

from mt_metadata.timeseries.stationxml import XMLInventoryMTExperiment
from mth5.clients import FDSN
from mth5.utils.helpers import initialize_mth5
from loguru import logger


def enrich_channel_summary(mth5_object, df, keyword):
    """
    Operates on a channel summary df and adds some information in a new column.

    Parameters
    ----------
    mth5_object: mth5.mth5.MTH5
    df: pd.DataFrame
        A channel summary dataframe
    keyword: str
        supported keywords are ["num_filters",]
        "num_filters" computes the number of filters associated with each row (channel-run) and adds that "num_filters" column of df

    Returns
    -------
    df: pd.DataFrame
        The channel summary df with the new column
    """
    df[keyword] = -1
    if keyword == "num_filters":
        for i_row, row in df.iterrows():
            channel = mth5_object.get_channel(
                row.station, row.run, row.component, row.survey
            )
            num_filters = len(channel.channel_response.filters_list)
            df[keyword].iat[i_row] = num_filters
    elif keyword == "filter_units_in":
        for i_row, row in df.iterrows():
            channel = mth5_object.get_channel(
                row.station, row.run, row.component, row.survey
            )
            units_in = [x.units_in for x in channel.channel_response.filters_list]
            df[keyword].iat[i_row] = units_in
    elif keyword == "filter_units_out":
        for i_row, row in df.iterrows():
            channel = mth5_object.get_channel(
                row.station, row.run, row.component, row.survey
            )
            units_out = [x.units_out for x in channel.channel_response.filters_list]
            df[keyword].iat[i_row] = units_out
    return df


def augmented_channel_summary(mth5_object, df=None):
    """
    Adds column "n_filters" to mth5 channel summary.

    This function was used when debugging and wide scale testing at IRIS/Earthscope.

    Development Notes:
    TODO: Consider supporting a list of keyeords that tell what columns to add

    Parameters
    ----------
    df: channel summary dataframe


    Returns
    -------
    df: pd.Dataframe
        Same as input but with new column
    """
    if not df:
        df = mth5_object.channel_summary.to_dataframe()
    df["n_filters"] = -1
    for i_row, row in df.iterrows():
        channel = mth5_object.get_channel(
            row.station, row.run, row.component, row.survey
        )
        n_filters = len(channel.channel_response.filters_list)
        df.n_filters.iat[i_row] = n_filters
    return df


def build_request_df(
    network_id,
    station_id,
    channels=None,
    start=None,
    end=None,
    time_period_dict={},
    mth5_version="0.2.0",
) -> pd.DataFrame:
    """
    Given some information about an earthscope dataset, format the dataset description
     into a request_df dataframe.

    Parameters
    ----------
    network_id: string
        Two-character network identifier string fro FDSN.
    station_id: string
        Short identifier code used by FDSN, e.g. CAS04, NVR11
    channels: list or None
        3-character channel identifiers, e.g. ["LQ2", "MFZ"],
        support for wildcards of the form ["*F*", "*Q*",] is experimental
        Does not support wildcards of the form ["*",]
    start: string
        ISO-8601 representation of a timestamp
    end: string
        ISO-8601 representation of a timestamp
    time_period_dict: dict
        Keyed by same values as channels, this gives explicit start/end times for each of the channels
    mth5_version: str
        From ["0.1.0", "0.2.0"]

    Returns
    -------
    request_df: pd.DataFrame
        A formatted dataframe that can be passed to mth5.clients.FDSN to request metdata or data.

    """

    def get_time_period_bounds(ch):
        if ch in time_period_dict.keys():
            # time_interval = time_period_dict[ch]
            ch_start = time_period_dict[ch].left.isoformat()
            ch_end = time_period_dict[ch].right.isoformat()
        else:
            if start is None:
                ch_start = "1970-01-01 00:00:00"
            else:
                ch_start = start
            if end is None:
                ch_end = datetime.datetime.now()
                ch_end = ch_end.replace(hour=0, minute=0, second=0, microsecond=0)
                ch_end = str(ch_end)
            else:
                ch_end = end
        return ch_start, ch_end

    fdsn_object = FDSN(mth5_version=mth5_version)
    fdsn_object.client = "IRIS"

    request_list = []
    for channel in channels:
        ch_start, ch_end = get_time_period_bounds(channel)
        request_list.append([network_id, station_id, "", channel, ch_start, ch_end])

    logger.info(f"request_list: {request_list}")

    request_df = pd.DataFrame(request_list, columns=fdsn_object.request_columns)
    # workaround for having a channel with missing run
    # request_df["start"] = request_df["start"].max()
    return request_df


def get_experiment_from_obspy_inventory(inventory):
    """Converts an FDSN inventory to an MTH5 Experiment object"""
    translator = XMLInventoryMTExperiment()
    experiment = translator.xml_to_mt(inventory_object=inventory)
    return experiment


def mth5_from_experiment(experiment, h5_path=None):
    """
    Converts an experiment object into an mth5 file.

    Parameters
    ----------
    experiment
    h5_path

    Returns
    -------

    """
    mth5_obj = initialize_mth5(h5_path)
    mth5_obj.from_experiment(experiment)
    return mth5_obj


def get_channel_summary(h5_path):
    """
    Gets a channel summary from an mth5;
    TODO: This can be replaced by methods in mth5.

    Parameters
    ----------
    h5_path: pathlib.Path
        Where is the h5

    Returns
    -------
    channel_summary_df: pd.DataFrame
        channel summary from mth5
    """
    mth5_obj = initialize_mth5(
        h5_path=h5_path,
    )
    mth5_obj.channel_summary.summarize()
    channel_summary_df = mth5_obj.channel_summary.to_dataframe()
    mth5_obj.close_mth5()
    logger.info(channel_summary_df)
    return channel_summary_df
