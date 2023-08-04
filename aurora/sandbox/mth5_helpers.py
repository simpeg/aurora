"""
This module was inside of mth5/clients/helper_functions.py
on branch issue_76_make_mth5_factoring

Some of these functions are handy, and should eventually be merged into mth5.

I would also like to use some of these functions from time-to-time, so I am putting
them here for now, until we can decide what to move to mth5 and what to keep in
aurora (and what to throw out).
"""
import datetime
import pandas as pd


from mt_metadata.timeseries.stationxml import XMLInventoryMTExperiment
from mth5.clients import FDSN
from mth5.utils.helpers import initialize_mth5

def enrich_row_of_channel_summary(row, keyword):
    """

    Parameters
    ----------
    row
    keyword

    Returns
    -------

    """
    if keyword=="num_filters":
        pass

def augmented_channel_summary(mth5_object, df=None):#, **kwargs):
    """
    Consider supportig kwargs, such as a list of keyords that tell what columns to add
    For now, we only want to add n_filters
    Parameters
    ----------
    df: channel summary dataframe


    Returns
    -------

    """
    if not df:
        df = mth5_object.channel_summary.to_dataframe()
    df["n_filters"] = -1
    for i_row, row in df.iterrows():
        channel = mth5_object.get_channel(row.station, row.run, row.component, row.survey)
        n_filters = len(channel.channel_response_filter.filters_list)
        df.n_filters.iat[i_row] = n_filters
    return df


def build_request_df(network_id, station_id, channels=None, start=None, end=None):
    """

    Args:
        network_id: string
            Two-character network identifier string fro FDSN.
        station_id: string
            Short identifier code used by FDSN, e.g. CAS04, NVR11
        channels: list or None
            3-character channel identifiers, e.g. ["LQ2", "MFZ"], also supports wildcards of the form ["*F*", "*Q*",]
             Does not support wildcards of the form ["*",]
        start: string
            ISO-8601 representation of a timestamp
        end: string
            ISO-8601 representation of a timestamp

    Returns:
        request_df: pd.DataFrame
        A formatted dataframe that can be passed to mth5.clients.FDSN to request metdata or data.

    """
    from mth5.clients import FDSN
    fdsn_object = FDSN(mth5_version='0.2.0')
    fdsn_object.client = "IRIS"
    if start is None:
        start = '1970-01-01 00:00:00'
    if end is None:
        end = datetime.datetime.now()
        end = end.replace(hour=0, minute=0, second=0, microsecond=0)
        end = str(end)

    request_list = []
    for channel in channels:
        request_list.append([network_id, station_id, '', channel, start, end])

    print(f"request_list: {request_list}")

    request_df = pd.DataFrame(request_list, columns=fdsn_object.request_columns)
    return request_df


def get_experiment_from_obspy_inventory(inventory):
    translator = XMLInventoryMTExperiment()
    experiment = translator.xml_to_mt(inventory_object=inventory)
    return experiment


def mth5_from_experiment(experiment, h5_path=None):
    """

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
    print(channel_summary_df)
    return channel_summary_df
