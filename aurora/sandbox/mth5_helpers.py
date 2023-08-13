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

def repair_missing_filters(mth5_path, mth5_version):
    m = initialize_mth5(mth5_path, file_version=mth5_version)
    channel_summary_df = m.channel_summary.to_dataframe()
    # if len(channel_summary_df) == 0:
    #     print("whoops, no channel summary")
    #     m.channel_summary.summarize()
    #     channel_summary_df = m.channel_summary.to_dataframe()
    channel_summary_df = enrich_channel_summary(m, channel_summary_df, "num_filters")
    sssr_grouper = channel_summary_df.groupby(["survey", "station", "sample_rate"])
    for (survey, station, sample_rate), sub_df in sssr_grouper:
        runs_and_starts = sub_df.groupby(["run", "start"]).size().reset_index()[["run", "start"]]

        for i_row, row in sub_df.iterrows():
            if row.num_filters < 1:
                print(f"Filterless channel detected in row {i_row} fo sub_df")
                print(f"survey={survey}, station={station}, sample_rate={sample_rate}")
                print("Try to fix it with filter from a previous run")
                channel = m.get_channel(row.station, row.run, row.component, row.survey)
                start_time = pd.Timestamp(row.start)
                earlier_runs = runs_and_starts[runs_and_starts.start < row.start]
                if len(earlier_runs) == 0:
                    print("No earlier runs -- so we cannot fix the missing filters")
                previous_run = earlier_runs.iloc[-1].run
                previous_channel = m.get_channel(row.station, previous_run, row.component, row.survey)
                print("WE NEED TO ASSERT METADATA ARE SAME (except filters")
                channel.metadata.filter = previous_channel.metadata.filter
                channel.write_metadata()
    m.close_mth5()


def enrich_channel_summary(mth5_object, df, keyword):
    """

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
    if keyword=="num_filters":
        for i_row, row in df.iterrows():
            channel = mth5_object.get_channel(row.station, row.run, row.component, row.survey)
            num_filters = len(channel.channel_response_filter.filters_list)
            df[keyword].iat[i_row] = num_filters
    return df

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
