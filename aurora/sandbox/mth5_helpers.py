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

from aurora.time_series.filters.filter_helpers import make_coefficient_filter
from mt_metadata.timeseries.stationxml import XMLInventoryMTExperiment
from mth5.clients import FDSN
from mth5.utils.helpers import initialize_mth5

ELECTRIC_SI_TO_MT = make_coefficient_filter(name="electric_si_units", gain=1e-6,
                                            units_in="mV/km", units_out="V/m")
MAGNETIC_SI_TO_MT = make_coefficient_filter(name="magnetic_si_units", gain=1e-9,
                                            units_in="nT", units_out="T")
ELECTRIC_SI_TO_MT_UNKNOWN_UNITS = make_coefficient_filter(name="electric_si_unknown_units", gain=1e-6,)
MAGNETIC_SI_TO_MT_UNKNOWN_UNITS = make_coefficient_filter(name="magnetic_si_unknown_units", gain=1e-9,)
def check_if_22111(channel_summary_df):
    """
    Picks the earliest run, and checks if it is 22111
    This should be improved to check if the output is V/m on Electruc and T on magnetic
    Parameters
    ----------
    channel_summary_df: pd.DataFrame
        This is an "enriched" channel summary, with num_filters column

    Returns
    -------

    """
    run_id = channel_summary_df.iloc[0].run # assuming channel_summary_df is sorted by time, and only 1 station
    run_sub_df = channel_summary_df[channel_summary_df.run==run_id]
    num_filters = run_sub_df.num_filters.to_numpy()
    if (num_filters==[2,2,1,1,1]).all():
        return True
    elif (num_filters == [6, 6, 3, 3, 3]).all():
        return False
    elif (num_filters == [3, 3, 2, 2, 2]).all():
        print("looks like this one already got fixex")
        return False
    else:
        print(f"UNEXPECTED num fulters {num_filters}")
        return False
    return True

def repair_missing_filters(mth5_path, mth5_version, triage_units=False):
    """

    Parameters
    ----------
    mth5_path
    mth5_version
    triage_units: Bool
        Pick the earliest run, and check if it is 22111
        Find E-fields with only 2 stages, and if so, add a 1e-6 coefficient
        Find B-fields with only 1 stages, and if so, add a 1e-9 coefficient

    Returns
    -------

    """
    def check_units_are_known(channel):
        """
        Logic can be expanded here, this is just a placeholder in a function that is alraedy a workaround:/
        Parameters
        ----------
        channel

        Returns
        -------

        """
        units_in =  [x.units_in for x in channel.channel_response_filter.filters_list]
        units_out = [x.units_out for x in channel.channel_response_filter.filters_list]
        if 'unknown' in units_in:
            return False
        else:
            return True


    m = initialize_mth5(mth5_path, file_version=mth5_version)

    channel_summary_df = m.channel_summary.to_dataframe()
    if len(channel_summary_df) == 1:
        print("whoops, no channel summary")
        m.channel_summary.summarize()
        channel_summary_df = m.channel_summary.to_dataframe()
        if len(channel_summary_df) == 1:
            print(f"There maybe no data in {mth5_path.name}")
            print(f"Filesize is only {mth5_path.stat().st_size}")
            print("SKIP IT")
            m.close_mth5()
            return
    channel_summary_df = enrich_channel_summary(m, channel_summary_df, "num_filters")

    # TRIAGE UNITS
    if triage_units:
        is_22111 = check_if_22111(channel_summary_df)
        if is_22111:
            survey_id = channel_summary_df.iloc[0].survey
            survey = m.get_survey(survey_id)
            survey.filters_group.add_filter(ELECTRIC_SI_TO_MT)
            survey.filters_group.add_filter(MAGNETIC_SI_TO_MT)
            survey.filters_group.add_filter(ELECTRIC_SI_TO_MT_UNKNOWN_UNITS)
            survey.filters_group.add_filter(MAGNETIC_SI_TO_MT_UNKNOWN_UNITS)
            survey.write_metadata()
            for i_row, row in channel_summary_df.iterrows():
                if row.measurement_type == "electric":
                    if row.num_filters == 2:
                        print("looks like a 22111, should probably check some other things too but ... assume SI for now")
                        channel = m.get_channel(row.station, row.run, row.component, row.survey)
                        current_filter = channel.metadata.filter
                        cfd = current_filter.to_dict()
                        cfd['filtered']["applied"] = [False, ] + cfd['filtered']["applied"]
                        units_known = check_units_are_known(channel)
                        if units_known:
                            cfd['filtered']["name"] = ["electric_si_units", ] + cfd['filtered']["name"]
                        else:
                            cfd['filtered']["name"] = ["electric_si_unknown_units", ] + cfd['filtered']["name"]
                        current_filter.from_dict(cfd)
                        channel.metadata.filter = current_filter
                        channel.write_metadata()
                elif row.measurement_type == "magnetic":
                    if row.num_filters == 1:
                        print("looks like a 22111, should probably check some other things too but ... assume SI for now")
                        channel = m.get_channel(row.station, row.run, row.component, row.survey)
                        current_filter = channel.metadata.filter
                        cfd = current_filter.to_dict()
                        cfd['filtered']["applied"] = [False, ] + cfd['filtered']["applied"]
                        units_known = check_units_are_known(channel)
                        if units_known:
                            cfd['filtered']["name"] = ["magnetic_si_units", ] + cfd['filtered']["name"]
                        else:
                            cfd['filtered']["name"] = ["magnetic_si_unknown_units", ] + cfd['filtered']["name"]
                        current_filter.from_dict(cfd)
                        channel.metadata.filter = current_filter
                        channel.write_metadata()
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
                    continue
                previous_run = earlier_runs.iloc[-1].run
                previous_channel = m.get_channel(row.station, previous_run, row.component, row.survey)
                channel_time_period = channel.metadata.time_period

                channel.metadata = previous_channel.metadata
                channel.metadata.time_period = channel_time_period
                #channel.metadata.filter = previous_channel.metadata.filter
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
