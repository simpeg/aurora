"""
These tests are specific to widescale tests on earthscope.
They are not intended for genearl use, and are not robust.

They are intended to handle the following issues encountered during the summer of 2023 during widescale tests:
1. Different vintages of metadata;
Almost all the channel filters metadata encountered fell neatly into three categories

A)  Two filters for electric channels and 1 filter for magnetic -- results in SI units.
    num_filter_details: {'ex': 2, 'ey': 2, 'hx': 1, 'hy': 1, 'hz': 1}
    filter_units_in_details: {'ex': ['V/m', 'V'], 'ey': ['V/m', 'V'], 'hx': ['T'], 'hy': ['T'], 'hz': ['T']}
    filter_units_out_details: {'ex': ['V', 'count'], 'ey': ['V', 'count'], 'hx': ['count'], 'hy': ['count'], 'hz': ['count']}

B) Two filters for electric channels and 1 filter for magnetic -- results in mixed (including) unknown units
   num_filter_details: {'ex': 2, 'ey': 2, 'hx': 1, 'hy': 1, 'hz': 1}
   filter_units_in_details: {'ex': ['unknown', 'unknown'], 'ey': ['unknown', 'unknown'], 'hx': ['T'], 'hy': ['T'], 'hz': ['T']}
   filter_units_out_details: {'ex': ['unknown', 'unknown'], 'ey': ['unknown', 'unknown'], 'hx': ['unknown'], 'hy': ['unknown'], 'hz': ['unknown']}

C) Six filters for electric channels and three for for magnetic -- results in MT units
   num_filter_details: {'ex': 6, 'ey': 6, 'hx': 3, 'hy': 3, 'hz': 3}
   filter_units_in_details: {'ex': ['mV/km', 'V/m', 'V', 'V', 'V', 'count'], 'ey': ['mV/km', 'V/m', 'V', 'V', 'V', 'count'], 'hx': ['nT', 'V', 'count'], 'hy': ['nT', 'V', 'count'], 'hz': ['nT', 'V', 'count']}
   filter_units_out_details: {'ex': ['V/m', 'V', 'V', 'V', 'count', 'count'], 'ey': ['V/m', 'V', 'V', 'V', 'count', 'count'], 'hx': ['V', 'count', 'count'], 'hy': ['V', 'count', 'count'], 'hz': ['V', 'count', 'count']}

Cases A and B above are sometimes referred to by the shorthand 22111, and case C as 66333.

"""
from aurora.time_series.filters.filter_helpers import make_coefficient_filter

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
        print("looks like this one already got fixed")
        return False
    else:
        msg = f"Unexpected number of filters {num_filters}"
        raise NotImplementedError(msg)
        return False
    return True


def check_units_are_known(channel):
    """
    Returns True if there are unknown units in the channel.
    ToDo: add this to channel in mth5

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

def robustly_get_channel_summary(m):
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
        else:
            return channel_summary_df

def triage_22111(m, channel_summary_df):
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
    return


def add_filters_if_none_are_found(m, channel_summary_df):
    """deprecated, this was due to a bug in mth5 that is fixed"""
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
                # channel.metadata.filter = previous_channel.metadata.filter
                channel.write_metadata()
    return

def repair_missing_filters(mth5_path, mth5_version, triage_units=False, add_filters_where_none=False):
    """
    This is highly specific to wide-scale testing at IRIS Earthscope summer 2023, and should not be used in general.

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
    m = initialize_mth5(mth5_path, file_version=mth5_version)
    channel_summary_df = robustly_get_channel_summary(m)
    if not channel_summary_df:
        return

    channel_summary_df = enrich_channel_summary(m, channel_summary_df, "num_filters")

    # TRIAGE UNITS
    if triage_units:
        is_22111 = check_if_22111(channel_summary_df)
        if is_22111:
            triage_22111(m, channel_summary_df)

    if add_filters_where_none:
        add_filters_if_none_are_found(m, channel_summary_df)
    m.close_mth5()

