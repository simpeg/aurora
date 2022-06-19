import pandas as pd

import mth5
from mth5.mth5 import MTH5
from mth5.utils.helpers import initialize_mth5


INPUT_CHANNELS = ["hx", "hy", ]
OUTPUT_CHANNELS = ["ex", "ey", "hz", ]

def channel_summary_to_run_summary(ch_summary,
                                   allowed_input_channels=INPUT_CHANNELS,
                                   allowed_output_channels=OUTPUT_CHANNELS,
                                   sortby=["station_id", "run_id", "start"]):
    """
    TODO: replace station_id with station, and run_id with run
    TODO: Add logic for handling input and output channels based on channel
    summary.  Specifically, consider the case where there is no vertical magnetic
    field, this information is available via ch_summary, and output channels should
    then not include hz.

    When creating the dataset dataframe, make it have these columns:
    [
            "station_id",
            "run_id",
            "start",
            "end",
            "mth5_path",
            "sample_rate",
            "input_channels",
            "output_channels",
            "remote",
	    "channel_scale_factors",
        ]

    Parameters
    ----------
    ch_summary: mth5.tables.channel_table.ChannelSummaryTable or pandas DataFrame
       If its a dataframe it is a representation of an mth5 channel_summary.
        Maybe restricted to only have certain stations and runs before being passed to
        this method
    allowed_input_channels: list of strings
        Normally ["hx", "hy", ]
        These are the allowable input channel names for the processing.  See further
        note under allowed_output_channels.
    allowed_output_channels: list of strings
        Normally ["ex", "ey", "hz", ]
        These are the allowable output channel names for the processing.
        A global list of these is kept at the top of this module.  The purpose of
        this is to distinguish between runs that have different layouts, for example
        some runs will have hz and some will not, and we cannot process for hz the
        runs that do not have it.  By making this a kwarg we sort of prop the door
        open for more general names (see issue #74).
    sortby: bool or list
        Default: ["station_id", "run_id", "start"]

    Returns
    -------
    run_summary: pd.Dataframe
        A table with one row per "acquistion run" that was in the input channel
        summary table
    """
    if isinstance(ch_summary, mth5.tables.channel_table.ChannelSummaryTable):
        ch_summary_df = ch_summary.to_dataframe()
    elif isinstance(ch_summary, pd.DataFrame):
        ch_summary_df = ch_summary
    grouper = ch_summary_df.groupby(["station", "run"])
    n_station_runs = len(grouper)
    station_ids = n_station_runs * [None]
    run_ids = n_station_runs * [None]
    start_times = n_station_runs * [None]
    end_times = n_station_runs * [None]
    sample_rates = n_station_runs * [None]
    input_channels = n_station_runs * [None]
    output_channels = n_station_runs * [None]
    channel_scale_factors = n_station_runs * [None]
    i = 0
    for (station_id, run_id), group in grouper:
        #print(f"{i} {station_id} {run_id}")
        #print(group)
        station_ids[i] = station_id
        run_ids[i] = run_id
        start_times[i] = group.start.iloc[0]
        end_times[i] = group.end.iloc[0]
        sample_rates[i] = group.sample_rate.iloc[0]
        channels_list = group.component.to_list()
        num_channels = len(channels_list)
        input_channels[i] = [x for x in channels_list if x in allowed_input_channels]
        output_channels[i] = [x for x in channels_list if x in allowed_output_channels]
        channel_scale_factors[i] = dict(zip(channels_list, num_channels*[1.0]))
        i += 1

    data_dict = {}
    data_dict["station_id"] = station_ids
    data_dict["run_id"] = run_ids
    data_dict["start"] = start_times
    data_dict["end"] = end_times
    data_dict["sample_rate"] = sample_rates
    data_dict["input_channels"] = input_channels
    data_dict["output_channels"] = output_channels
    data_dict["channel_scale_factors"] = channel_scale_factors
    run_summary = pd.DataFrame(data=data_dict)
    if sortby:
        run_summary.sort_values(by=sortby, inplace=True)
    return run_summary



def extract_run_summary_from_mth5(mth5_obj, summary_type="run"):
    """

    Parameters
    ----------
    mth5_obj: mth5.mth5.MTH5
        The initialized mth5 object that will be interrogated
    summary_type: str
        One of ["run", "channel"].  Returns a run summary or a channel summary

    Returns
    -------
    out_df: pd.Dataframe
        Table summarizing the available runs in the input mth5_obj
    """
    channel_summary_df = mth5_obj.channel_summary.to_dataframe()
    #check that the mth5 has been summarized already
    if len(channel_summary_df) < 2:
        print("The channel summary may not have been initialized yet, at least 4 "
              "channels are expected.")
        mth5_obj.channel_summary.summarize()
        channel_summary_df = mth5_obj.channel_summary.to_dataframe()
    if summary_type=="run":
        out_df = channel_summary_to_run_summary(channel_summary_df)
    else:
        out_df = channel_summary_df
    out_df["mth5_path"] = str(mth5_obj.filename)
    return out_df


def extract_run_summaries_from_mth5s(mth5_list, summary_type="run", deduplicate=True):
    """
    ToDo: Move this method into mth5? or mth5_helpers?
    ToDo: Make this a class so that the __repr__ is a nice visual representation of the
    df, like what channel summary does in mth5

    2022-05-28 Modified to allow this method to accept mth5 objects as well as the
    already supported types of pathlib.Path or str

    Given a list of mth5s, this returns a dataframe of all available runs

    In order to drop duplicates I used the solution here:
    https://stackoverflow.com/questions/43855462/pandas-drop-duplicates-method-not-working-on-dataframe-containing-lists

    Parameters
    ----------
    mth5_paths: list
        paths or strings that point to mth5s
    summary_type: string
        one of ["channel", "run"]
        "channel" returns concatenated channel summary, 
        "run" returns concatenated run summary,
    deduplicate: bool
        Default is True, deduplicates the summary_df

    Returns
    -------
    super_summary: pd.DataFrame

    """
    dfs = len(mth5_list) * [None]

    for i, mth5_elt in enumerate(mth5_list):
        if isinstance(mth5_elt, MTH5):
            mth5_obj = mth5_elt
        else:   #mth5_elt is a path or a string
            mth5_obj = initialize_mth5(mth5_elt, mode="a")

        df = extract_run_summary_from_mth5(mth5_obj, summary_type=summary_type)

        #close it back up if you opened it
        if not isinstance(mth5_elt, MTH5):
            mth5_obj.close_mth5()
        dfs[i] = df

    #merge all summaries into a super_summary
    super_summary = pd.concat(dfs)
    super_summary.reset_index(drop=True, inplace=True)
    if deduplicate:
        keep_indices = super_summary.astype(str).drop_duplicates().index
        super_summary = super_summary.loc[keep_indices]
        super_summary.reset_index(drop=True, inplace=True)
    return super_summary