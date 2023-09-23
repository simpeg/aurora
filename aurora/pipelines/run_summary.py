"""

Note 1: Functionality of RunSummary()
1. User can get a list of local_station options, which correspond to unique pairs
of values: (survey_id,  station_id)

2. User can see all possible ways of processing the data:
- one list per (survey_id,  station_id) pair in the run_summary

Some of the following functionalities may end up in KernelDataset:
3. User can select local_station
-this can trigger a reduction of runs to only those that are from the local staion
and simultaneous runs at other stations
4. Given a local station, a list of possible reference stations can be generated
5. Given a remote reference station, a list of all relevent runs, truncated to
maximize coverage of the local station runs is generated
6. Given such a "restricted run list", runs can be dropped
7. Time interval endpoints can be changed


"""

import copy
import pandas as pd

import mth5
from mth5.utils.helpers import initialize_mth5


INPUT_CHANNELS = [
    "hx",
    "hy",
]
OUTPUT_CHANNELS = [
    "ex",
    "ey",
    "hz",
]
RUN_SUMMARY_COLUMNS = [
    "survey",
    "station_id",
    "run_id",
    "start",
    "end",
    "sample_rate",
    "input_channels",
    "output_channels",
    "remote",
    "mth5_path",
]


class RunSummary:
    """
    The dependencies aren't clear yet.
    Maybe still Dataset:
        Could have methods
            "drop_runs_shorter_than"
            "fill_gaps_by_time_interval"
            "fill_gaps_by_run_names"
            "

    For the full MMT case this may need modification to a channel based summary.

    Question: To return a copy or modify in-place when querying.  Need to decide on
    standards and syntax.  Handling this in general could use a decorator that allows
    df as kwarg, and if it is not passed the modification is done in place.
    The user who doesn't want to modify in place can work with a clone.
    Could also try the @staticmethod decorator so that it returns a modified df.

    """

    def __init__(self, **kwargs):
        self.column_dtypes = [str, str, pd.Timestamp, pd.Timestamp]
        self._input_dict = kwargs.get("input_dict", None)
        self.df = kwargs.get("df", None)
        self._mini_summary_columns = ["survey", "station_id", "run_id", "start", "end"]

    def clone(self):
        """
        2022-10-20:
        Cloning may be causing issues with extra instances of open h5 files ...

        """
        return copy.deepcopy(self)

    def from_mth5s(self, mth5_list):
        run_summary_df = extract_run_summaries_from_mth5s(mth5_list)
        self.df = run_summary_df

    @property
    def mini_summary(self):
        return self.df[self._mini_summary_columns]

    @property
    def print_mini_summary(self):
        print(self.mini_summary)

    def add_duration(self, df=None):
        """

        Parameters
        ----------
        df

        """
        if df is None:
            df = self.df
        timedeltas = df.end - df.start
        durations = [x.total_seconds() for x in timedeltas]
        df["duration"] = durations
        return

    def check_runs_are_valid(self, drop=False, **kwargs):
        """kwargs can tell us what sorts of conditions to check, for example all_zero, there are nan, etc."""
        # check_for_all_zero_runs
        for i_row, row in self.df.iterrows():
            print(f"Checking row for zeros {row}")
            m = mth5.mth5.MTH5()
            m.open_mth5(row.mth5_path)
            run_obj = m.get_run(row.station_id, row.run_id, row.survey)
            runts = run_obj.to_runts()
            if runts.dataset.to_array().data.__abs__().sum() == 0:
                print("CRITICAL: Detected a run with all zero values")
                self.df["valid"].at[i_row] = False
            # load each run, and take the median of the sum of the absolute values
        if drop:
            self.drop_invalid_rows()
        return

    def drop_invalid_rows(self):
        self.df = self.df[self.df.valid]
        self.df.reset_index(drop=True, inplace=True)

    # BELOW FUNCTION CAN BE COPIED FROM METHOD IN KernelDataset()
    # def drop_runs_shorter_than(self, duration, units="s"):
    #     if units != "s":
    #         raise NotImplementedError
    #     if "duration" not in self.df.columns:
    #         self.add_duration()
    #     drop_cond = self.df.duration < duration
    #     # df = self.df[drop_cond]
    #     self.df.drop(self.df[drop_cond].index, inplace=True)
    #     df = df.reset_index()
    #
    #     self.df = df
    #     return df


def channel_summary_to_run_summary(
    ch_summary,
    allowed_input_channels=INPUT_CHANNELS,
    allowed_output_channels=OUTPUT_CHANNELS,
    sortby=["station_id", "start"],
):
    """
    TODO: replace station_id with station, and run_id with run
    Note will need to modify: aurora/tests/config$ more test_dataset_dataframe.py
    TODO: Add logic for handling input and output channels based on channel
    summary.  Specifically, consider the case where there is no vertical magnetic
    field, this information is available via ch_summary, and output channels should
    then not include hz.
    TODO: Just inherit all the run-level and higher el'ts of the channel_summary,
    including n_samples?

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
        Default: ["station_id", "start"]

    Returns
    -------
    run_summary_df: pd.Dataframe
        A table with one row per "acquistion run" that was in the input channel
        summary table
    """
    if isinstance(ch_summary, mth5.tables.channel_table.ChannelSummaryTable):
        ch_summary_df = ch_summary.to_dataframe()
    elif isinstance(ch_summary, pd.DataFrame):
        ch_summary_df = ch_summary
    group_by_columns = ["survey", "station", "run"]
    grouper = ch_summary_df.groupby(group_by_columns)
    n_station_runs = len(grouper)
    survey_ids = n_station_runs * [None]
    station_ids = n_station_runs * [None]
    run_ids = n_station_runs * [None]
    start_times = n_station_runs * [None]
    end_times = n_station_runs * [None]
    sample_rates = n_station_runs * [None]
    input_channels = n_station_runs * [None]
    output_channels = n_station_runs * [None]
    channel_scale_factors = n_station_runs * [None]
    i = 0
    for group_values, group in grouper:
        group_info = dict(zip(group_by_columns, group_values))  # handy for debug
        # for k, v in group_info.items():
        #     print(f"{k} = {v}")
        survey_ids[i] = group_info["survey"]
        station_ids[i] = group_info["station"]
        run_ids[i] = group_info["run"]
        start_times[i] = group.start.iloc[0]
        end_times[i] = group.end.iloc[0]
        sample_rates[i] = group.sample_rate.iloc[0]
        channels_list = group.component.to_list()
        num_channels = len(channels_list)
        input_channels[i] = [x for x in channels_list if x in allowed_input_channels]
        output_channels[i] = [x for x in channels_list if x in allowed_output_channels]
        channel_scale_factors[i] = dict(zip(channels_list, num_channels * [1.0]))
        i += 1

    data_dict = {}
    data_dict["survey"] = survey_ids
    data_dict["station_id"] = station_ids
    data_dict["run_id"] = run_ids
    data_dict["start"] = start_times
    data_dict["end"] = end_times
    data_dict["sample_rate"] = sample_rates
    data_dict["input_channels"] = input_channels
    data_dict["output_channels"] = output_channels
    data_dict["channel_scale_factors"] = channel_scale_factors
    data_dict["valid"] = True

    run_summary_df = pd.DataFrame(data=data_dict)
    if sortby:
        run_summary_df.sort_values(by=sortby, inplace=True)
    return run_summary_df


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
    # check that the mth5 has been summarized already
    if len(channel_summary_df) < 2:
        print("Channel summary maybe not initialized yet, 3 or more channels expected.")
        mth5_obj.channel_summary.summarize()
        channel_summary_df = mth5_obj.channel_summary.to_dataframe()
    if summary_type == "run":
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
        if isinstance(mth5_elt, mth5.mth5.MTH5):
            mth5_obj = mth5_elt
        else:  # mth5_elt is a path or a string
            mth5_obj = initialize_mth5(mth5_elt, mode="a")

        df = extract_run_summary_from_mth5(mth5_obj, summary_type=summary_type)

        # close it back up if you opened it
        if not isinstance(mth5_elt, mth5.mth5.MTH5):
            mth5_obj.close_mth5()
        dfs[i] = df

    # merge all summaries into a super_summary
    super_summary = pd.concat(dfs)
    super_summary.reset_index(drop=True, inplace=True)

    # drop rows that correspond to TFs:
    run_rows = super_summary.sample_rate != 0
    super_summary = super_summary[run_rows]
    super_summary.reset_index(drop=True, inplace=True)

    if deduplicate:
        keep_indices = super_summary.astype(str).drop_duplicates().index
        super_summary = super_summary.loc[keep_indices]
        super_summary.reset_index(drop=True, inplace=True)
    return super_summary
