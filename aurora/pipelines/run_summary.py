"""

This module contains the RunSummary class.

This is a helper class that summarizes the Runs in an mth5.

TODO: This class and methods could be replaced by methods in MTH5.

Functionality of RunSummary()
1. User can get a list of local_station options, which correspond to unique pairs
of values: (survey,  station)

2. User can see all possible ways of processing the data:
- one list per (survey,  station) pair in the run_summary

Some of the following functionalities may end up in KernelDataset:
3. User can select local_station
-this can trigger a reduction of runs to only those that are from the local staion
and simultaneous runs at other stations
4. Given a local station, a list of possible reference stations can be generated
5. Given a remote reference station, a list of all relevent runs, truncated to
maximize coverage of the local station runs is generated
6. Given such a "restricted run list", runs can be dropped
7. Time interval endpoints can be changed


Development Notes:
    TODO: consider adding methods:
     "drop_runs_shorter_than": removes short runs from summary
     "fill_gaps_by_time_interval": allows runs to be merged if gaps between are short
     "fill_gaps_by_run_names": allows runs to be merged if gaps between are short
    TODO: Consider whether this should return a copy or modify in-place when querying the df.

"""

import copy
import pandas as pd

import mth5
from mth5.utils.helpers import initialize_mth5
from loguru import logger
from typing import Optional, Union

RUN_SUMMARY_COLUMNS = [
    "survey",
    "station",
    "run",
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
    Class to contain a run-summary table from one or more mth5s.
            "
    WIP: For the full MMT case this may need modification to a channel based summary.


    """

    def __init__(
        self,
        input_dict: Optional[Union[dict, None]] = None,
        df: Optional[Union[pd.DataFrame, None]] = None,
    ):
        """
        Constructor

        Parameters
        ----------
        kwargs
        """
        self.column_dtypes = [str, str, pd.Timestamp, pd.Timestamp]
        self._input_dict = input_dict
        self.df = df
        self._mini_summary_columns = [
            "survey",
            "station",
            "run",
            "start",
            "end",
        ]

    def clone(self):
        """
        2022-10-20:
        Cloning may be causing issues with extra instances of open h5 files ...

        """
        return copy.deepcopy(self)

    def from_mth5s(self, mth5_list):
        """Iterates over mth5s in list and creates one big dataframe summarizing the runs"""
        run_summary_df = extract_run_summaries_from_mth5s(mth5_list)
        self.df = run_summary_df

    @property
    def mini_summary(self):
        """shows the dataframe with only a few columns for readbility"""
        return self.df[self._mini_summary_columns]

    @property
    def print_mini_summary(self):
        """Calls minisummary through logger so it is formatted."""
        logger.info(self.mini_summary)

    def add_duration(
        self, df: Optional[Union[pd.DataFrame, None]] = None
    ) -> None:
        """
        Adds a column called "duration" to the dataframe

        Parameters
        ----------
        df: Optional[Union[pd.DataFrame, None]]
            If not provided use self.df

        """
        if df is None:
            df = self.df
        timedeltas = df.end - df.start
        durations = [x.total_seconds() for x in timedeltas]
        df["duration"] = durations
        return

    def check_runs_are_valid(self, drop: bool = False):
        """

        Checks for runs that are identically zero.
        TODO: Add optional arguments for other conditions to check, for example there are nan, etc.

        Parameters
        ----------
        drop: bool
            If True, drop invalid rows from dataframe

        """
        # check_for_all_zero_runs
        for i_row, row in self.df.iterrows():
            logger.info(f"Checking row for zeros {row}")
            m = mth5.mth5.MTH5()
            m.open_mth5(row.mth5_path)
            run_obj = m.get_run(row.station, row.run, row.survey)
            runts = run_obj.to_runts()
            if runts.dataset.to_array().data.__abs__().sum() == 0:
                logger.critical("CRITICAL: Detected a run with all zero values")
                self.df["valid"].at[i_row] = False
            # load each run, and take the median of the sum of the absolute values
        if drop:
            self.drop_invalid_rows()
        return

    def drop_invalid_rows(self) -> None:
        """
        Drops rows marked invalid (df.valid is False) and resets the index of self.df

        """
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


def extract_run_summary_from_mth5(mth5_obj, summary_type="run"):
    """
    Given a single mth5 object, get the channel_summary and compress it to a run_summary.

    Development Notes:
    TODO: Move this into MTH5 or replace with MTH5 built-in run_summary method.

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

    if summary_type == "run":
        out_df = mth5_obj.run_summary
    else:
        out_df = mth5_obj.channel_summary.to_dataframe()
    out_df["mth5_path"] = str(mth5_obj.filename)
    return out_df


def extract_run_summaries_from_mth5s(
    mth5_list, summary_type="run", deduplicate=True
):
    """
    Given a list of mth5's, iterate over them, extracting run_summaries and merging into one big table.

    Development Notes:
    ToDo: Move this method into mth5? or mth5_helpers?
    ToDo: Make this a class so that the __repr__ is a nice visual representation of the
    df, like what channel summary does in mth5
    - 2022-05-28 Modified to allow this method to accept mth5 objects as well as the
    already supported types of pathlib.Path or str


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
        Given a list of mth5s, a dataframe of all available runs

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
