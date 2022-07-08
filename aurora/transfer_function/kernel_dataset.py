"""
This may wind up in aurora/transfer_function/kernel_dataset.py

Players on the stage:  One or more mth5s.

Each mth5 has a mth5_obj.channel_summary dataframe which tells what data are available.
I am using a compressed view of this df with one line per acquisition run as the main
reference point for defining the TFKernel.  I've been calling that a "run_summary", and
 this could likely be pushed up into mth5, so that each mth5 has a
 mth5_obj.run_summary dataframe.

The run_summary provides options for the local and possibly remote reference stations.
 Candidates for local station are the unique value in the station column.
 *It maybe that we need to groupby survey & station, for now I am considering station
 names to be unique.

 For any given candidate station, there are some integer n runs available.
 This yields 2**n - 1 possible combinations that can be processed, neglecting any
 flagging of time intervals within any run, or any joining of runs.
 (There are actually 2**n, but we ignore the empty set, so -1)

 Intuition suggests default ought to be to process n runs in n+1 configurations:
 {all runs} + each run individually.  This will give a bulk answer, and bad runs can
 be flagged by comparing them.  After an initial processing, the tfs can be reviewed
 and the problematic runs can be addressed.

The user can interact with the run_summary_df, selecting sub dataframes via querying,
and in future maybe via some GUI (or a spreadsheet!).




The Decision Tree looks like this:
0. Start with a list of mth5s
1. Extract channel_summaries from each mth5 and join them vertically
2. Compress to a run_summay
3. Stare at the run_summary_df & Select a station "S" to process
4. For the given station, select a non-empty set of runs for that station
5. Select a remote reference "RR", (this is allowed to be empty)
6. Extract the sub-dataframe corresponding to you local_station acquistion_runs,
and the remote station acquition runs
7. If the remote is non-empty,
a) Drop the runs (rows) associated with the remote that DO NOT intersect with local
b) restrict the start/end times of the remote runs that DO intersect with the
local so that overlap is complete.
c) restrict start/end times of the local runs so that they DO intersect with remote
8. This is now a TFKernel Dataset Definition (ish).  Initialize a default
processing object and pass it this df:
cc = ConfigCreator(config_path=CONFIG_PATH)
p = cc.create_run_processing_object(emtf_band_file=emtf_band_setup_file)
p.stations.from_dataset_dataframe(dd_df)
9. Edit the Processing appropriately,

"""

import copy
import pandas as pd


class KernelDataset:
    """
    Could be called "ProcessableDataset", KernelDataset, InputDataset or something
    like that.  This class is intended to work with mth5-derived channel_summary or
    run_summary dataframes, that specify time series intervals.

    This class is closely related to (may actually be an extension of) RunSummary

    The main idea is to specify one or two stations, and a list of acquisition "runs"
    that can be merged into a "processing run".
    Each acquistion run can be further divided into non-overlapping chunks by specifying
    time-intervals associated with that acquistion run.  An empty iterable of
    time-intervals associated with a run is interpretted as the interval
    corresponding to the entire run.

    The time intervals can be used for several purposes but primarily:
    To specify contiguous chunks of data:
    1.  to STFT, that will be made into merged FC data structures
    2. to bind together into xarray time series, for eventual gap fill (and then STFT)
    3. To manage and analyse the availability of reference time series

    The basic data strucutre can be represented as a table or as a tree:
    Station <-- run <-- [Intervals],

    This is described in issue #118 https://github.com/simpeg/aurora/issues/118

    Desired Properties
    a) This should be able to take a dictionary (tree) and return the tabular (
    DataFrame) representation and vice versa.
    b) When there are two stations, can apply interval intersection rules, so that
    only time intervals when both stations are acquiring data are kept

    From (a) above we can see that a simple table per station can
    represent the available data.  That table can be generated by default from
    the mth5, and intervals to exclude some data can be added as needed.

    (b) is really just the case of considering pairs of tables like (a)



    2022-03-11:
    Following notes in Issue #118, want to get a fully populated dataframe from an mth5.
    If I pass a station_id, then get all runs, if I pass a (station_id, run_id),
    then just get the run start and end times.

    # Question: To return a copy or modify in-place when querying.  Need to decide on
    # standards and syntax.  Handling this in general is messy because every function
    # needs to be modified.  Maybe better to use a decorator that allows for df kwarg
    # to be passed, and if it is not passed the modification is done in place.
    # The user who doesn't want to modify in place can work with a clone.

    """

    def __init__(self, **kwargs):
        self.df = kwargs.get("df")
        self.local_station_id = kwargs.get("local_station_id")
        self.remote_station_id = kwargs.get("remote_station_id")
        self._mini_summary_columns = ["station_id", "run_id", "start", "end"]

    def clone(self):
        return copy.deepcopy(self)

    def clone_dataframe(self):
        return copy.deepcopy(self.df)

    def from_run_summary(self, run_summary, local_station_id, remote_station_id=None):
        self.local_station_id = local_station_id
        self.remote_station_id = remote_station_id

        station_ids = [
            local_station_id,
        ]
        if remote_station_id:
            station_ids.append(remote_station_id)
        df = restrict_to_station_list(run_summary.df, station_ids, inplace=False)
        df["remote"] = False
        if remote_station_id:
            cond = df.station_id == remote_station_id
            df.remote = cond

        self.df = df

    @property
    def mini_summary(self):
        print(self.df[self._mini_summary_columns])

    @property
    def add_duration(self):
        """ """
        timedeltas = self.df.end - self.df.start
        durations = [x.total_seconds() for x in timedeltas]
        self.df["duration"] = durations
        return

    def drop_runs_shorter_than(self, duration, units="s"):
        if units != "s":
            raise NotImplementedError
        if "duration" not in self.df.columns:
            self.add_duration
        drop_cond = self.df.duration < duration
        self.df.drop(self.df[drop_cond].index, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        return

    def select_station_runs(self, station_runs_dict, keep_or_drop):
        df = select_station_runs(self.df, station_runs_dict, keep_or_drop)
        self.df = df
        return

    @property
    def is_single_station(self):
        if self.local_station_id:
            if self.remote_station_id:
                return False
            else:
                return True
        else:
            return False

    @property
    def is_remote_reference(self):
        raise NotImplementedError

    def restrict_run_intervals_to_simultaneous(self):
        """
        For each run in local_station_id we check if it has overlap with other runs

        Room for optimiztion here

        Note that you can wind up splitting runs here.  For example, in that case where
        local is running continuously, but remote is intermittent.  Then the local
        run may break into several chunks.

        Returns
        -------

        """
        local_df = self.df[self.df.station_id == self.local_station_id]
        remote_df = self.df[self.df.station_id == self.remote_station_id]
        output_sub_runs = []
        for i_local, local_row in local_df.iterrows():
            for i_remote, remote_row in remote_df.iterrows():
                if intervals_overlap(
                    local_row.start, local_row.end, remote_row.start, remote_row.end
                ):
                    print(f"OVERLAP {i_local}, {i_remote}")
                    olap_start, olap_end = overlap(
                        local_row.start, local_row.end, remote_row.start, remote_row.end
                    )
                    print(
                        f"{olap_start} -- {olap_end}\n "
                        f"{(olap_end-olap_start).seconds}s\n\n"
                    )

                    local_sub_run = local_row.copy(deep=True)
                    # local_sub_run.drop("index", inplace=True)
                    remote_sub_run = remote_row.copy(deep=True)
                    # remote_sub_run.drop("index", inplace=True)
                    local_sub_run.start = olap_start
                    local_sub_run.end = olap_end
                    remote_sub_run.start = olap_start
                    remote_sub_run.end = olap_end
                    output_sub_runs.append(local_sub_run)
                    output_sub_runs.append(remote_sub_run)
                else:
                    print(f"NOVERLAP {i_local}, {i_remote}")
        df = pd.DataFrame(output_sub_runs)
        df = df.reset_index(drop=True)
        self.df = df
        return

    def get_station_metadata(self, local_station_id):
        """
        Helper function for archiving the TF

        Parameters
        ----------
        local_station_id: str
            The name of the local station

        Returns
        -------

        """
        # get a list of local runs:
        cond = self.df["station_id"] == local_station_id
        sub_df = self.df[cond]
        sub_df.drop_duplicates(subset="run_id", inplace=True)

        # sanity check:
        run_ids = sub_df.run_id.unique()
        assert len(run_ids) == len(sub_df)

        # iterate over these runs, packing metadata into
        station_metadata = None
        for i, row in sub_df.iterrows():
            local_run_obj = row.run
            if station_metadata is None:
                station_metadata = local_run_obj.station_group.metadata
                station_metadata._runs = []
            run_metadata = local_run_obj.metadata
            station_metadata.add_run(run_metadata)
        return station_metadata


def restrict_to_station_list(df, station_ids, inplace=True):
    """
    Drops all rows of run_summary dataframe where station_ids are NOT in
    the provided list of station_ids.  Operates on a deepcopy of self.df if a df
    isn't provided

    Parameters
    ----------
    station_ids: str or list of strings
        These are the station ids to keep, normally local and remote
    overwrite: bool
        If True, self.df is overwritten with the reduced dataframe

    Returns
    -------
        reduced dataframe with only stations associated with the station_ids
    """
    if isinstance(station_ids, str):
        station_ids = [
            station_ids,
        ]
    if not inplace:
        df = copy.deepcopy(df)
    cond1 = ~df["station_id"].isin(station_ids)
    df.drop(df[cond1].index, inplace=True)
    df = df.reset_index(drop=True)
    return df


def select_station_runs(
    df,
    station_runs_dict,
    keep_or_drop,
    overwrite=True,
):
    """
    Drops all rows where station_id==station_id, and run_id is NOT in the provided
     list of keep_run_ids.  Operates on a deepcopy df if inplace=False
    Uncommon use case the way this is coded, because it will restrict to a single
    station processing case.  Better to use drop runs, or a dict-style input

    Note1: Logic of keep/drop
    keep where cond1 is false
    keep where cond1 & cond2 both true
    drop where cond1 is true but cond2 is false

    Parameters
    ----------
    station_runs_dict: dict
        Keys are string ids of the stations to keep
        Values are lists of string labels for run_ids to keep
    keep_or_drop: str
        If "keep": returns df with only the station_rus specified in station_runs_dict
        If "drop": returns df with station_runs_dict excised
    overwrite: bool
        If True, self.df is overwritten with the reduced dataframe

    Returns
    -------
        reduced dataframe with only run_ids provided removed.
    """

    if not overwrite:
        df = copy.deepcopy(df)
    for station_id, run_ids in station_runs_dict.items():
        if isinstance(run_ids, str):
            run_ids = [
                run_ids,
            ]
        cond1 = df["station_id"] == station_id
        cond2 = df["run_id"].isin(run_ids)
        if keep_or_drop == "keep":
            drop_df = df[cond1 & ~cond2]
        else:
            drop_df = df[cond1 & cond2]

        df.drop(drop_df.index, inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


def intervals_overlap(start1, end1, start2, end2):
    """
    https://stackoverflow.com/questions/3721249/python-date-interval-intersection

    Parameters
    ----------
    start1
    end1
    start2
    end2

    Returns
    -------

    """
    return (start1 <= start2 <= end1) or (start2 <= start1 <= end2)


def overlap(t1start, t1end, t2start, t2end):
    """
    https://stackoverflow.com/questions/3721249/python-date-interval-intersection

    Parameters
    ----------
    t1start
    t1end
    t2start
    t2end

    Returns
    -------

    """
    if t1start <= t2start <= t2end <= t1end:
        return t2start, t2end
    elif t1start <= t2start <= t1end:
        return t2start, t1end
    elif t1start <= t2end <= t1end:
        return t1start, t2end
    elif t2start <= t1start <= t1end <= t2end:
        return t1start, t1end
    else:
        return None


def main():
    return


if __name__ == "__main__":
    main()
