"""
Players on the stage:  One or more mth5s.

Each mth5 has a mth5_obj.channel_summary dataframe which tells what data are available.
Here we use a compressed view of this df with one line per acquisition run. I've been
calling that a "run_summary".  That object could be moved to mth5, so that each mth5
has a mth5_obj.run_summary dataframe.  As of Mar 29, 2023 a RunSummary is available at the
station level in mth5, but the aurora version is still being used.  This should be merged if possible
so that aurora uses the built-in mth5 method.

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
and in future maybe via some GUI (or a spreadsheet).


The process looks like this:
0. Start with a list of mth5s
1. Extract channel_summaries from each mth5 and join them vertically
2. Compress to a run_summay
3. Stare at the run_summary_df & Select a station "S" to process
4. Given "S"", select a non-empty set of runs for that station
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
cc = ConfigCreator()
p = cc.create_from_kernel_dataset(kernel_dataset, emtf_band_file=emtf_band_setup_file)
9. Edit the Processing Config appropriately,

ToDo: Consider supporting a default value for 'channel_scale_factors' that is None,
"""

import copy
import pandas as pd

from aurora.pipelines.run_summary import RUN_SUMMARY_COLUMNS
from mt_metadata.utils.list_dict import ListDict

# Add these to a standard, so we track add/subtract columns
KERNEL_DATASET_COLUMNS = RUN_SUMMARY_COLUMNS + [
    "channel_scale_factors",
    "duration",
    "fc",
]


class KernelDataset:
    """
    This class is intended to work with mth5-derived channel_summary or run_summary
    dataframes, that specify time series intervals.

    This class is closely related to (may actually be an extension of) RunSummary

    The main idea is to specify one or two stations, and a list of acquisition "runs"
    that can be merged into a "processing run".
    Each acquistion run can be further divided into non-overlapping chunks by specifying
    time-intervals associated with that acquistion run.  An empty iterable of
    time-intervals associated with a run is interpretted as the interval
    corresponding to the entire run.

    The time intervals can be used for several purposes but primarily:
    To specify contiguous chunks of data for:
    1. STFT, that will be made into merged FC data structures
    2. binding together into xarray time series, for eventual gap fill (and then STFT)
    3. managing and analyse the availability of reference time series

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


    Question: To return a copy or modify in-place when querying.  Need to decide on
    standards and syntax.  Handling this in general is messy because every function
    needs to be modified.  Maybe better to use a decorator that allows for df kwarg
    to be passed, and if it is not passed the modification is done in place.
    The user who doesn't want to modify in place can work with a clone.

    """

    def __init__(self, **kwargs):
        self.df = kwargs.get("df")
        self.local_station_id = kwargs.get("local_station_id")
        self.remote_station_id = kwargs.get("remote_station_id")
        self._mini_summary_columns = [
            "survey",
            "station_id",
            "run_id",
            "start",
            "end",
            "duration",
        ]
        self.survey_metadata = {}

    def clone(self):
        return copy.deepcopy(self)

    def clone_dataframe(self):
        return copy.deepcopy(self.df)

    def from_run_summary(self, run_summary, local_station_id, remote_station_id=None):
        """

        Parameters
        ----------
        run_summary: aurora.pipelines.run_summary.RunSummary
            Summary of available data for processing from one or more stations
        local_station_id: string
            Label of the station for which an estimate will be computed
        remote_station_id: string
            Label of the remote reference station

        Returns
        -------

        """
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
        if remote_station_id:
            self.restrict_run_intervals_to_simultaneous()
        # ADD A CHECK HERE df is non-empty
        if len(self.df) == 0:
            print("No Overlap between local and remote station data streams")
            print("Remote reference processing not a valid option")
        else:
            self._add_duration_column()
        self.df["fc"] = False

    @property
    def mini_summary(self):
        return self.df[self._mini_summary_columns]

    @property
    def print_mini_summary(self):
        print(self.mini_summary)

    @property
    def local_survey_id(self):
        survey_id = self.df.loc[~self.df.remote].survey.unique()[0]
        if survey_id in ["none"]:
            survey_id = "0"
        return survey_id

    @property
    def local_survey_metadata(self):
        return self.survey_metadata[self.local_survey_id]

    def _add_duration_column(self):
        """ """
        timedeltas = self.df.end - self.df.start
        durations = [x.total_seconds() for x in timedeltas]
        self.df["duration"] = durations
        return

    def drop_runs_shorter_than(self, duration, units="s"):
        """
        This needs to have duration refreshed before hand
        Parameters
        ----------
        duration
        units

        Returns
        -------

        """
        if units != "s":
            raise NotImplementedError
        if "duration" not in self.df.columns:
            self._add_duration_column()
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

        There is room for optimiztion here

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
                    local_row.start,
                    local_row.end,
                    remote_row.start,
                    remote_row.end,
                ):
                    # print(f"OVERLAP {i_local}, {i_remote}")
                    olap_start, olap_end = overlap(
                        local_row.start,
                        local_row.end,
                        remote_row.start,
                        remote_row.end,
                    )
                    # print(
                    #     f"{olap_start} -- {olap_end}\n "
                    #     f"{(olap_end-olap_start).seconds}s\n\n"
                    # )

                    local_sub_run = local_row.copy(deep=True)
                    remote_sub_run = remote_row.copy(deep=True)
                    local_sub_run.start = olap_start
                    local_sub_run.end = olap_end
                    remote_sub_run.start = olap_start
                    remote_sub_run.end = olap_end
                    output_sub_runs.append(local_sub_run)
                    output_sub_runs.append(remote_sub_run)
                else:
                    pass
                    # print(f"NOVERLAP {i_local}, {i_remote}")
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
            local_run_obj = self.get_run_object(row)
            if station_metadata is None:
                station_metadata = local_run_obj.station_metadata
                station_metadata.runs = ListDict()
            run_metadata = local_run_obj.metadata
            station_metadata.add_run(run_metadata)
        return station_metadata

    @property
    def num_sample_rates(self):
        return len(self.df.sample_rate.unique())

    @property
    def sample_rate(self):
        if self.num_sample_rates != 1:
            msg = "Aurora does not yet process data from mixed sample rates"
            print(f"{msg}")
            raise NotImplementedError(msg)
        sample_rate = self.df.sample_rate.unique()[0]
        return sample_rate

    def initialize_dataframe_for_processing(self, mth5_objs):
        """
        Adds extra columns needed for processing, populates them with mth5 objects,
        run_reference, and xr.Datasets.

        Note #1: When assigning xarrays to dataframe cells, df dislikes xr.Dataset,
        so we convert to xr.DataArray before packing df

        Note #2: [OPTIMIZATION] By accesssing the run_ts and packing the "run_dataarray" column of the df with it, we
         perform a non-lazy operation, and essentially forcing the entire decimation_level=0 dataset to be
         loaded into memory.  Seeking a lazy method to handle this maybe worthwhile.  For example, using
         a df.apply() approach to initialize only ione row at a time would allow us to gernerate the FCs one
         row at a time and never ingest more than one run of data at a time ...


        Parameters
        ----------
        mth5_objs: dict,  keyed by station_id
        """

        self.add_columns_for_processing(mth5_objs)

        for i, row in self.df.iterrows():
            run_obj = row.mth5_obj.get_run(
                row.station_id, row.run_id, survey=row.survey
            )
            self.df["run_reference"].at[i] = run_obj.hdf5_group.ref

            if row.fc:
                msg = f"row {row} already has fcs prescribed by processing confg "
                msg += "-- skipping time series initialzation"
                print(msg)
            #    continue
            # the line below is not lazy, See Note #2
            run_ts = run_obj.to_runts(start=row.start, end=row.end)
            self.df["run_dataarray"].at[i] = run_ts.dataset.to_array("channel")

            # wrangle survey_metadata into kernel_dataset
            survey_id = run_ts.survey_metadata.id
            if i == 0:
                self.survey_metadata[survey_id] = run_ts.survey_metadata
            elif i > 0:
                if row.station_id in self.survey_metadata[survey_id].stations.keys():
                    self.survey_metadata[survey_id].stations[row.station_id].add_run(
                        run_ts.run_metadata
                    )
                else:
                    self.survey_metadata[survey_id].add_station(run_ts.station_metadata)
            if len(self.survey_metadata.keys()) > 1:
                raise NotImplementedError

        print("DATASET DF POPULATED")

    def add_columns_for_processing(self, mth5_objs):
        """
        Moving this into kernel_dataset from processing_pipeline

        Q: Should mth5_objs be keyed by survey-station?
        A: Yes, and ...
        since the KernelDataset dataframe will be iterated over we should probably
        write an iterator method.  This can iterate over survey-station tuples
        for multiple station processing.

        Parameters
        ----------
        mth5_objs: dict,  keyed by station_id

        """
        columns_to_add = ["run_dataarray", "stft", "run_reference"]
        mth5_obj_column = len(self.df) * [None]
        for i, station_id in enumerate(self.df["station_id"]):
            mth5_obj_column[i] = mth5_objs[station_id]
        self.df["mth5_obj"] = mth5_obj_column
        for column_name in columns_to_add:
            self.df[column_name] = None

    def get_run_object(self, index_or_row):
        """

        Parameters
        ----------
        index_or_row: integer index of df, or pd.Series object

        Returns
        -------

        """
        if isinstance(index_or_row, int):
            row = self.df.loc[index_or_row]
        else:
            row = index_or_row
        run_obj = row.mth5_obj.from_reference(row.run_reference)
        return run_obj

    def close_mths_objs(self):
        """
        Loop over all unique mth5_objs in the df and make sure they are closed

        """
        mth5_objs = self.df["mth5_obj"].unique()
        for mth5_obj in mth5_objs:
            mth5_obj.close_mth5()
        return


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
    start1: pd.Timestamp
        Start of interval 1
    end1: pd.Timestamp
        End of interval 1
    start2: pd.Timestamp
        Start of interval 2
    end2: pd.Timestamp
        End of interval 2

    Returns
    -------
    cond: bool
        True of the intervals overlap, False if they do now

    """
    cond = (start1 <= start2 <= end1) or (start2 <= start1 <= end2)
    return cond


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
        return None, None


def main():
    return


if __name__ == "__main__":
    main()
