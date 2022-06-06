"""
This will likely wind up in aurora/transfer_function/tf_kernel/dataset.py
"""

import copy
import pandas as pd

import mth5
RUN_SUMMARY_COLUMNS = ["station_id", "run_id", "start", "end", "sample_rate",
                       "input_channels", "output_channels", "remote", "mth5_path"]
INPUT_CHANNELS = ["hx", "hy", ]
OUTPUT_CHANNELS = ["ex", "ey", "hz", ]

def channel_summary_to_run_summary(ch_summary,
                                   allowed_input_channels=INPUT_CHANNELS,
                                   allowed_output_channels=OUTPUT_CHANNELS):
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



    Returns
    -------

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
    return run_summary


class DatasetDefinition():
    """
    Could be called "ProcessableDataset", KernelDataset, InputDataset or something
    like that.  This is a specification of time series and intervals that is
    compatible with mth5.  The main idea is to specify one or two stations, together
    with a list of acquisition "runs" that can be merged into a "processing run".
    Each acquistion run can be further divided into non-overlapping chunks by specifying
    time-intervals associated with that acquistion run.  An empty iterable of
    time-intervals associated with a run is interpretted as the interval
    corresponding to the entire run.

    The time intervals can be used for several purposes but primarily:
    To specify contiguous chunks of data:
    1.  to STFT, that will be made into merged FC data structures
    2. to bind together into xarray time series, for eventual gap fill or STFT.

    The basic data strucutre can be represented as a table or as a tree:
    Station --> run --> [Intervals],
    where the --> symbol is reads "branches that specify (a)".

    This is described in issue #118 https://github.com/simpeg/aurora/issues/118

    Desired Properties
    a) This should be able to take a dictionary (tree) and return the tabular (
    DataFrame) representation and vice versa.
    b) Ability (when there are two or more stations) apply interval intersection
    rules, so that only time intervals when both stations are acquiring data are
    returned

    From (a) above we can see that a simple table per station can
    represent the available data.  That simple table can be generated by default from
    the mth5, and intervals to exclude some data can be added as needed.

    (b) is really just the case of considering pairs of tables like (a)

    In thinking all that through, I think we actually want a simple basecalss that
    we can call StationDataset.  The RR case will then be handled by pairing two of
    these, as StationPairDataset.

    In a perfect world, we would write the ChannelDataset class here, and make
    StationDataset a collection of ChannelDatasets, but that should be add-inable
    later.  For the full MMT case we could then consider ChannelPairDataset objects.

    In the DataFrame representation we have the following schema columns:
    station_id
    run_id,
    start,
    end,

    2022-03-11:
    Following notes in Issue #118, want to get a fully populated dataframe from an mth5.
    If I pass a station_id, then get all runs, if I pass a (station_id, run_id),
    then just get the run start and end times.
    This is basically a reformatting of the
    """
    def __init__(self, **kwargs):
        self.columns = ["station_id", "run_id", "start", "end"]
        self.column_dtypes = [str, str, pd.Timestamp, pd.Timestamp]
        self._input_dict = kwargs.get("input_dict", None)
        self.df = kwargs.get("df", None)

    def empty_dataframe(self):
        pass

    def from_mth5_channel_summary(self, channel_summary):
        if isinstance(channel_summary, mth5.tables.channel_table.ChannelSummaryTable):
            #this requires that the mth5 still be open
            channel_summary_df = channel_summary.to_dataframe()
        elif isinstance(channel_summary, pd.DataFrame):
            channel_summary_df = channel_summary
        df = channel_summary_to_run_summary(channel_summary_df)
        df.sort_values(by=["station_id", "run_id", "start"], inplace=True)
        self.df = df
        return self.df

    def restrict_to_station_list(self, station_ids, overwrite=False, df=None):
        """
        Drops all rows of dataset_definiation dataframe where station_ids are NOT in
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
            station_ids = [station_ids, ]
        if df is None:
            df = copy.deepcopy(self.df)
        cond1 = df["station_id"].isin(station_ids)
        df.drop(df[cond1].index, inplace=True)
        df = df.reset_index()
        if overwrite:
            self.df=df
        return df



    def restrict_runs_by_station(self, station_id, keep_run_ids, overwrite=False,
                                 df=None):
        """
        Drops all rows of dataset_definiation dataframe where station_id matches
        input arg, and run_id is NOT in the provided list of keep_run_ids.  Operates on
        a deepcopy of self.df if a df isn't provided

        Note1: Logic of keep/drop
        keep where cond1 is false
        keep where cond1 & cond2 both true
        drop where cond1 is true but cond2 is false

        Parameters
        ----------
        station_id: str
            The id of the station for which runs are to be dropped
        keep_run_ids: str or list of strings
            These are the run ids to keep.
        overwrite: bool
            If True, self.df is overwritten with the reduced dataframe

        Returns
        -------
            reduced dataframe with only run_ids provided removed.
        """
        if isinstance(keep_run_ids, str):
            keep_run_ids = [keep_run_ids, ]
        if df is None:
            df = copy.deepcopy(self.df)
        cond1 = df["station_id"]==station_id
        cond2 = df["run_id"].isin(keep_run_ids)
        #See Note1 above:
        drop_df = df[cond1 & ~cond2]
        df.drop(drop_df.index, inplace=True)
        df = df.reset_index()
        if overwrite:
            self.df=df
        return df




def main():
    return


if __name__ == "__main__":
    main()
