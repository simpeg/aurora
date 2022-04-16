"""
2022-03-11
Placeholder for TFKernel which defines the inputs and methods for generating a TF

Players on the stage:  One or more mth5s.  

Each mth5 has a mth5_obj.channel_summary dataframe which tells what data are available.
I am using a compressed view of this df with one line per acquisition run as the main
reference point for defining the TFKernel.  I've been calling that a
"dataset_definition" but I think it is better termed a "run_summary", and this could
likely be pushed up into mth5, so that each mth5 has a mth5_obj.run_summary dataframe.

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
import pandas as pd


from mth5.utils.helpers import initialize_mth5

class TransferFunctionKernel(object):
    """

    """
    def __init__(self, **kwargs):
        self.mth5_path = kwargs.get("mth5_path", "")
        self.dataset = None
        self.processing = None

    def get_channel_summary(self, csv_path=None, mth5_obj=None, mth5_path=None):
        """
        2022-03-18: Modify this to accept lists of mth5_objs and mth5_paths.
        In this way, comprehensive summaries can be built across several mth5s.
        Parameters
        ----------
        csv_path: string or path
            This option makes testing a lot faster by skipping the mth5 channel
            summary creation
        mth5_obj
        mth5_path

        Returns
        -------
        channel_summary: pd.DataFrame (see mth5 channel_summary for doc)

        """
        if csv_path:
            print("accessing channel summary from file, faster but less robust")
            df = pd.read_csv(csv_path, parse_dates=["start", "end"])
        elif mth5_obj:
            df = mth5_obj.channel_summary
        else:
            mth5_obj = initialize_mth5(self.mth5_path, mode="r")
            df = mth5_obj.channel_summary
            mth5_obj.close_mth5()
        return df

    def validate_channel_summary(self):
        """
        Sanity check that each station-run has a unique start and end per channel
        Returns
        -------

        """
        pass