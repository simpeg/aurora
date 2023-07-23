import numpy as np
import pandas as pd
import psutil


from aurora.pipelines.helpers import initialize_config
from aurora.pipelines.time_series_helpers import prototype_decimate
from aurora.transfer_function.kernel_dataset import KernelDataset
from mth5.utils.helpers import initialize_mth5
from mt_metadata.transfer_functions.processing.aurora import Processing

class TransferFunctionKernel(object):
    def __init__(self, dataset=None, config=None):
        processing_config = initialize_config(config)
        self._config = processing_config
        self._dataset = dataset
        self._mth5_objs = None

    @property
    def dataset(self):
        return self._dataset

    @property
    def kernel_dataset(self):
        return self._dataset

    @property
    def dataset_df(self):
        return self._dataset.df

    @property
    def processing_config(self):
        return self._config

    @property
    def config(self):
        return self._config

    @property
    def processing_summary(self):
        if self._processing_summary is None:
            self.make_processing_summary()
        return self._processing_summary

    @property
    def mth5_objs(self):
        if self._mth5_objs is None:
            self.initialize_mth5s()
        return self._mth5_objs

    def initialize_mth5s(self):
        """
        returns a dict of open mth5 objects, keyed by
        Consder moving the initialize_mth5s() method out of config, since it depends on mth5.
        In this way, we can keep the dependencies on mth5 out of the metadatd object (of which config is one)

        Returns
        -------
        mth5_objs : dict
            Keyed by station_ids.
            local station id : mth5.mth5.MTH5
            remote station id: mth5.mth5.MTH5
        """

        local_mth5_obj = initialize_mth5(self.config.stations.local.mth5_path, mode="r")
        if self.config.stations.remote:
            remote_path = self.config.stations.remote[0].mth5_path
            remote_mth5_obj = initialize_mth5(remote_path, mode="r")
        else:
            remote_mth5_obj = None

        mth5_objs = {self.config.stations.local.id: local_mth5_obj}
        if self.config.stations.remote:
            mth5_objs[self.config.stations.remote[0].id] = remote_mth5_obj
        self._mth5_objs = mth5_objs
        return # mth5_objs

    def update_dataset_df(self,i_dec_level):
        """
        This could and probably should be moved to TFK.update_dataset_df()

        This function has two different modes.  The first mode, initializes values in the
        array, and could be placed into TFKDataset.initialize_time_series_data()
        The second mode, decimates. The function is kept in pipelines becasue it calls
        time series operations.


        Notes:
        1. When iterating over dataframe, (i)ndex must run from 0 to len(df), otherwise
        get indexing errors.  Maybe reset_index() before main loop? or push reindexing
        into TF Kernel, so that this method only gets a cleanly indexed df, restricted to
        only the runs to be processed for this specific TF?
        2. When assigning xarrays to dataframe cells, df dislikes xr.Dataset,
        so we convert to DataArray before assignment


        Parameters
        ----------
        i_dec_level: int
            decimation level id, indexed from zero
        config: mt_metadata.transfer_functions.processing.aurora.decimation_level.DecimationLevel
            decimation level config

        Returns
        -------
        dataset_df: pd.DataFrame
            Same df that was input to the function but now has columns:


        """
        if i_dec_level == 0:
            # Consider moving the line below into update_dataset_df.  This will allow bypassing loading of the data if
            # FC Level of mth5 is used.
            # Assign additional columns to dataset_df, populate with mth5_objs and xr_ts
            # ANY MERGING OF RUNS IN TIME DOMAIN WOULD GO HERE
            self.dataset.initialize_dataframe_for_processing(self.mth5_objs)

            # APPLY TIMING CORRECTIONS HERE
        else:
            print(f"DECIMATION LEVEL {i_dec_level}")
            # See Note 1 top of module
            # See Note 2 top of module
            for i, row in self.dataset_df.iterrows():
                if not self.is_valid_dataset(row, i_dec_level):
                    continue
                run_xrds = row["run_dataarray"].to_dataset("channel")
                decimation = self.config.decimations[i_dec_level].decimation
                decimated_xrds = prototype_decimate(decimation, run_xrds)
                self.dataset_df["run_dataarray"].at[i] = decimated_xrds.to_array("channel")

        print("DATASET DF UPDATED")
        return

    def check_if_fc_levels_already_exist(self):
        """
        Iterate over the processing summary_df, grouping by unique "Station-Run"s.
        When all FC Levels for a given station-run are already built, mark the RunSummary with a True in
        the (yet-to-be-built) mth5_has_FCs column
        Returns:

        """
        ps_df = self.processing_summary
        groupby = ['survey', 'station_id', 'run_id',]
        grouper = ps_df.groupby(groupby)
        print(len(grouper))
        for (survey, station_id, run_id), df in grouper:
            cond1 = self.dataset_df.survey==survey
            cond2 = self.dataset_df.station_id == station_id
            cond3 = self.dataset_df.run_id == run_id
            run_row = self.dataset_df[cond1 & cond2 & cond3].iloc[0]
            print("Need to update mth5_objs dict so that it is keyed by survey, then station, otra vez might break when "
                  "mixing in data from other surveys (if the stations are named the same)")
            # When addresssing the above issue -- consider adding the mth5_obj to self.dataset_df instead of keeping
            # the dict around ...
            mth5_obj = self.mth5_objs[station_id]
            # Make compatible with v0.1.0 and v0.2.0:
            # station_obj = mth5_obj.survey_group.stations_group.get_station(station_id)
            # INSERT get_survey here so it works with v010 and v020
            survey = mth5_obj.get_survey(survey)
            station_obj = survey.stations_group.get_station(station_id)
            if not station_obj.fourier_coefficients_group.groups_list:
                print("Nothign to see here folks, return False")
                return False
            else:
                print(df)
                print("New Logic for FC existence and satisfaction of processing requirements goes here")
                raise NotImplementedError

        return


    def make_processing_summary(self):
        """
        Melt the decimation levels over the run summary.  Add columns to estimate
        the number of FFT windows for each row

        Returns
        -------
        processing_summary: pd.DataFrame
            One row per each run-deciamtion pair
        """
        from aurora.time_series.windowing_scheme import WindowingScheme

        # Create correctly shaped dataframe
        tmp = self.kernel_dataset.df.copy(deep=True)
        id_vars = list(tmp.columns)
        decimation_info = self.config.decimation_info()
        for i_dec, dec_factor in decimation_info.items():
            tmp[i_dec] = dec_factor
        tmp = tmp.melt(id_vars=id_vars, value_name="dec_factor", var_name="dec_level")
        sortby = ["survey", "station_id", "run_id", "start", "dec_level"]
        tmp.sort_values(by=sortby, inplace=True)
        tmp.reset_index(drop=True, inplace=True)
        tmp.drop("sample_rate", axis=1, inplace=True)  # not valid for decimated data

        # Add window info
        group_by = [
            "survey",
            "station_id",
            "run_id",
            "start",
        ]
        groups = []
        grouper = tmp.groupby(group_by)
        for group, df in grouper:
            print(group)
            print(df)
            try:
                assert (df.dec_level.diff()[1:] == 1).all()  # dec levels increment by 1
                assert df.dec_factor.iloc[0] == 1
                assert df.dec_level.iloc[0] == 0
            except AssertionError:
                raise AssertionError("Decimation levels not structured as expected")
            # df.sample_rate /= np.cumprod(df.dec_factor)  # better to take from config
            window_params_df = self.config.window_scheme(as_type="df")
            df.reset_index(inplace=True, drop=True)
            df = df.join(window_params_df)
            df["num_samples"] = np.floor(df.duration * df.sample_rate)
            num_windows = np.zeros(len(df))
            for i, row in df.iterrows():
                ws = WindowingScheme(
                    num_samples_window=row.num_samples_window,
                    num_samples_overlap=row.num_samples_overlap,
                )
                num_windows[i] = ws.available_number_of_windows(row.num_samples)
            df["num_stft_windows"] = num_windows
            groups.append(df)

            # tmp.loc[df.index, "sample_rate"] = df.sample_rate
        processing_summary = pd.concat(groups)
        processing_summary.reset_index(drop=True, inplace=True)
        print(processing_summary)
        self._processing_summary = processing_summary
        return processing_summary

    def validate_decimation_scheme_and_dataset_compatability(
        self, min_num_stft_windows=2
    ):
        """
        Refers to issue #182 (and #103, and possibly #196 and #233).
        Determine if there exist (one or more) runs that will yield decimated datasets
        that have too few samples to be passed to the STFT algorithm.

        Strategy for handling this:
        Mark as invlaid rows of the processing summary that do not yield long
        enough time series to window.  This way all other rows, with decimations up to
        the invalid cases will still process.


        WCGW: If the runs are "chunked" we could encounter a situation where the
        chunk fails to yield a deep decimation level, yet the level could be produced
        if the chunk size were larger.
        In general, handling this seems a bit complicated.  We will ignore this for
        now and assume that the user will select a chunk size that is appropriate to
        the decimation scheme, i.e. use large chunks for deep decimations.

        A general solution: return a log that tells the user about each run and
        decimation level ... how many STFT-windows it yielded at each decimation level.
        This conjures the notion of (run, decimation_level) pairs
        -------

        """
        self.processing_summary["valid"] = (
            self.processing_summary.num_stft_windows >= min_num_stft_windows
        )

    def validate_processing(self):
        """
        Things that are validated:
        1. The default estimation engine from the json file is "RME_RR", which is fine (
        we expect to in general to do more RR processing than SS) but if there is only
        one station (no remote)then the RME_RR should be replaced by default with "RME".

        2. make sure local station id is defined (correctly from kernel dataset)
        """

        # Make sure a RR method is not being called for a SS config
        if not self.config.stations.remote:
            for decimation in self.config.decimations:
                if decimation.estimator.engine == "RME_RR":
                    print("No RR station specified, switching RME_RR to RME")
                    decimation.estimator.engine = "RME"

        # Make sure that a local station is defined
        if not self.config.stations.local.id:
            print("WARNING: Local station not specified")
            print("Setting local station from Kernel Dataset")
            self.config.stations.from_dataset_dataframe(self.kernel_dataset.df)

    def validate(self):
        self.validate_processing()
        self.validate_decimation_scheme_and_dataset_compatability()
        if self.memory_warning():
            raise Exception

    def valid_decimations(self):
        """

        Returns
        -------

        """
        # identify valid rows of processing summary
        tmp = self.processing_summary[self.processing_summary.valid]
        valid_levels = tmp.dec_level.unique()

        dec_levels = [x for x in self.config.decimations]
        dec_levels = [x for x in dec_levels if x.decimation.level in valid_levels]
        print(f"After validation there are {len(dec_levels)} valid decimation levels")
        return dec_levels

    def is_valid_dataset(self, row, i_dec):
        """
        Given a row from the RunSummary, answers:
        Will this decimation level yield a valid dataset?

        Parameters
        ----------
        row: pandas.core.series.Series
            Row of the self._dataset_df (corresponding to a run that will be processed)
        i_dec: integer
            refers to decimation level

        Returns
        -------
        is_valid: Bool
            Whether the (run, decimation_level) pair associated with this row yields
            a valid dataset
        """
        # Query processing on survey, station, run, decimation
        cond1 = self.processing_summary.survey == row.survey
        cond2 = self.processing_summary.station_id == row.station_id
        cond3 = self.processing_summary.run_id == row.run_id
        cond4 = self.processing_summary.dec_level == i_dec
        cond5 = self.processing_summary.start == row.start

        cond = cond1 & cond2 & cond3 & cond4 & cond5
        processing_row = self.processing_summary[cond]
        assert len(processing_row) == 1
        is_valid = processing_row.valid.iloc[0]
        return is_valid

    def memory_warning(self):
        """
        Checks if we should be anitcipating a RAM issue
        Requires an estimate of available RAM, and an estimate of the dataset size
        Available RAM is taken from psutil,
        Dataset size is number of samples, times the number of bytes per sample
        Bits per sample is estimated to be 64 by default, which is 8-bytes
        Returns
        -------

        """
        bytes_per_sample = 8  # 64-bit
        memory_threshold = 0.5

        # get the total amount of RAM in bytes
        total_memory = psutil.virtual_memory().total

        # print the total amount of RAM in GB
        print(f"Total memory: {total_memory / (1024 ** 3):.2f} GB")
        num_samples = self.dataset_df.duration * self.dataset_df.sample_rate
        total_samples = num_samples.sum()
        total_bytes = total_samples * bytes_per_sample
        print(f"Total Bytes of Raw Data: {total_bytes / (1024 ** 3):.3f} GB")

        ram_fraction = 1.0 * total_bytes / total_memory
        print(f"Raw Data will use: {100 * ram_fraction:.3f} % of memory")

        # Check a condition
        if total_bytes > memory_threshold * total_memory:
            return True
        else:
            return False
