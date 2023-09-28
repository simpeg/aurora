import numpy as np
import pandas as pd
import psutil


from aurora.pipelines.helpers import initialize_config
from aurora.pipelines.time_series_helpers import prototype_decimate
from aurora.transfer_function.kernel_dataset import KernelDataset
from mth5.utils.exceptions import MTH5Error
from mth5.utils.helpers import initialize_mth5
from mt_metadata.transfer_functions.core import TF
from mt_metadata.transfer_functions.processing.aurora import Processing


class TransferFunctionKernel(object):
    def __init__(self, dataset=None, config=None):
        """

        Parameters
        ----------
        dataset: aurora.transfer_function.kernel_dataset.KernelDataset
        config: aurora.config.metadata.processing.Processing
        """
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

    def initialize_mth5s(self, mode="r"):
        """
        returns a dict of open mth5 objects, keyed by station_id

        A future version of this for multiple station processing may need nested dict with [survey_id][station_id]

        Returns
        -------
        mth5_objs : dict
            Keyed by station_ids.
            local station id : mth5.mth5.MTH5
            remote station id: mth5.mth5.MTH5
        """

        local_mth5_obj = initialize_mth5(
            self.config.stations.local.mth5_path, mode=mode
        )
        if self.config.stations.remote:
            remote_path = self.config.stations.remote[0].mth5_path
            remote_mth5_obj = initialize_mth5(remote_path, mode="r")
        else:
            remote_mth5_obj = None

        mth5_objs = {self.config.stations.local.id: local_mth5_obj}
        if self.config.stations.remote:
            mth5_objs[self.config.stations.remote[0].id] = remote_mth5_obj
        self._mth5_objs = mth5_objs
        return

    def update_dataset_df(self, i_dec_level):
        """
        This function has two different modes.  The first mode initializes values in the
        array, and could be placed into TFKDataset.initialize_time_series_data()
        The second mode, decimates. The function is kept in pipelines becasue it calls
        time series operations.

        Notes:
        1. When assigning xarrays to dataframe cells, df dislikes xr.Dataset,
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
            # ANY MERGING OF RUNS IN TIME DOMAIN WOULD GO HERE

            # Assign additional columns to dataset_df, populate with mth5_objs and xr_ts
            self.dataset.initialize_dataframe_for_processing(self.mth5_objs)

            # APPLY TIMING CORRECTIONS HERE
        else:
            print(f"DECIMATION LEVEL {i_dec_level}")

            for i, row in self.dataset_df.iterrows():
                if not self.is_valid_dataset(row, i_dec_level):
                    continue
                if row.fc:
                    row_ssr_str = f"survey: {row.survey}, station_id: {row.station_id}, run_id: {row.run_id}"
                    msg = f"FC already exists for {row_ssr_str} -- skipping decimation"
                    print(msg)
                    continue
                run_xrds = row["run_dataarray"].to_dataset("channel")
                decimation = self.config.decimations[i_dec_level].decimation
                decimated_xrds = prototype_decimate(decimation, run_xrds)
                self.dataset_df["run_dataarray"].at[i] = decimated_xrds.to_array(
                    "channel"
                )  # See Note 1 above

        print("DATASET DF UPDATED")
        return

    def apply_clock_zero(self, dec_level_config):
        """get clock-zero from data if needed"""
        if dec_level_config.window.clock_zero_type == "data start":
            dec_level_config.window.clock_zero = str(self.dataset_df.start.min())
        return dec_level_config

    def check_if_fc_levels_already_exist(self):
        """
        Iterate over the processing summary_df, grouping by unique "Survey-Station-Run"s.
        (Could also iterate over kernel_dataset.dataframe, to get the groupby).

        If all FC Levels for a given station-run are already built, mark the RunSummary with a True in
        the "fc" column.  Otherwise its False

        Note 1:  Because decimation is a cascading operation, we avoid the case where some (valid) decimation
        levels exist in the mth5 FC archive and others do not.  The maximum granularity tolerated will be at the
        "station-run level, so for a given run, either all relevant FCs are packed into the h5 or we treat as if none
        of them are.  Sounds harsh, but if you want to add the logic otherwise, feel free.  If one wanted to support
        variations at the decimation-level, an appropriate way to address would be to store teh decimated time series
        in the archive as well (they would simply be runs with different sample rates, and some extra filters).

        Note 2: At this point in the logic, it is established that there are FCs associated with run_id and there are
        at least as many FC decimation levels as we require as per the processing config.  The next step is to
        assert whether it is True that the existing FCs conform to the recipe in the processing config.

        Note #3: Need to update mth5_objs dict so that it is keyed by survey, then station, else might break when
        mixing in data from other surveys (if the stations are named the same.  This can be addressed in
        the initialize_mth5s() method of TFK.  When addresssing the above issue -- consider adding the mth5_obj to
         self.dataset_df instead of keeping the dict around ..., the concern about doing this is that multiple rows
         of the dataset_df may refernece the same h5, and I don't know if updating one row will have unintended
         consequences.

        Note #4: associated_run_sub_df may have multiple rows, even though the run id is unique.
        This could happen for example when you have a long run at the local station, but multiple (say two) shorter runs
        at the reference station.  In that case, the processing summary will have a separate row for the
        intersection of the long run with each of the remote runs. We ignore this for now, selecting only the first
        element of the associated_run_sub_df, under the assumption that FCs have been created for the entire run,
        or not at all.  This assumption can be relaxed in future by using the time_period attribute of the FC layer.
        For now, we proceed with the all-or-none logic.  That is, if a ['survey', 'station_id', 'run_id',] has FCs,
        assume that the FCs are present for the entire run. We assign the "fc" column of dataset_df to have the same
         boolean value for all rows of same  ['survey', 'station_id', 'run_id',] .

        Returns: None
            Modifies self.dataset_df inplace, assigning bools to the "fc" column

        """
        groupby = [
            "survey",
            "station_id",
            "run_id",
        ]
        grouper = self.processing_summary.groupby(groupby)

        for (survey_id, station_id, run_id), df in grouper:
            cond1 = self.dataset_df.survey == survey_id
            cond2 = self.dataset_df.station_id == station_id
            cond3 = self.dataset_df.run_id == run_id
            associated_run_sub_df = self.dataset_df[cond1 & cond2 & cond3]

            if len(associated_run_sub_df) > 1:
                # See Note #4
                print(
                    "Warning -- not all runs will processed as a continuous chunk -- in future may need to loop over runlets to check for FCs"
                )

            dataset_df_indices = np.r_[associated_run_sub_df.index]
            # dataset_df_indices = associated_run_sub_df.index.to_numpy()
            run_row = associated_run_sub_df.iloc[0]
            row_ssr_str = f"survey: {run_row.survey}, station_id: {run_row.station_id}, run_id: {run_row.run_id}"

            # See Note #3 above relating to mixing multiple surveys in a processing scheme
            mth5_obj = self.mth5_objs[station_id]
            survey_obj = mth5_obj.get_survey(survey_id)
            station_obj = survey_obj.stations_group.get_station(station_id)
            if not station_obj.fourier_coefficients_group.groups_list:
                msg = f"Prebuilt Fourier Coefficients not detected for {row_ssr_str} -- will need to build them "
                print(msg)
                self.dataset_df.loc[dataset_df_indices, "fc"] = False
            else:
                print(
                    "Prebuilt Fourier Coefficients detected -- checking if they satisfy processing requirements..."
                )
                # Assume FC Groups are keyed by run_id, check if there is a relevant group
                try:
                    fc_group = station_obj.fourier_coefficients_group.get_fc_group(
                        run_id
                    )
                except MTH5Error:
                    self.dataset_df.loc[dataset_df_indices, "fc"] = False
                    print(
                        f"Run ID {run_id} not found in FC Groups, -- will need to build them "
                    )
                    continue

                if (
                    len(fc_group.groups_list)
                    < self.processing_config.num_decimation_levels
                ):
                    self.dataset_df.loc[dataset_df_indices, "fc"] = False
                    print(
                        f"Not enough FC Groups available for {row_ssr_str} -- will need to build them "
                    )
                    continue

                # Can check time periods here if desired, but unique (survey, station, run) should make this unneeded
                # processing_run = self.processing_config.stations.local.get_run(run_id)
                # for tp in processing_run.time_periods:
                #    assert tp in fc_group time periods

                # See note #2
                fcs_already_there = fc_group.supports_aurora_processing_config(
                    self.processing_config, run_row.remote
                )
                self.dataset_df.loc[dataset_df_indices, "fc"] = fcs_already_there

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
                try:
                    assert (
                        df.dec_level.diff()[1:] == 1
                    ).all()  # dec levels increment by 1
                except AssertionError:
                    print(f"Skipping {group} because decimation levels are messy.")
                    continue
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
        self, min_num_stft_windows=None
    ):
        """
        Refers to issue #182 (and #103, and possibly #196 and #233).
        Determine if there exist (one or more) runs that will yield decimated datasets
        that have too few samples to be passed to the STFT algorithm.

        Strategy for handling this:
        Mark as invlaid any rows of the processing summary that do not yield long
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
        if min_num_stft_windows is None:
            min_stft_window_info = {
                x.decimation.level: x.min_num_stft_windows
                for x in self.processing_config.decimations
            }
            min_stft_window_list = [
                min_stft_window_info[x] for x in self.processing_summary.dec_level
            ]
            min_num_stft_windows = pd.Series(min_stft_window_list)

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
        if len(processing_row) != 1:
            return False
        is_valid = processing_row.valid.iloc[0]
        return is_valid

    @property
    def processing_type(self):
        """
        A description of the processing, will get passed to TF object,
        can be used for Z-file

        Could add a version or a hashtag to this
        Could also check dataset_df
        If remote.all==False append "Single Station"
        """
        processing_type = "Aurora"
        if self.dataset_df.remote.any():
            processing_type = f"{processing_type} Robust Remote Reference"
        else:
            processing_type = f"{processing_type} Robust Single Station"
        if "processing_type" in self.dataset_df.columns:
            processing_type = self.dataset_df["processing_type"].iloc[0]

        return processing_type

    def export_tf_collection(self, tf_collection):
        """
        Assign transfer_function, residual_covariance, inverse_signal_power, station, survey

        Parameters
        ----------
        tf_collection: aurora.transfer_function.transfer_function_collection.TransferFunctionCollection
            Contains TF estimates, covariance, and signal power values

        Returns
        -------
        tf_cls: mt_metadata.transfer_functions.core.TF
            Transfer function container
        """

        def make_decimation_dict_for_tf(tf_collection, processing_config):
            """
            Decimation dict is used by mt_metadata's TF class when it is writng z-files.
            If no z-files will be written this is not needed

            sample element of decimation_dict:
            '1514.70134': {'level': 4, 'bands': (5, 6), 'npts': 386, 'df': 0.015625}}

            Note #1:  The line that does
            period_value["npts"] = tf_collection.tf_dict[i_dec].num_segments.data[0, i_band]
            doesn't feel very robust ... it would be better to explicitly select based on the num_segments
            xarray location where period==fb.center_period.  This is a placeholder for now.  In future,
            the TTFZ class is likely to be deprecated in favour of directly packing mt_metadata TFs during processing.

            Parameters
            ----------
            tfc

            Returns
            -------

            """
            from mt_metadata.transfer_functions.io.zfiles.zmm import PERIOD_FORMAT

            decimation_dict = {}

            for i_dec, dec_level_cfg in enumerate(processing_config.decimations):
                for i_band, band in enumerate(dec_level_cfg.bands):
                    period_key = f"{band.center_period:{PERIOD_FORMAT}}"
                    period_value = {}
                    period_value["level"] = i_dec + 1  # +1 to match EMTF standard
                    period_value["bands"] = tuple(band.harmonic_indices[np.r_[0, -1]])
                    period_value["sample_rate"] = dec_level_cfg.sample_rate_decimation
                    try:
                        period_value["npts"] = tf_collection.tf_dict[
                            i_dec
                        ].num_segments.data[0, i_band]
                    except KeyError:
                        print("Possibly invalid decimation level")
                        period_value["npts"] = 0
                    decimation_dict[period_key] = period_value

            return decimation_dict

        channel_nomenclature = self.config.channel_nomenclature
        channel_nomenclature_dict = channel_nomenclature.to_dict()[
            "channel_nomenclature"
        ]
        merged_tf_dict = tf_collection.get_merged_dict(channel_nomenclature)
        decimation_dict = make_decimation_dict_for_tf(
            tf_collection, self.processing_config
        )

        tf_cls = TF(
            channel_nomenclature=channel_nomenclature_dict,
            decimation_dict=decimation_dict,
        )
        renamer_dict = {"output_channel": "output", "input_channel": "input"}
        tmp = merged_tf_dict["tf"].rename(renamer_dict)
        tf_cls.transfer_function = tmp

        isp = merged_tf_dict["cov_ss_inv"]
        renamer_dict = {"input_channel_1": "input", "input_channel_2": "output"}
        isp = isp.rename(renamer_dict)
        tf_cls.inverse_signal_power = isp

        res_cov = merged_tf_dict["cov_nn"]
        renamer_dict = {"output_channel_1": "input", "output_channel_2": "output"}
        res_cov = res_cov.rename(renamer_dict)
        tf_cls.residual_covariance = res_cov

        # Set key as first el't of dict, nor currently supporting mixed surveys in TF
        tf_cls.survey_metadata = self.dataset.local_survey_metadata
        tf_cls.station_metadata.transfer_function.processing_type = self.processing_type
        return tf_cls

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
