import numpy as np
import pandas as pd
import psutil


from aurora.pipelines.helpers import initialize_config
from aurora.pipelines.time_series_helpers import prototype_decimate
from aurora.transfer_function.kernel_dataset import KernelDataset
from mth5.utils.exceptions import MTH5Error
from mth5.utils.helpers import initialize_mth5

from mt_metadata.transfer_functions.processing.aurora import Processing

def check_if_fcdecimation_group_has_fcs(fcdec_group, decimation_level, remote):
    """
    This could be made a method of mth5.groups.fourier_coefficients.FCDecimationGroup

    The following are not required to check:
    "decimation_factor",
    "decimation_level",

    ToDo: Add to github checklist:
    - AAF should be
    1. made None in dec_level 0 of processing config and
    2. set to default in the FC Config
    - "min_num_stft_windows" is not in ProcessingConfig, should be default==2


    Parameters
    ----------
    fcdec_group
    decimation_level

    Returns
    -------

    """
    # "channels_estimated"
    if remote:
        required_channels = decimation_level.reference_channels
    else:
        required_channels = decimation_level.input_channels + decimation_level.output_channels
    try:
        assert set(fcdec_group.metadata.channels_estimated) == set(required_channels)
    except AssertionError:
        msg = f"required_channels for processing {required_channels} not available in fc channels estimated {fcdec_group.metadata.channels_estimated}"
        print(msg)
        return False

    # anti_alias_filter (AAF)
    try:
        assert fcdec_group.metadata.anti_alias_filter == decimation_level.anti_alias_filter
    except AssertionError:
        cond1 = decimation_level.anti_alias_filter == "default"
        cond2 = fcdec_group.metadata.anti_alias_filter is None
        if cond1 & cond2:
            pass
        else:
            msg = "Antialias Filters Not Compatible -- need to add handling for "
            msg =f"{msg} fcdec {fcdec_group.metadata.anti_alias_filter} and processing config:{decimation_level.anti_alias_filter}"
            raise NotImplementedError(msg)

    # Sample rate
    try:
        assert fcdec_group.metadata.sample_rate == decimation_level.sample_rate_decimation
    except AssertionError:
        msg = f"Sample rates do not agree: fc {fcdec_group.metadata.sample_rate} vs pc {decimation_level.sample_rate_decimation}"
        print(msg)
        return False

    # Method (fft, wavelet, etc.)
    try:
        assert fcdec_group.metadata.method == decimation_level.method
    except AssertionError:
        msg = f"Transform methods do not agree"
        print(msg)
        return False

    # prewhitening_type
    try:
        assert fcdec_group.metadata.prewhitening_type == decimation_level.prewhitening_type
    except AssertionError:
        msg = f"prewhitening_type does not agree"
        print(msg)
        return False

    # recoloring
    try:
        assert fcdec_group.metadata.recoloring == decimation_level.recoloring
    except AssertionError:
        msg = f"recoloring does not agree"
        print(msg)
        return False

    # pre_fft_detrend_type
    try:
        assert fcdec_group.metadata.pre_fft_detrend_type == decimation_level.pre_fft_detrend_type
    except AssertionError:
        msg = f"pre_fft_detrend_type does not agree"
        print(msg)
        return False

    # min_num_stft_windows
    try:
        assert fcdec_group.metadata.min_num_stft_windows == decimation_level.min_num_stft_windows
    except AssertionError:
        msg = f"min_num_stft_windows do not agree {fcdec_group.metadata.min_num_stft_windows} vs {decimation_level.min_num_stft_windows}"
        print(msg)
        return False

    # window
    # decimation_level.window.type = "boxcar"; print("REMOVE DEBUG!!!!")
    try:
        assert fcdec_group.metadata.window == decimation_level.window
    except AssertionError:
        msg = "window does not agree:\n"
        msg = f"{msg} FC Group: {fcdec_group.metadata.window}\n"
        msg = f"{msg} Processing Config  {decimation_level.window}"
        print(msg)
        return False


    # harmonic_indices
    # Since agreement on sample_rate and window length is already established, we can use integer indices of FCs
    # rather than work with frequencies explcitly.
    # note that if harmonic_indices is -1, it means keep all so we can skip this check.
    if -1 in fcdec_group.metadata.harmonic_indices:
        pass
    else:
        harmonic_indices_requested = decimation_level.harmonic_indices()
        print(f"determined that {harmonic_indices_requested} are the requested indices")
        fcdec_group_set = set(fcdec_group.metadata.harmonic_indices)
        processing_set = set(harmonic_indices_requested)
        if processing_set.issubset(fcdec_group_set):
            pass
        else:
            msg = f"Processing FC indices {processing_set} is not contained in FC indices {fcdec_group_set}"
            print(msg)
            return False
    #failed no checks if you get here
    return True


def check_if_fcgroup_supports_processing_config(fc_group, processing_config, remote):
    """
    This could be made a method of mth5.groups.fourier_coefficients.FCGroup

    As per Note #1 in check_if_fc_levels_already_exist(), this is an all-or-nothing check:
    Either every (valid) decimation level in the processing config is available or we will build all FCs.

    Currently using a nested for loop, but this can be made a bit quicker by checking if sample_rates
    are in agreement (if they aren't we don't need to check any other parameters)
    Also, Once an fc_decimation_id is found to match a dec_level, we don't need to keep checking that fc_decimation_id

    Note #1: The idea is to
    Parameters
    ----------
    fc_group
    processing_config

    Returns
    -------

    """
    fc_decimation_ids_to_check = fc_group.groups_list
    levels_present = np.full(processing_config.num_decimation_levels, False)
    for i, dec_level in enumerate(processing_config.decimations):
        # See Note #1
        #print(f"{i}")
        #print(f"{dec_level}")

        # All or nothing condition
        if (i > 0):
            if not levels_present[i - 1]:
                return False

        # iterate over fc_group decimations
        # This can be done smarter ... once an fc_decimation_id is found to
        for fc_decimation_id in fc_decimation_ids_to_check:
            fc_dec_group = fc_group.get_decimation_level(fc_decimation_id)
            levels_present[i] = check_if_fcdecimation_group_has_fcs(fc_dec_group, dec_level, remote)

            if levels_present[i]:
                fc_decimation_ids_to_check.remove(fc_decimation_id) #no need to look at this one again
                break #break inner loop


    return levels_present.all()




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

        local_mth5_obj = initialize_mth5(self.config.stations.local.mth5_path, mode=mode)
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

    def update_dataset_df(self,i_dec_level, fc_existence_info=None):
        """
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
            # ANY MERGING OF RUNS IN TIME DOMAIN WOULD GO HERE

            # Assign additional columns to dataset_df, populate with mth5_objs and xr_ts
            self.dataset.initialize_dataframe_for_processing(self.mth5_objs)

            # APPLY TIMING CORRECTIONS HERE
        else:
            print(f"DECIMATION LEVEL {i_dec_level}")
            print("UNDER CONSTRUCTION 20230902 -- Need to skip if FC TRUE")
            # See Note 1 top of module
            # See Note 2 top of module
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
                self.dataset_df["run_dataarray"].at[i] = decimated_xrds.to_array("channel")

        print("DATASET DF UPDATED")
        return

    def check_if_fc_levels_already_exist(self):
        """
        Iterate over the processing summary_df, grouping by unique "Survey-Station-Run"s.
        (Could also iterate over kernel_dataset.dataframe, to get the groupby).

        If all FC Levels for a given station-run are already built, mark the RunSummary with a True in
        the "fc" column.

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

        Returns: None
            Modifies self.dataset_df inplace, assigning bool to the "fc" column

        """
        groupby = ['survey', 'station_id', 'run_id',]
        grouper = self.processing_summary.groupby(groupby)
        # print(len(grouper))
        for (survey_id, station_id, run_id), df in grouper:
            cond1 = self.dataset_df.survey == survey_id
            cond2 = self.dataset_df.station_id == station_id
            cond3 = self.dataset_df.run_id == run_id
            associated_run_sub_df = self.dataset_df[cond1 & cond2 & cond3]
            assert len(associated_run_sub_df) == 1 # should be unique
            dataset_df_index = associated_run_sub_df.index[0]
            run_row = associated_run_sub_df.iloc[0]

            # See Note #3 above
            mth5_obj = self.mth5_objs[station_id]
            survey_obj = mth5_obj.get_survey(survey_id)
            station_obj = survey_obj.stations_group.get_station(station_id)
            if not station_obj.fourier_coefficients_group.groups_list:
                print("Prebuilt Fourier Coefficients not detected -- will need to build them ")
                self.dataset_df["fc"].iat[dataset_df_index] = False
            else:
                print("Prebuilt Fourier Coefficients detected -- checking if they satisfy processing requirements...")
                # Assume FC Groups are keyed by run_id, check if there is a relevant group
                try:
                    fc_group = station_obj.fourier_coefficients_group.get_fc_group(run_id)
                except MTH5Error:
                    self.dataset_df["fc"].iat[dataset_df_index] = False
                    print(f"Run ID {run_id} not found in FC Groups, -- will need to build them ")
                    continue

                if len(fc_group.groups_list) < self.processing_config.num_decimation_levels:
                    self.dataset_df["fc"].iat[dataset_df_index] = False
                    print(f"Not enough FC Groups available -- will need to build them ")
                    continue

                # Can check time periods here if desired, but unique (survey, station, run) should make this unneeded
                # processing_run = self.processing_config.stations.local.get_run("001")
                # for tp in processing_run.time_periods:
                #    assert tp in fc_group time periods


                # See note #2
                fcs_already_there = check_if_fcgroup_supports_processing_config(fc_group,
                                                                                self.processing_config,
                                                                                run_row.remote)
                self.dataset_df["fc"].iat[dataset_df_index] = fcs_already_there

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
            min_stft_window_info = {x.decimation.level: x.min_num_stft_windows for x in self.processing_config.decimations}
            min_stft_window_list = [min_stft_window_info[x] for x in self.processing_summary.dec_level]
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
