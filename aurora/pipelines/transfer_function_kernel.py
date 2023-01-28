import numpy as np
import pandas as pd

from aurora.config.metadata.processing import Processing
from aurora.pipelines.helpers import initialize_config
from aurora.transfer_function.kernel_dataset import KernelDataset


class TransferFunctionKernel(object):
    def __init__(self, dataset=None, config=None):
        processing_config = initialize_config(config)
        self._config = processing_config
        self._dataset = dataset

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
            assert (df.dec_level.diff()[1:] == 1).all()  # dec levels increment by 1
            assert df.dec_factor.iloc[0] == 1
            assert df.dec_level.iloc[0] == 0
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
        Essentially we want to make sure that the data don't get so decimated that
        the time-series cannot be windowed.

        Determine if there exist (one or more) runs that will yield decimated datasets
        that have too few samples to be passed to the STFT algorithm.

        Strategy for handling this:
        We will mark as invlaid any rows of the processing summary that do not yield
        enough time series.

        WCGW: If the runs are "chunked" we could encounter a situation where the
        chunk fails to yield a deep decimation level, yet the level could be produced
        if the chunk size were larger.
        In general, handling this seems a bit complicated.  We will ignore this for
        now and assume that the user will select a chunk size that is appropriate to
        the decimation scheme, i.e. use large chunks for deep decimations.



        If there are any such runs, we should also look at strategies for handling this.

        We may hove some long runs that don't encounter the issue at all, and some
        runs that do.
        If we have this mixed situation, the best solution (since the user did ask
        for the decimated frequency bands) is to return them the decimated runs when
        available, and skip the ones that are not

        This makes it seem like a general solution would be to return a sort of log
        that tells the user about each run and decimation level ... how many
        STFT-windows it yielded at each decimation level.

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
        Given a row form the RunSummary, will this decimation level yield a valid
        dataset?

        Parameters
        ----------
        row: a row of the self._dataset_df (corresponding to a run that will be
        processed
        i_dec: integer - refers to decimation level

        Returns
        -------

        """
        # Query processing on survey, station, run, decimation
        cond1 = self.processing_summary.survey == row.survey
        cond2 = self.processing_summary.station_id == row.station_id
        cond3 = self.processing_summary.run_id == row.run_id
        cond4 = self.processing_summary.dec_level == i_dec
        cond = cond1 & cond2 & cond3 & cond4
        processing_row = self.processing_summary[cond]
        assert len(processing_row) == 1
        return processing_row.valid.iloc[0]
