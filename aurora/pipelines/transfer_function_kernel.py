import numpy as np
import pandas as pd

from aurora.config.metadata.processing import Processing
from aurora.transfer_function.kernel_dataset import KernelDataset


class TransferFunctionKernel(object):
    def __init__(self, dataset=None, config=None):
        self._config = config
        self._dataset = dataset

    @property
    def kernel_dataset(self):
        return self._dataset

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
            assert (
                df.dec_level.diff()[1:] == 1
            ).all()  # dec levels are incrementing by 1
            assert df.dec_factor.iloc[0] == 1
            assert df.dec_level.iloc[0] == 0
            # df.sample_rate /= np.cumprod(df.dec_factor)  # better to take from processing config
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
                try:
                    num_windows[i] = ws.available_number_of_windows(row.num_samples)
                except:
                    pass  # probably not enough data
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


        If there are any such runs, we should also look at strategies for handling this.

        Cases:
        1. There is only one run
        If the decimation scheme is fine, ok, if not, suggest removing last N levels.

        2. If there is more than one run:
        For each run we can treat as above.  Note that in this case, we may hove some
        long runs that don't encounter the issue at all, and some runs that do.
        If we have this mixed situation, the best solution (since the user did ask
        for the decimated frequency bands) is to return them the decimated runs when
        available, and skip the ones that are not ... This is quite general but
        requires a different approach to the one I was considering ...

        This makes it seem like a general solution would be to return a sort of log
        that tells the user about each run and decimation level ... how many
        STFT-windows it yielded at each decimation level.

        This conjures the notion of (run, decimation_level) pairs
        With this we can handle unacceptable runs at a fixed case 2 by adding some
        degenerate value to the stft concatenation, but then we
        only nee

        This can be approached a few ways ...

        One way is to start at
        -------

        """
        self.processing_summary["valid"] = (
            self.processing_summary.num_stft_windows >= min_num_stft_windows
        )

    def validate(self):
        self.validate_decimation_scheme_and_dataset_compatability()
