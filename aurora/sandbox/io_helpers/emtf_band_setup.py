from loguru import logger

import numpy as np
import pandas as pd


class EMTFBandSetupFile:
    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        sample_rate: float
            this is the sampling rate of the un-decimated data,
            aka decimation level 1 in EMTF (and probably zero in the
            aurora nomenclature)
        kwargs
        """
        self.filepath = kwargs.get("filepath", None)
        self.df = None
        self._num_bands = None
        self.sample_rate = kwargs.get("sample_rate", None)
        if self.filepath:
            self.load()

    @property
    def num_bands(self):
        return self._num_bands

    @property
    def num_decimation_levels(self):
        num_dec = len(self.decimation_levels)
        return num_dec

    @property
    def decimation_levels(self):
        dec_levels = self.df.decimation_level.unique()
        dec_levels.sort()
        return dec_levels

    def load(self, filepath=None):
        if filepath is None:
            filepath = self.filepath
        msg = f"loading band setup file {filepath}"
        logger.debug(msg)
        with open(str(filepath), "r") as f:
            num_bands = f.readline()
        self._num_bands = int(num_bands)
        f.close()
        df = pd.read_csv(
            filepath,
            skiprows=1,
            sep="\s+",
            names=["decimation_level", "lower_bound_index", "upper_bound_index"],
        )
        if len(df) != self.num_bands:
            msg = f"unexpected number of bounds read in from {filepath}"
            logger.exception(msg)
            raise Exception(msg)
        self.df = df

    def get_decimation_level(self, decimation_level, order="ascending_frequency"):
        if self.df is None:
            self.load()
        decimation_level_df = self.df[self.df["decimation_level"] == decimation_level]
        if order == "ascending_frequency":
            decimation_level_df = decimation_level_df.sort_values(
                by="lower_bound_index"
            )

        return decimation_level_df

    def compute_band_edges(self, decimation_factors, num_samples_window):
        """
        Adds columns to df defining some useful numbers
        sample_rate
        Returns
        -------

        """
        band_edges = {}
        lower_edges = pd.Series(index=self.df.index, dtype="float64")
        upper_edges = pd.Series(index=self.df.index, dtype="float64")
        if not self.sample_rate:
            print("cannot define frequencies of sample rate undefined")
            raise Exception
        if len(decimation_factors) != self.num_decimation_levels:
            print("Number of decimation_factors must equal number of decimation lvels")
            print(
                f"There are {len(decimation_factors)} decimation_factors but "
                f"{self.num_decimation_levels} decimation lvels"
            )
        # add a similar check for num_samples_window
        for i_dec, decimation_level in enumerate(self.decimation_levels):
            indices = self.df.decimation_level == decimation_level

            # emtf decimation levels start at 1, but i_dec indexes them from 0
            downsample_factor = 1.0 * np.prod(decimation_factors[0 : i_dec + 1])
            sample_rate = self.sample_rate / downsample_factor
            delta_f = sample_rate / num_samples_window[i_dec]
            half_df = delta_f / 2.0
            lower_edges[indices] = self.df["lower_bound_index"] * delta_f - half_df
            upper_edges[indices] = self.df["upper_bound_index"] * delta_f + half_df
            band_edges[i_dec] = np.vstack(
                (lower_edges[indices].values, upper_edges[indices].values)
            ).T

        self.df["lower_edge"] = lower_edges
        self.df["upper_edge"] = upper_edges
        return band_edges

    def to_band_averaging_scheme(self):
        """
        probably better to give band averaging scheme a "from emtf"
        method
        Returns
        -------

        """
        pass
