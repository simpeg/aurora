# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 14:15:20 2022

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
import numpy as np
import pandas as pd

from mt_metadata.base.helpers import write_lines
from mt_metadata.base import get_schema, Base
from .standards import SCHEMA_FN_PATHS

from . import Window, Decimation, Band, Regression, Estimator

from aurora.time_series.frequency_domain_helpers import get_fft_harmonics

# =============================================================================
attr_dict = get_schema("decimation_level", SCHEMA_FN_PATHS)
attr_dict.add_dict(get_schema("decimation", SCHEMA_FN_PATHS), "decimation")
attr_dict.add_dict(get_schema("window", SCHEMA_FN_PATHS), "window")
attr_dict.add_dict(get_schema("regression", SCHEMA_FN_PATHS), "regression")
attr_dict.add_dict(get_schema("estimator", SCHEMA_FN_PATHS), "estimator")


# =============================================================================
class DecimationLevel(Base):
    __doc__ = write_lines(attr_dict)

    def __init__(self, **kwargs):

        self.window = Window()
        self.decimation = Decimation()
        self.regression = Regression()
        self.estimator = Estimator()

        self._bands = []

        super().__init__(attr_dict=attr_dict, **kwargs)

    @property
    def bands(self):
        """
        get bands, something weird is going on with appending.

        """
        return_list = []
        for band in self._bands:
            if isinstance(band, dict):
                b = Band()
                b.from_dict(band)
            elif isinstance(band, Band):
                b = band
            return_list.append(b)
        return return_list

    @bands.setter
    def bands(self, value):
        """
        Set bands make sure they are a band object

        :param value: list of bands
        :type value: list, Band

        """

        if isinstance(value, Band):
            self._bands = [value]

        elif isinstance(value, list):
            self._bands = []
            for obj in value:
                if not isinstance(obj, (Band, dict)):
                    raise TypeError(f"List entry must be a Band object not {type(obj)}")
                if isinstance(obj, dict):
                    band = Band()
                    band.from_dict(obj)

                else:
                    band = obj

                self._bands.append(band)
        else:
            raise TypeError(f"Not sure what to do with {type(value)}")

    def add_band(self, band):
        """
        add a band
        """

        if not isinstance(band, (Band, dict)):
            raise TypeError(f"List entry must be a Band object not {type(band)}")
        if isinstance(band, dict):
            obj = Band()
            obj.from_dict(band)

        else:
            obj = band

        self._bands.append(obj)

    @property
    def lower_bounds(self):
        """
        get lower bounds index values into an array.
        """

        return np.array(sorted([band.index_min for band in self.bands]))

    @property
    def upper_bounds(self):
        """
        get upper bounds index values into an array.
        """

        return np.array(sorted([band.index_max for band in self.bands]))

    @property
    def bands_dataframe(self):
        """
        This is just a utility function that transforms a list of bands into a dataframe
        ToDo: Consider make this a method of Bands()

        Note: The decimation_level here is +1 to agree with EMTF convention.
        Not clear this is really necessary

        ToDo: Consider adding columns lower_edge, upper_edge to df

        Parameters
        ----------
        band_list: list
            obtained from aurora.config.metadata.decimation_level.DecimationLevel.bands

        Returns
        -------
        out_df: pd.Dataframe
            Same format as that generated by EMTFBandSetupFile.get_decimation_level()
        """
        from aurora.time_series.frequency_band_helpers import df_from_bands

        bands_df = df_from_bands(self.bands)
        return bands_df

    @property
    def band_edges(self):
        bands_df = self.bands_dataframe
        band_edges = np.vstack(
            (bands_df.frequency_min.values, bands_df.frequency_max.values)
        ).T
        return band_edges

    def frequency_bands_obj(self):
        """
        Gets a FrequencyBands object that is used as input to processing.
        This used to be needed because I only had

        ToDO: consider adding .to_frequnecy_bands() method directly to self.bands
        Returns
        -------

        """
        from aurora.time_series.frequency_band import FrequencyBands

        frequency_bands = FrequencyBands(band_edges=self.band_edges)
        return frequency_bands

    @property
    def windowing_scheme(self):
        from aurora.time_series.windowing_scheme import WindowingScheme

        windowing_scheme = WindowingScheme(
            taper_family=self.window.type,
            num_samples_window=self.window.num_samples,
            num_samples_overlap=self.window.overlap,
            taper_additional_args=self.window.additional_args,
            sample_rate=self.decimation.sample_rate,
        )
        return windowing_scheme

    @property
    def fft_frequecies(self):
        freqs = get_fft_harmonics(self.window.num_samples, self.decimation.sample_rate)
        return freqs

    # def to_stft_config_dict(self):
    #     """
    #     taper_family
    #     num_samples_window
    #     num_samples_overlap
    #     taper_additional_args
    #     sample_rate,
    #     prewhitening_type
    #     extra_pre_fft_detrend_type
    #     Returns
    #     -------
    #     """
    #     output = {}
    #     print(self.window.type)
    #     output["taper_family"] = self.window.type
    #     output["num_samples_window"] = self.window.num_samples
    #     output["num_samples_overlap"] = self.window.overlap
    #     output["taper_additional_args"] = self.window.additional_args
    #     output["sample_rate"] = self.decimation.sample_rate
    #     output["prewhitening_type"] = self.prewhitening_type
    #     output["extra_pre_fft_detrend_type"] = self.extra_pre_fft_detrend_type
    #     return output
