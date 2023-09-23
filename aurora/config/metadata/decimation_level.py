# -*- coding: utf-8 -*-
"""
Extend DecimationLevel class with some aurora-specific methods
"""
# =============================================================================
# Imports
# =============================================================================
from mt_metadata.transfer_functions.processing.aurora import DecimationLevel


class DecimationLevel(DecimationLevel):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    @property
    def windowing_scheme(self):
        from aurora.time_series.windowing_scheme import WindowingScheme

        windowing_scheme = WindowingScheme(
            taper_family=self.window.type,
            num_samples_window=self.window.num_samples,
            num_samples_overlap=self.window.overlap,
            taper_additional_args=self.window.additional_args,
            sample_rate=self.sample_rate_decimation,
        )
        return windowing_scheme

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
    #     output["sample_rate"] = self.sample_rate_decimation
    #     output["prewhitening_type"] = self.prewhitening_type
    #     output["extra_pre_fft_detrend_type"] = self.extra_pre_fft_detrend_type
    #     return output
