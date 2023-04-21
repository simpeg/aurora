# -*- coding: utf-8 -*-
"""
Extend the Processing class with some aurora-specific methods
"""
# =============================================================================
# Imports
# =============================================================================
import pandas as pd

from aurora.time_series.windowing_scheme import window_scheme_from_decimation
from mt_metadata.transfer_functions.processing.aurora.processing import Processing
from mth5.utils.helpers import initialize_mth5


class Processing(Processing):
    def __init__(self, **kwargs):

        # super().__init__(attr_dict=attr_dict, **kwargs)
        super().__init__(**kwargs)

    def initialize_mth5s(self):
        """

        Returns
        -------
        mth5_objs : dict
            Keyed by station_ids.
            local station id : mth5.mth5.MTH5
            remote station id: mth5.mth5.MTH5
        """
        local_mth5_obj = initialize_mth5(self.stations.local.mth5_path, mode="r")
        if self.stations.remote:
            remote_path = self.stations.remote[0].mth5_path
            remote_mth5_obj = initialize_mth5(remote_path, mode="r")
        else:
            remote_mth5_obj = None

        mth5_objs = {self.stations.local.id: local_mth5_obj}
        if self.stations.remote:
            mth5_objs[self.stations.remote[0].id] = remote_mth5_obj

        return mth5_objs

    def window_scheme(self, as_type="df"):
        """
        Make a dataframe of processing parameters one row per decimation level.
        Returns
        -------

        """
        window_schemes = [window_scheme_from_decimation(x) for x in self.decimations]
        data_dict = {}
        data_dict["sample_rate"] = [x.sample_rate for x in window_schemes]
        data_dict["window_duration"] = [x.window_duration for x in window_schemes]
        data_dict["num_samples_window"] = [x.num_samples_window for x in window_schemes]
        data_dict["num_samples_overlap"] = [
            x.num_samples_overlap for x in window_schemes
        ]
        data_dict["num_samples_advance"] = [
            x.num_samples_advance for x in window_schemes
        ]
        if as_type == "dict":
            return data_dict
        elif as_type == "df":
            df = pd.DataFrame(data=data_dict)
            return df
        else:
            print(f"unexpected rtype for window_scheme {as_type}")
            raise TypeError

    def decimation_info(self):
        decimation_ids = [x.decimation.level for x in self.decimations]
        decimation_factors = [x.decimation.factor for x in self.decimations]
        decimation_info = dict(zip(decimation_ids, decimation_factors))
        return decimation_info

    def save_as_json(self, filename=None, nested=True, required=False):
        if filename is None:
            filename = self.json_fn()
        json_str = self.to_json(nested=nested, required=required)
        with open(filename, "w") as f:
            f.write(json_str)

    def make_tf_header(self, dec_level_id):
        """

        Parameters
        ----------
        dec_level_id: int
            This may tolerate strings in the future, but keep as int for now

        Returns
        -------
        tfh: mt_metadata.transfer_functions.processing.aurora.transfer_function_header.TransferFunctionHeader
        """
        from aurora.transfer_function.transfer_function_header import (
            TransferFunctionHeader,
        )

        tfh = TransferFunctionHeader(
            processing_scheme=self.decimations[dec_level_id].estimator.engine,
            local_station=self.stations.local,
            reference_station=self.stations.remote,
            input_channels=self.decimations[dec_level_id].input_channels,
            output_channels=self.decimations[dec_level_id].output_channels,
            reference_channels=self.decimations[dec_level_id].reference_channels,
            decimation_level_id=dec_level_id,
        )

        return tfh

    def make_tf_level(self, dec_level_id):
        """
        Initialize container for a single decimation level -- "flat" transfer function.

        Parameters
        ----------
        dec_level_id: int
            This may tolerate strings in the future, but keep as int for now

        Returns
        -------
        tf_obj: aurora.transfer_function.TTFZ.TTFZ
        """
        # from aurora.transfer_function.base import TransferFunction
        from aurora.transfer_function.TTFZ import TTFZ

        tf_header = self.make_tf_header(dec_level_id)
        tf_obj = TTFZ(tf_header, self.decimations[dec_level_id].frequency_bands_obj())

        return tf_obj
