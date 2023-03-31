# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 14:15:20 2022

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
import pandas as pd
from deprecated import deprecated
import numpy as np
from pathlib import Path

from aurora.config import BANDS_DEFAULT_FILE
from aurora.time_series.frequency_band import FrequencyBand
from mt_metadata.base.helpers import write_lines
from mt_metadata.base import get_schema, Base
from mth5.utils.helpers import initialize_mth5

from .standards import SCHEMA_FN_PATHS
from . import DecimationLevel, Stations, Band, ChannelNomenclature


# =============================================================================
attr_dict = get_schema("processing", SCHEMA_FN_PATHS)
attr_dict.add_dict(Stations()._attr_dict, "stations")
attr_dict.add_dict(ChannelNomenclature()._attr_dict, "channel_nomenclature")


# =============================================================================
class Processing(Base):
    __doc__ = write_lines(attr_dict)

    def __init__(self, **kwargs):

        self.stations = Stations()
        self._decimations = []
        self.channel_nomenclature = ChannelNomenclature()

        super().__init__(attr_dict=attr_dict, **kwargs)

    @property
    def decimations(self):
        return_list = []
        for item in self._decimations:
            if isinstance(item, dict):
                level = DecimationLevel()
                level.from_dict(item)
            elif isinstance(item, DecimationLevel):
                level = item
            return_list.append(level)

        return return_list

    @decimations.setter
    def decimations(self, value):
        """
        dictionary of decimations levels

        :param value: dict of decimation levels
        :type value: dict

        """

        if isinstance(value, DecimationLevel):
            self._decimations.append(value)

        elif isinstance(value, dict):
            self._decimations = []
            for key, obj in value.items():
                if not isinstance(obj, DecimationLevel):
                    raise TypeError(
                        f"List entry must be a DecimationLevel object not {type(obj)}"
                    )
                else:
                    self._decimations.append(obj)

        elif isinstance(value, list):
            self._decimations = []
            for obj in value:
                if isinstance(value, DecimationLevel):
                    self._decimations.append(obj)

                elif isinstance(obj, dict):
                    level = DecimationLevel()
                    level.from_dict(obj)
                    self._decimations.append(level)
                else:
                    raise TypeError(
                        f"List entry must be a DecimationLevel or dict object not {type(obj)}"
                    )

        else:
            raise TypeError(f"Not sure what to do with {type(value)}")

    @property
    def decimations_dict(self):
        """
        need to have a dictionary, but it can't be an attribute cause that
        gets confusing when reading in a json file

        :return: DESCRIPTION
        :rtype: TYPE

        """
        return dict([(d.decimation.level, d) for d in self.decimations])

    def get_decimation_level(self, level):
        """
        Get a decimation level for easy access

        :param level: DESCRIPTION
        :type level: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        try:
            decimation = self.decimations_dict[level]

        except KeyError:
            raise KeyError(f"Could not find {level} in decimations.")

        if isinstance(decimation, dict):
            decimation_level = DecimationLevel()
            decimation_level.from_dict(decimation)
            return decimation_level

        return decimation

    def add_decimation_level(self, decimation_level):
        """
        add a decimation level
        """

        if not isinstance(decimation_level, (DecimationLevel, dict)):
            raise TypeError(
                f"List entry must be a DecimationLevel object not {type(decimation_level)}"
            )
        if isinstance(decimation_level, dict):
            obj = DecimationLevel()
            obj.from_dict(decimation_level)

        else:
            obj = decimation_level

        self._decimations.append(obj)

    @property
    def band_edges_dict(self):
        band_edges_dict = {}
        for i_dec, decimation in enumerate(self.decimations):
            band_edges_dict[i_dec] = decimation.band_edges
        return band_edges_dict

    def assign_bands(
        self, band_edges_dict, sample_rate, decimation_factors, num_samples_window
    ):
        """

        Warning: This does not actually tell us how many samples we are decimating down
        at each level.  That is assumed to be 4 but we need a way to bookkeep this in general

        Parameters
        ----------
        band_edges: dict
            keys are integers, starting with 0, values are arrays of edges

        """
        num_decimation_levels = len(band_edges_dict)
        if isinstance(num_samples_window, int):
            num_samples_window = num_decimation_levels * [num_samples_window]

        for i_level in sorted(band_edges_dict.keys()):
            band_edges = band_edges_dict[i_level]
            if i_level in [0, "0"]:
                d = decimation_factors[i_level]  # 1
                sr = sample_rate
            else:
                # careful with this hardcoded assumption of decimation by 4
                d = decimation_factors[i_level]  # 4
                sr = 1.0 * sample_rate / (d ** int(i_level))
            decimation_obj = DecimationLevel()
            decimation_obj.decimation.level = int(i_level)  # self.decimations_dict[key]
            decimation_obj.decimation.factor = d
            decimation_obj.decimation.sample_rate = sr
            decimation_obj.window.num_samples = num_samples_window[i_level]
            frequencies = decimation_obj.fft_frequecies

            for low, high in band_edges:
                fb = FrequencyBand(left=low, right=high)
                indices = fb.fourier_coefficient_indices(frequencies)
                try:
                    band = Band(
                        decimation_level=i_level,
                        frequency_min=low,
                        frequency_max=high,
                        index_min=indices[0],
                        index_max=indices[-1],
                    )
                except IndexError:
                    print("WHAAAAAAA?")
                # now refine frequency edges based on "canonical" or "exact"
                # self.decimations_dict[i_level].add_band(band)
                decimation_obj.add_band(band)
            self.add_decimation_level(decimation_obj)

    def num_decimation_levels(self):
        return len(self.decimations)

    def drop_reference_channels(self):
        for decimation in self.decimations:
            decimation.reference_channels = []
        return

    def set_input_channels(self, channels):
        for decimation in self.decimations:
            decimation.input_channels = channels

    def set_output_channels(self, channels):
        for decimation in self.decimations:
            decimation.output_channels = channels

    def set_reference_channels(self, channels):
        for decimation in self.decimations:
            decimation.reference_channels = channels

    def set_default_input_output_channels(self):
        self.set_input_channels(self.channel_nomenclature.default_input_channels)
        self.set_output_channels(self.channel_nomenclature.default_output_channels)

    def set_default_reference_channels(self):
        self.set_reference_channels(
            self.channel_nomenclature.default_reference_channels
        )

    def validate_processing(self, kernel_dataset):
        """
        Placeholder.  Some of the checks and methods here maybe better placed in
        TFKernel, which would validate the dataset against the processing config.

        Things that are validated:
        1. The default estimation engine from the json file is "RME_RR", which is fine (
        we expect to in general to do more RR processing than SS) but if there is only
        one station (no remote)then the RME_RR should be replaced by default with "RME".

        2. make sure local station id is defined (correctly from kernel dataset)
        """

        # Make sure a RR method is not being called for a SS config
        if not self.stations.remote:
            for decimation in self.decimations:
                if decimation.estimator.engine == "RME_RR":
                    print("No RR station specified, switching RME_RR to RME")
                    decimation.estimator.engine = "RME"

        # Make sure that a local station is defined
        if not self.stations.local.id:
            print("WARNING: Local station not specified")
            print("Local station should be set from Kernel Dataset")
            self.stations.from_dataset_dataframe(kernel_dataset.df)

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
        window_schemes = [x.windowing_scheme for x in self.decimations]
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

    def json_fn(self, file_id=None):
        json_fn = f"{self.id}_processing_config.json"
        return json_fn

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
        tfh: aurora.transfer_function.transfer_function_header.TransferFunctionHeader
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
