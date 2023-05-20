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
from mt_metadata.utils.list_dict import ListDict
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

        tfh = EMTFTFHeader(
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

        tf_obj = TTFZ(
            dec_level_id,
            self.decimations[dec_level_id].frequency_bands_obj(),
            processing_config=self,
        )

        return tf_obj


class EMTFTFHeader(ListDict):
    """
    Convenince class for storing metadata for a TF estimate.
    Based on Gary Egbert's TFHeader.m originally in
    iris_mt_scratch/egbert_codes-20210121T193218Z-001/egbert_codes/matlabPrototype_10-13-20/TF/classes

    It completely depends on the Processing class
    """

    def __init__(self, **kwargs):
        """
        Parameters
        _local_station : mt_metadata.transfer_functions.tf.station.Station()
            Station metadata object for the station to be estimated (
            location, channel_azimuths, etc.)
        _reference_station: same object type as local station
            if no remote reference then this can be None
        output_channels: list
            Probably a list of channel keys -- usually ["ex","ey","hz"]
        input_channels : list
            Probably a list of channel keys -- usually ["hx","hy"]
            These are the channels being provided as input to the regression.
        reference_channels : list
            These are the channels being used from the RR station. This is a
            channel list -- usually ["hx", "hy"]
        processing_scheme: str
            Denotes the regression engine used to estimate the transfer
            function.  One of "OLS" or "RME", "RME_RR.  Future
            versions could include , "multivariate array", "multiple remote",
            etc.

        """
        super().__init__()
        self.processing_scheme = kwargs.get("processing_scheme", None)
        self._local_station = kwargs.get("local_station", None)
        self._reference_station = kwargs.get("reference_station", None)
        self.input_channels = kwargs.get("input_channels", ["hx", "hy"])
        self.output_channels = kwargs.get("output_channels", ["ex", "ey"])
        self.reference_channels = kwargs.get("reference_channels", [])
        self.decimation_level_id = kwargs.get("decimation_level_id", None)
        self.user_meta_data = None  # placeholder for anything

    @property
    def local_station_id(self):
        try:
            station_id = self.local_station.id
        except AttributeError:
            station_id = self._local_station_id
        return station_id

    @property
    def remote_station_id(self):
        try:
            station_id = self.remote_station.id
        except AttributeError:
            station_id = self._remote_station_id
        return station_id

    @property
    def local_station(self):
        return self._local_station

    @property
    def num_input_channels(self):
        return len(self.input_channels)

    @property
    def num_output_channels(self):
        return len(self.output_channels)

    @property
    def local_channels(self):
        return self.input_channels + self.output_channels
