# -*- coding: utf-8 -*-
"""
Extend the mt_metadata.transfer_functions.processing.aurora.processing.Processing class
with some aurora-specific methods.
"""
import pathlib

# =============================================================================
# Imports
# =============================================================================

from aurora.time_series.windowing_scheme import window_scheme_from_decimation
from loguru import logger
from mt_metadata.transfer_functions.processing.aurora.processing import Processing
from mt_metadata.utils.list_dict import ListDict
from typing import Optional, Union
import pandas as pd


class Processing(Processing):
    def __init__(self, **kwargs):
        """
        Constructor

        Parameters
        ----------
        kwargs
        """
        # super().__init__(attr_dict=attr_dict, **kwargs)
        super().__init__(**kwargs)

    def window_scheme(self, as_type="df"):
        """
        Make a dataframe of processing parameters one row per decimation level.

        Parameters
        ----------
        as_type: Optional[str]
            if "df" return a dataframe, if "dict" return dict
        Returns
        -------
        windowing: Union[dict, pd.DataFrame]
            return type depends on as_type argument.

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
            logger.error(f"unexpected rtype for window_scheme {as_type}")
            raise TypeError

    def decimation_info(self):
        """
        Zips decimation level ids to the Decimation objects adn returns as a dict

        Returns
        -------
        decimation_info: dict
            The decimation objects keyed by decimation level id.
        """
        decimation_ids = [x.decimation.level for x in self.decimations]
        decimation_factors = [x.decimation.factor for x in self.decimations]
        decimation_info = dict(zip(decimation_ids, decimation_factors))
        return decimation_info

    def save_as_json(
        self,
        filename: Optional[Union[str, pathlib.Path, None]] = None,
        nested: Optional[bool] = True,
        required: Optional[bool] = False,
    ) -> None:
        """
        Exports self to a JSON

        Parameters
        ----------
        filename: Optional[Union[str, pathlib.Path, None]
            Where to write the json
        nested: Optional[bool] = True,
            An mt_metadata argument
        required: Optional[bool] = False,
            An mt_metadata argument

        """
        if filename is None:
            filename = self.json_fn()
        json_str = self.to_json(nested=nested, required=required)
        with open(filename, "w") as f:
            f.write(json_str)

    def emtf_tf_header(self, dec_level_id: int) -> ListDict:
        """
        Returns a ListDict object that has the information that was in the old EMTF TF
         Header object.  This may be deprecated in future -- it is an artefact of the
         old matlab implementation.

        Parameters
        ----------
        dec_level_id: int
            This may tolerate strings in the future, but keep as int for now

        Returns
        -------
            tfh: ListDict
            Object with the properties of the old EMTF TransferFunctionHeader class.

        """
        tfh = ListDict()
        tfh.processing_scheme = self.decimations[dec_level_id].estimator.engine
        tfh.local_station = self.stations.local
        tfh.remote_station = self.stations.remote
        tfh.input_channels = self.decimations[dec_level_id].input_channels
        tfh.output_channels = self.decimations[dec_level_id].output_channels
        tfh.reference_channels = self.decimations[dec_level_id].reference_channels
        tfh.decimation_level_id = dec_level_id

        return tfh

    def make_tf_level(self, dec_level_id: int):
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
    Convenience class for storing metadata for a TF estimate.
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
        _remote_station: same object type as local station
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
        self.local_station = kwargs.get("local_station", None)
        self.remote_station = kwargs.get("remote_station", None)
        self.input_channels = kwargs.get("input_channels", ["hx", "hy"])
        self.output_channels = kwargs.get("output_channels", ["ex", "ey"])
        self.reference_channels = kwargs.get("reference_channels", [])
        self.decimation_level_id = kwargs.get("decimation_level_id", None)
        self.user_meta_data = None  # placeholder for anything
