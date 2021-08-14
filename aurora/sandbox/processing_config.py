"""
Here is a placeholder for a config file.
In the first iteration it will be a dictionary representing the config at a
single decimation level.  We will later (probably) bind a collection of these
together keyed by decimation_level_id.

A good way to approach the various decimation levels maybe to allow a
processing config to generate a decimated config.  After all, it is unlikely
that one will change parameters besides the frequency bands, and even these
are reusable provided we use an EMTF-style band averaging scheme

Once this is mature, put it into aurora.pipelines
"""
# from collections.abc import MutableMapping
from pathlib import Path

import json

from mt_metadata.base import BaseDict
from mt_metadata.base.helpers import NumpyEncoder


class ProcessingConfig(BaseDict):
    """
    ToDo: Deprecate mth5_path from this level after addressing strategy in
    issue #13
    """

    def __init__(self, *args, **kwargs):
        self.mth5_path = kwargs.get("mth5_path", "")  # str or Path()

        # <DECIMATION CONFIG>
        # Support None value for DecimationConfig
        self.decimation_level_id = kwargs.get("decimation_level", 0)
        self.decimation_factor = kwargs.get("decimation_factor", 1)
        self.decimation_method = kwargs.get("decimation_method", "default")
        self.anti_alias_filter = kwargs.get("AAF", "default")
        # <DECIMATION CONFIG>

        # <FOURIER TRANSFORM CONFIG>
        # self.spectral_transform_config = {}
        # spectral_transform_config["taper_family"] = "hamming"

        self.taper_family = "dpss"
        self.taper_additional_args = {"alpha": 3.0}
        self.taper_family = "hamming"
        self.taper_additional_args = {}
        self.num_samples_window = 256
        self.num_samples_overlap = int(self.num_samples_window * 3.0 / 4)
        self.sample_rate = 0.0
        self.prewhitening_type = "first difference"
        self.extra_pre_fft_detrend_type = "linear"
        # </FOURIER TRANSFORM CONFIG>

        # <FREQUENCY BANDS>
        # <EMTF>
        self.band_setup_style = "EMTF"  # "default"
        self.emtf_band_setup_file = "bs_256.cfg"
        # </EMTF>
        # <defualt>
        self.minimum_number_of_cycles = 10  # not used if emtf_band_setup_file present
        # </defualt>
        # </FREQUENCY BANDS>

        # <TRANSFER FUNCTION CONFIG>

        # <ITERATOR>
        self.max_number_of_iterations = 10
        # </ITERATOR>

        # <STATIONS>
        self.local_station_id = ""
        self.reference_station_id = ""
        # </STATIONS>

        # <ESTIMATION>
        self.estimation_engine = "OLS"  # RME
        self.input_channels = ["hx", "hy"]  # optional, default ["hx", "hy"]
        self.output_channels = ["ex", "ey"]  # optional, default ["ex", "ey", "hz"]
        self.reference_channels = []  # optional, default ["hx", "hy"],
        # </ESTIMATION>

        # </TRANSFER FUNCTION CONFIG>

    def validate(self):
        if self.sample_rate <= 0:
            print("sample rate not given")
            raise Exception

    # def __setitem__(self, key, value):
    #     #self.__dict__[key] = validators.validate_value_dict(value)
    #     self.__dict__[key] = value

    # @property
    # def emtf_band_setup_file(self):
    #     return str(self._emtf_band_setup_file)
    #
    # @emtf_band_setup_file.setter
    # def emtf_band_setup_file(self, emtf_band_setup_file):
    #     self._emtf_band_setup_file = Path(emtf_band_setup_file)

    # @property
    # def local_station_id(self, local_station_id):
    #     return self._local_station_id
    #
    # @local_station_id.setter
    # def local_station_id(self, local_station_id):
    #     self._local_station_id = local_station_id

    def from_json(self, json_fn):
        """

        Read schema standards from json

        :param json_fn: full path to json file
        :type json_fn: string or Path
        :return: full path to json file
        :rtype: Path

        """

        json_fn = Path(json_fn)
        if not json_fn.exists():
            msg = f"JSON schema file {json_fn} does not exist"
            print(msg)
            raise Exception

        with open(json_fn, "r") as fid:
            json_dict = json.load(fid)
        print("SKIPPING VALIDATION FOR NOW")
        self.__dict__ = json_dict


class RunConfig(BaseDict):
    """
    Class to contain a collection of DecimationLevelConfigs
    This will need some attention; the to/from json methods are not robust in
    the sense that we need to overwrite BaseDict's method.

    If we add attributes to the
    run_config on the high level
    (such as an ID, or a label) we cannot
    """

    def __init__(self, **kwargs):
        self.config_id = kwargs.get("config_id", "run_config")
        self.mth5_path = kwargs.get("mth5_path", "")
        self.local_station_id = ""
        self.reference_station_id = ""
        self.initial_sample_rate = kwargs.get("initial_sample_rate", 0.0)
        self.decimation_level_configs = {}

    @property
    def number_of_decimation_levels(self):
        return len(self.decimation_level_configs.keys())

    @property
    def decimation_level_ids(self):
        return sorted(self.decimation_level_configs.keys())

    def to_json(self, json_fn=None, indent=" " * 4):
        if json_fn is None:
            json_fn = self.config_id
        json_fn = Path(json_fn)
        self_dict = self.__dict__["decimation_level_configs"]
        json_dict = {}
        json_dict["config_id"] = self.config_id
        json_dict["mth5_path"] = self.mth5_path
        json_dict["local_station_id"] = self.local_station_id
        json_dict["reference_station_id"] = self.reference_station_id
        print(self.decimation_level_ids)
        for dec_level_id in self.decimation_level_ids:
            json_dict[dec_level_id] = self_dict[dec_level_id].__dict__

        with open(json_fn, "w") as fid:
            json.dump(json_dict, fid, cls=NumpyEncoder, indent=indent)

        return

    def from_json(self, json_fn):
        """

        Read schema standards from json

        :param json_fn: full path to json file
        :type json_fn: string or Path
        :return: full path to json file
        :rtype: Path

        """

        json_fn = Path(json_fn)
        if not json_fn.exists():
            msg = f"JSON schema file {json_fn} does not exist"
            print(msg)
            raise Exception

        with open(json_fn, "r") as fid:
            json_dict = json.load(fid)

        self.config_id = json_dict.pop("config_id")
        self.mth5_path = json_dict.pop("mth5_path")
        self.local_station_id = json_dict.pop("local_station_id")
        self.reference_station_id = json_dict.pop("reference_station_id")
        decimation_level_ids = sorted(json_dict.keys())
        for decimation_level_id in decimation_level_ids:
            decimation_level_processing_config = ProcessingConfig()
            decimation_level_processing_config.__dict__ = json_dict[decimation_level_id]
            self.decimation_level_configs[
                int(decimation_level_id)
            ] = decimation_level_processing_config


def test_create_decimation_level_config():
    cfg = ProcessingConfig()
    cfg.sample_rate = 1.0
    cfg.local_station_id = "PKD"
    cfg.validate()


def test_create_run_config():
    json_fn = "run_config_00.json"
    run_config = RunConfig()
    decimation_factors = [1, 4, 4, 4]
    for i_decimation_level in range(len(decimation_factors)):
        cfg = ProcessingConfig()
        cfg.decimation_level_id = i_decimation_level
        cfg.decimation_factor = decimation_factors[i_decimation_level]
        run_config.decimation_level_configs[i_decimation_level] = cfg
    run_config.to_json(json_fn=json_fn)

    run_config_from_file = RunConfig()
    run_config_from_file.from_json(json_fn)
    return run_config


def main():
    test_create_decimation_level_config()
    test_create_run_config()


if __name__ == "__main__":
    main()
