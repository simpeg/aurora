import json
from pathlib import Path

from mt_metadata.base import BaseDict


class DecimationLevelConfig(BaseDict):
    """
    ToDo: Deprecate mth5_path from this level after addressing strategy in
    issue #13

    """

    def __init__(self, *args, **kwargs):
        """

        Parameters
        ----------
        arg
        kwargs
        """
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
        self.estimation_engine = kwargs.get("estimation_engine", "OLS")  # RME
        self.estimate_per_channel = (
            True  # all channels at once or one channel at a time
        )
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
