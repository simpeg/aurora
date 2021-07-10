"""
Here is a placeholder for a config file.
In the first iteration it will be a dictionary representing the config at a
single decimation level.  We will later (probably) bind a collection of these
together keyed by decimation_level_id.

"""
#from collections.abc import MutableMapping
from pathlib import Path

import json

from mt_metadata.base import BaseDict

class ProcessingConfig(BaseDict):

    def __init__(self, *args, **kwargs):
        self.mth5_path = kwargs.get("mth5_path", "")
        #str or Path()

        # <FOURIER TRANSFORM CONFIG>
        # self.spectral_transform_config = {}
        # spectral_transform_config["taper_family"] = "hamming"
        
        self.taper_family = "hamming"
        self.taper_additional_args = {}
        self.num_samples_window = 128
        self.num_samples_overlap = int(self.num_samples_window*3./4)
        self.sample_rate = 0.0
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
        self.remote_reference_station_id = ""
            # </STATIONS>

            # <ESTIMATION>
        self.estimation_engine = "OLS" #RME
        self.input_channels = ["hx", "hy"]  # optional, default ["hx", "hy"]
        self.output_channels = ["ex", "ey"]  # optional, default ["ex", "ey", "hz"]
        self.reference_channels = [] # optional, default ["hx", "hy"],
            # </ESTIMATION>

        # </TRANSFER FUNCTION CONFIG>

    def validate(self):
        if self.sample_rate <=0:
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
            logger.error(msg)
            MTSchemaError(msg)

        with open(json_fn, "r") as fid:
            json_dict = json.load(fid)
        print("SKIPPING VALIDATION FOR NOW")
        self.__dict__ = json_dict

def test_can_create_config():
    cfg = ProcessingConfig()
    cfg.sample_rate = 1.0
    cfg.local_station_id = "PKD"
    cfg.validate()


def main():
    test_can_create_config()

if __name__ == '__main__':
    main()
