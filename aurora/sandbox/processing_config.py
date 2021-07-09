"""
Here is a placeholder for a config file.
In the first iteration it will be a dictionary representing the config at a
single decimation level.  We will later (probably) bind a collection of these
together keyed by decimation_level_id.

"""
#from collections.abc import MutableMapping
from pathlib import Path

import json

from aurora.general_helper_functions import TEST_PATH
from mt_metadata.base import BaseDict

class ProcessingConfig(BaseDict):

    def __init__(self, *args, **kwargs):
        self.mth5_path = kwargs.get("mth5_path", "")

        # <FOURIER TRANSFORM CONFIG>
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

def create_config_for_test_case(test_case_id):
    if test_case_id in ["test1", "test2", "test12rr"]:
        from aurora.general_helper_functions import SANDBOX
        cfg = ProcessingConfig()
        cfg.mth5_path = f"{test_case_id}.h5"
        cfg.num_samples_window = 128
        cfg.num_samples_overlap = 32
        cfg.local_station_id = f"{test_case_id}"
        cfg.sample_rate = 1.0
        cfg.emtf_band_setup_file = str(SANDBOX.joinpath("bs_256.cfg"))
        cfg.estimation_engine = "RME"
        if test_case_id=="test12rr":
            cfg.reference_channels = ["hx", "hy"]
            cfg.local_station_id = "test1"
            cfg.remote_reference_station_id = "test2"
        json_fn = test_case_id.replace(" ","_") + "_processing_config.json"
        json_path = TEST_PATH.joinpath("emtf_synthetic", json_fn)
        cfg.to_json(json_path)
    else:
        print(f"test_case_id {test_case_id} not recognized")
        raise Exception

def main():
    test_can_create_config()
    create_config_for_test_case("test1")
    create_config_for_test_case("test2")
    create_config_for_test_case("test12rr")

if __name__ == '__main__':
    main()
