"""
Here is a placeholder for a config file.
In the first iteration it will be a dictionary representing the config at a
single decimation level.  We will later (probably) bind a collection of these
together keyed by decimation_level_id.

"""
import json

from mt_metadata.base import BaseDict
from collections.abc import MutableMapping
from pathlib import Path

class ProcessingConfig(BaseDict):

    def __init__(self, *args, **kwargs):

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
        self._local_station_id = ""
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

    @property
    def local_station_id(self, local_station_id):
        return self._local_station_id

    @local_station_id.setter
    def local_station_id(self, local_station_id):
        self._local_station_id = local_station_id
    #
    # @emtf_band_setup_file.setter
    # def emtf_band_setup_file(self, emtf_band_setup_file):
    #     self._emtf_band_setup_file = Path(emtf_band_setup_file)
        #return str(self._emtf_band_setup_file)

    # def to_json(self, json_path, indent=" " * 4):
    #     """
    #     Write processing config to json
    #     Parameters
    #     ----------
    #     json_path: pathlib.Path or str
    #         full path to json file
    #     indent str
    #         json indentation string
    #
    #     Returns
    #     json_path: pathlib.Path or str
    #         full path to json file
    #     -------
    #
    #     """
    #     """
    #
    #     :return: full path to json file
    #     :rtype: Path
    #
    #             """
    #
    #     json_path = Path(json_path)
    #
    #     json_dict = dict(
    #         [(k, v) for k, v in self.items() if k not in ["logger"]])
    #     with open(json_fn, "w") as fid:
    #         json.dump(json_dict, fid, cls=NumpyEncoder, indent=indent)
    #
    #     return json_path
    #
    #
    # def from_json(self, json_path):
    #     json_fn = Path(json_path)
    #     if not json_path.exists():
    #         msg = f"JSON schema file {json_path} does not exist"
    #         print(msg)
    #         raise Exception
    #
    #     with open(json_path, "r") as fid:
    #         json_dict = json.load(fid)
    #
    #     # valid_dict = {}
    #     # for k, v in json_dict.items():
    #     #     valid_dict[k] = validators.validate_value_dict(v)
    #     # self.update(valid_dict)
    #     return json_dict

def test_can_create_config():
    cfg = ProcessingConfig()
    cfg.sample_rate = 1.0
    cfg.local_station_id = "PKD"
    cfg.validate()

def create_config_for_test_case(test_case_id):
    if test_case_id == "synthetic test 1":
        from aurora.general_helper_functions import SANDBOX
        cfg = ProcessingConfig()
        cfg.emtf_band_setup_file = str(SANDBOX.joinpath("bs_256.cfg"))
        json_path = "_".join(test_case_id)
        cfg.to_json(json_path)


def main():
    test_can_create_config()
    create_config_for_test_case("synthetic test 1")

if __name__ == '__main__':
    main()