from pathlib import Path

from aurora.general_helper_functions import SANDBOX
from aurora.general_helper_functions import TEST_PATH
from aurora.sandbox.processing_config import ProcessingConfig

def create_config_for_test_case(test_case_id):
    if test_case_id in ["test1", "test2", "test12rr"]:
        cfg = ProcessingConfig()
        cfg.mth5_path = str(Path("data",f"{test_case_id}.h5"))
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
        config_path = TEST_PATH.joinpath("synthetic", "config")
        config_path.mkdir(exist_ok=True)
        json_path = config_path.joinpath(json_fn)
        cfg.to_json(json_path)
    else:
        print(f"test_case_id {test_case_id} not recognized")
        raise Exception

def main():
    create_config_for_test_case("test1")
    create_config_for_test_case("test2")
    create_config_for_test_case("test12rr")

if __name__ == '__main__':
    main()
