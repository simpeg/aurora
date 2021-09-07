from pathlib import Path

from aurora.config.decimation_level_config import DecimationLevelConfig
from aurora.config.processing_config import RunConfig
from aurora.general_helper_functions import SANDBOX
from aurora.general_helper_functions import TEST_PATH


def create_run_config_for_test_case(test_case_id):
    if test_case_id in ["test1", "test2", "test12rr"]:
        decimation_factors = [1, 4, 4, 4]
        run_config = RunConfig()
        run_config.config_id = test_case_id
        run_config.mth5_path = str(Path("data", f"{test_case_id}.h5"))
        run_config.local_station_id = f"{test_case_id}"
        run_config.initial_sample_rate = 1.0
        if test_case_id == "test12rr":
            run_config.reference_channels = ["hx", "hy"]
            run_config.local_station_id = "test1"
            run_config.reference_station_id = "test2"

        downsample_factor = 1.0
        for i_decimation_level in range(len(decimation_factors)):
            decimation_factor = decimation_factors[i_decimation_level]
            downsample_factor /= decimation_factor
            cfg = DecimationLevelConfig()
            cfg.decimation_level_id = i_decimation_level
            cfg.decimation_factor = decimation_factor
            cfg.num_samples_window = 128
            cfg.num_samples_overlap = 32
            cfg.sample_rate = run_config.initial_sample_rate * downsample_factor
            # cfg.emtf_band_setup_file = str(SANDBOX.joinpath("bs_256.cfg"))
            cfg.emtf_band_setup_file = str(SANDBOX.joinpath("bs_test.cfg"))
            cfg.estimation_engine = "RME"
            cfg.output_channels = ["hz", "ex", "ey"]
            if test_case_id == "test12rr":
                cfg.estimation_engine = "TRME_RR"
                cfg.reference_channels = run_config.reference_channels  # HACKY
            run_config.decimation_level_configs[i_decimation_level] = cfg

        json_fn = test_case_id.replace(" ", "_") + "_run_config.json"
        config_path = TEST_PATH.joinpath("synthetic", "config")
        config_path.mkdir(exist_ok=True)
        json_path = config_path.joinpath(json_fn)
        run_config.to_json(json_fn=json_path)
    else:
        print(f"test_case_id {test_case_id} not recognized")
        raise Exception


def main():
    create_run_config_for_test_case("test1")
    create_run_config_for_test_case("test2")
    create_run_config_for_test_case("test12rr")


if __name__ == "__main__":
    main()
