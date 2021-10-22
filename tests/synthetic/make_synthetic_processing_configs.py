from aurora.config.config_creator import ConfigCreator
from aurora.general_helper_functions import TEST_PATH


CONFIG_PATH = TEST_PATH.joinpath("synthetic", "config")
DATA_PATH = TEST_PATH.joinpath("synthetic", "data")


def create_run_config_for_test_case(test_case_id):
    mth5_path = DATA_PATH.joinpath(f"{test_case_id}.h5")

    cc = ConfigCreator(config_path=CONFIG_PATH)
    estimation_engine = "RME"
    local_station_id = test_case_id
    reference_station_id = ""
    reference_channels = []
    if test_case_id == "test12rr":
        estimation_engine = "RME_RR"
        reference_channels = ["hx", "hy"]
        local_station_id = "test1"
        reference_station_id = "test2"

    run_config_path = cc.create_run_config(
        station_id=local_station_id,
        mth5_path=mth5_path,
        sample_rate=1.0,
        num_samples_window=128,
        num_samples_overlap=32,
        config_id=f"{test_case_id}",
        output_channels=["ex", "ey"],
        reference_station_id=reference_station_id,
        reference_channels=reference_channels,
        estimation_engine=estimation_engine,
    )
    return run_config_path


#
# def create_run_config_for_test_case_old(test_case_id):
#     """
#     2021-09-08: configs for test1 are now made in process_synthetic_data_standard
#
#     Parameters
#     ----------
#     test_case_id
#
#     Returns
#     -------
#
#     """
#     from pathlib import Path
#     from aurora.config.decimation_level_config import DecimationLevelConfig
#     from aurora.config.processing_config import RunConfig
#     from aurora.general_helper_functions import BAND_SETUP_PATH
#     if test_case_id in ["test1", "test2", "test12rr"]:
#         decimation_factors = [1, 4, 4, 4]
#         run_config = RunConfig()
#         run_config.config_id = test_case_id
#         run_config.mth5_path = str(Path("data", f"{test_case_id}.h5"))
#         run_config.local_station_id = f"{test_case_id}"
#         run_config.initial_sample_rate = 1.0
#         if test_case_id == "test12rr":
#             run_config.reference_channels = ["hx", "hy"]
#             run_config.local_station_id = "test1"
#             run_config.reference_station_id = "test2"
#
#         downsample_factor = 1.0
#         for i_decimation_level in range(len(decimation_factors)):
#             decimation_factor = decimation_factors[i_decimation_level]
#             downsample_factor /= decimation_factor
#             cfg = DecimationLevelConfig()
#             cfg.decimation_level_id = i_decimation_level
#             cfg.decimation_factor = decimation_factor
#             cfg.num_samples_window = 128
#             cfg.num_samples_overlap = 32
#             cfg.sample_rate = run_config.initial_sample_rate * downsample_factor
#             # cfg.emtf_band_setup_file = str(BAND_SETUP_PATH.joinpath("bs_256.cfg"))
#             cfg.emtf_band_setup_file = str(BAND_SETUP_PATH.joinpath("bs_test.cfg"))
#             cfg.estimation_engine = "RME"
#             cfg.output_channels = ["hz", "ex", "ey"]
#             if test_case_id == "test12rr":
#                 cfg.estimation_engine = "RME_RR"
#                 cfg.reference_channels = run_config.reference_channels  # HACKY
#             run_config.decimation_level_configs[i_decimation_level] = cfg
#
#         json_fn = test_case_id.replace(" ", "_") + "_run_config_old.json"
#         config_path = TEST_PATH.joinpath("synthetic", "config")
#         config_path.mkdir(exist_ok=True)
#         json_path = config_path.joinpath(json_fn)
#         run_config.to_json(json_fn=json_path)
#     else:
#         print(f"test_case_id {test_case_id} not recognized")
#         raise Exception


def main():
    create_run_config_for_test_case("test1")
    create_run_config_for_test_case("test2")
    create_run_config_for_test_case("test12rr")
    # create_run_config_for_test_case_old("test1")
    # create_run_config_for_test_case_old("test2")
    # create_run_config_for_test_case_old("test12rr")


if __name__ == "__main__":
    main()
