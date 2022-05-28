"""
Creates configs for process one decimation level and one run
"""
from deprecated import deprecated

from aurora.general_helper_functions import BAND_SETUP_PATH
from aurora.config.config_creator import ConfigCreator
from aurora.config.decimation_level_config import DecimationLevelConfig
from aurora.test_utils.parkfield.path_helpers import DATA_PATH
from aurora.test_utils.parkfield.path_helpers import CONFIG_PATH

CHANNEL_SCALE_FACTORS = {}
CHANNEL_SCALE_FACTORS["PKD"] = {}
# CHANNEL_SCALE_FACTORS["PKD"]["ex"] = 1e6
# CHANNEL_SCALE_FACTORS["PKD"]["ey"] = 1e6
CHANNEL_SCALE_FACTORS["SAO"] = {}
# CHANNEL_SCALE_FACTORS["SAO"]["ex"] = 1e6
# CHANNEL_SCALE_FACTORS["SAO"]["ey"] = 1e6

@deprecated(version="0.0.3", reason="new mt_metadata based config")
def create_decimation_level_test_config():
    cfg = DecimationLevelConfig()
    cfg.mth5_path = str(DATA_PATH.joinpath("pkd_test_00.h5"))
    cfg.sample_rate = 40.0
    cfg.emtf_band_setup_file = str(BAND_SETUP_PATH.joinpath("bs_256.cfg"))
    cfg.local_station_id = "PKD"
    cfg.estimation_engine = "OLS"
    output_json = CONFIG_PATH.joinpath("test_single_decimation_level.cfg")
    cfg.to_json(output_json)
    return cfg


@deprecated(version="0.0.3", reason="new mt_metadata based config")
def create_run_test_config():
    mth5_path = DATA_PATH.joinpath("pkd_test_00.h5")
    cc = ConfigCreator(config_path=CONFIG_PATH)
    run_config_path = cc.create_run_config(
        station_id="PKD",
        mth5_path=mth5_path,
        sample_rate=40.0,
        num_samples_window=128,
        num_samples_overlap=32,
        config_id="pkd_test",
        output_channels=["hz", "ex", "ey"],
        # output_channels=["ex", "ey"],
        channel_scale_factors=CHANNEL_SCALE_FACTORS,
    )
    return run_config_path


@deprecated(version="0.0.3", reason="new mt_metadata based config")
def create_run_test_config_remote_reference():
    mth5_path = DATA_PATH.joinpath("pkd_sao_test_00.h5")
    cc = ConfigCreator(config_path=CONFIG_PATH)
    run_config_path = cc.create_run_config(
        station_id="PKD",
        reference_station_id="SAO",
        mth5_path=mth5_path,
        sample_rate=40.0,
        num_samples_window=128,
        num_samples_overlap=32,
        config_id="pkd_test",
        output_channels=["ex", "ey"],
        estimation_engine="RME_RR",
        reference_channels=["hx", "hy"],
        max_number_of_iterations=10,
        channel_scale_factors=CHANNEL_SCALE_FACTORS,
    )
    return run_config_path


@deprecated(version="0.0.3", reason="new mt_metadata based config")
def main():
    create_decimation_level_test_config()
    create_run_test_config()
    create_run_test_config_remote_reference()


if __name__ == "__main__":
    main()
