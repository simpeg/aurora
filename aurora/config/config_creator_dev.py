from pathlib import Path

from aurora.config.decimation_level_config import DecimationLevelConfig
from aurora.config.processing_config import RunConfig
from aurora.general_helper_functions import SANDBOX


# pass an mth5, it has: station_id, run_id, mth5_path, sample_rate
def create_run_config(station_id, run_id, mth5_path, sample_rate, **kwargs):
    # station_id = kwargs.get("station_id", "station_id")
    # run_id = kwargs.get("run_id", "run_id")
    decimation_factors = kwargs.get("decimation_factors", [1, 4, 4, 4])
    num_samples_window = kwargs.get("num_samples_window", 128)
    num_samples_overlap = kwargs.get("num_samples_overlap", 32)
    output_channels = kwargs.get("output_channels", ["hz", "ex", "ey"])  # ["ex", "ey"]
    run_config = RunConfig()
    run_config.config_id = f"{station_id}-{run_id}"
    run_config.mth5_path = str(mth5_path)
    run_config.local_station_id = f"{station_id}"
    run_config.initial_sample_rate = 1.0

    downsample_factor = 1.0
    for i_decimation_level in range(len(decimation_factors)):
        decimation_factor = decimation_factors[i_decimation_level]
        downsample_factor /= decimation_factor
        cfg = DecimationLevelConfig()
        cfg.decimation_level_id = i_decimation_level
        cfg.decimation_factor = decimation_factor
        cfg.num_samples_window = num_samples_window
        cfg.num_samples_overlap = num_samples_overlap
        cfg.sample_rate = run_config.initial_sample_rate * downsample_factor
        # cfg.emtf_band_setup_file = str(SANDBOX.joinpath("bs_256.cfg"))
        cfg.emtf_band_setup_file = str(SANDBOX.joinpath("bs_test.cfg"))
        cfg.estimation_engine = "RME"
        cfg.output_channels = output_channels
        run_config.decimation_level_configs[i_decimation_level] = cfg
        json_fn = run_config.config_id + "_run_config.json"
        config_path = Path("config")
        config_path.mkdir(exist_ok=True)
        json_path = config_path.joinpath(json_fn)
        run_config.to_json(json_fn=json_path)


def main():
    create_run_config("CAS04", "003", "/home/cas04", 1.0)  # , **kwargs):


if __name__ == "__main__":
    main()
