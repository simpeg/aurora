"""
Helper class to make config files.

Note: the config is still evolving and this class and its methods are expected to
change.

Currently There are two critical classes.
DecimationLevelConfig: These are all the
info required to process a contiguous time series block at fixed samppling rate.

RunConfig: This is a collection of processing configs, together with specifications for
decimation.
"""
import os
from pathlib import Path

from aurora.config.decimation_level_config import DecimationLevelConfig
from aurora.config.processing_config import RunConfig
from aurora.general_helper_functions import BAND_SETUP_PATH


class ConfigCreator(object):
    def __init__(self, **kwargs):
        self.config_path = kwargs.get("config_path", Path(os.getcwd()).joinpath("cfg"))

    # def to_json(self):

    # pass an mth5, it has: station_id, run_id, mth5_path, sample_rate
    def create_run_config(
        self, station_id="", run_id="", mth5_path="", sample_rate=-1.0, **kwargs
    ):
        config_id = kwargs.get("config_id", f"{station_id}-{run_id}")
        decimation_factors = kwargs.get("decimation_factors", [1, 4, 4, 4])
        num_samples_window = kwargs.get("num_samples_window", 128)
        num_samples_overlap = kwargs.get("num_samples_overlap", 32)
        output_channels = kwargs.get(
            "output_channels", ["hz", "ex", "ey"]
        )  # ["ex", "ey"]

        run_config = RunConfig()
        run_config.config_id = config_id
        run_config.mth5_path = str(mth5_path)
        run_config.local_station_id = f"{station_id}"
        run_config.initial_sample_rate = sample_rate

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
            cfg.band_setup_style = "EMTF"
            cfg.emtf_band_setup_file = str(BAND_SETUP_PATH.joinpath("bs_test.cfg"))
            cfg.estimation_engine = "RME"
            cfg.output_channels = output_channels
            run_config.decimation_level_configs[i_decimation_level] = cfg

        json_fn = run_config.config_id + "_run_config.json"
        self.config_path.mkdir(exist_ok=True)
        json_path = self.config_path.joinpath(json_fn)
        run_config.to_json(json_fn=json_path)
        return json_path


def test_cas04():
    example_sta = "CAS04"
    example_run = "003"
    h5_path = "/home/kkappler/.cache/iris_mt/from_iris_dmc.h5"
    example_samplerate = 1.0
    config_maker = ConfigCreator()
    config_maker.create_run_config(
        example_sta, example_run, h5_path, example_samplerate
    )  # ,
    # **kwargs):


def main():
    test_cas04()


if __name__ == "__main__":
    main()
