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
from pathlib import Path

from aurora.config.decimation_level_config import DecimationLevelConfig
from aurora.config.processing_config import RunConfig
from aurora.general_helper_functions import BAND_SETUP_PATH
from aurora.config import Processing, Station, Run, BANDS_DEFAULT_FILE


class ConfigCreator:
    
    def __init__(self, **kwargs):
        default_config_path = Path("config")
        self.config_path = kwargs.get("config_path", default_config_path)


    # pass an mth5, it has: station_id, run_id, mth5_path, sample_rate
    def create_run_config(
        self, station_id="", run_id="", mth5_path="", sample_rate=-1.0, **kwargs
    ):
        default_band_setup = str(BAND_SETUP_PATH.joinpath("bs_test.cfg"))
        config_id = kwargs.get("config_id", f"{station_id}-{run_id}")
        decimation_factors = kwargs.get("decimation_factors", [1, 4, 4, 4])
        num_samples_window = kwargs.get("num_samples_window", 128)
        num_samples_overlap = kwargs.get("num_samples_overlap", 32)
        output_channels = kwargs.get("output_channels", ["hz", "ex", "ey"])
        band_setup_file = kwargs.get("band_setup_file", default_band_setup)
        reference_station_id = kwargs.get("reference_station_id", "")
        reference_channels = kwargs.get("reference_channels", [])
        channel_scale_factors = kwargs.get("channel_scale_factors", {})
        estimation_engine = kwargs.get("estimation_engine", "RME")
        max_number_of_iterations = kwargs.get("max_number_of_iterations", 10)
        max_number_of_redescending_iterations = kwargs.get(
            "max_number_of_redescending_iterations", 2
        )
        if reference_station_id:
            reference_mth5_path = kwargs.get("reference_mth5_path", mth5_path)
        else:
            reference_mth5_path = ""

        run_config = RunConfig()
        run_config.config_id = config_id
        run_config.mth5_path = str(mth5_path)
        run_config.local_station_id = f"{station_id}"
        run_config.initial_sample_rate = sample_rate
        run_config.reference_station_id = f"{reference_station_id}"
        run_config.reference_mth5 = str(reference_mth5_path)
        run_config.channel_scale_factors = channel_scale_factors

        if run_config.reference_station_id:
            config_id = f"{config_id}-RR_{run_config.reference_station_id}"
            run_config.config_id = config_id

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
            cfg.emtf_band_setup_file = band_setup_file
            cfg.estimation_engine = estimation_engine
            cfg.output_channels = output_channels
            cfg.reference_channels = reference_channels
            cfg.max_number_of_iterations = max_number_of_iterations
            cfg.max_number_of_redescending_iterations = (
                max_number_of_redescending_iterations
            )
            run_config.decimation_level_configs[i_decimation_level] = cfg

        json_fn = run_config.json_fn()#config_id + "_run_config.json"
        self.config_path.mkdir(exist_ok=True)
        json_path = self.config_path.joinpath(json_fn)
        run_config.to_json(json_fn=json_path)
        return json_path
    
    def create_run_processing_object(
            self, station_id=None, run_id=None, mth5_path=None, sample_rate=-1, 
            input_channels=["hx", "hy"], output_channels=["hz", "ex", "ey"], **kwargs):
        """
        Create a default processing object
        
        :return: DESCRIPTION
        :rtype: TYPE

        """
        processing_obj = Processing(id=f"{station_id}-{run_id}", **kwargs)
        
        if not isinstance(run_id, list):
            run_id = [run_id]
            
        runs = []
        for run in run_id:
            run_obj = Run(
                id=run_id,
                input_channels=input_channels,
                output_channels=output_channels,
                sample_rate=sample_rate)
            runs.append(run_obj)
            
        station_obj = Station(id=station_id, mth5_path=mth5_path)
        station_obj.runs = runs

        processing_obj.stations.local = station_obj
        processing_obj.read_emtf_bands(BANDS_DEFAULT_FILE)
        
        for key in sorted(processing_obj.decimations_dict.keys()):
            if key in [0, "0"]:
                d = 1
            else:
                d = 4
            processing_obj.decimations_dict[key].decimation.factor = d
        
        return processing_obj
    
    def to_json(self, path, processing_object, nested=True, required=False):
        """
        Write a processing object to path
        
        :param path: DESCRIPTION
        :type path: TYPE
        :param processing_object: DESCRIPTION
        :type processing_object: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        
        with open(path, "w") as fid:
            fid.write(processing_object.to_json(nested=nested, required=required))


def test_cas04():
    station_id = "CAS04"
    run_id = "003"
    h5_path = "/home/kkappler/.cache/iris_mt/from_iris_dmc.h5"
    sample_rate = 1.0
    config_maker = ConfigCreator()
    config_maker.create_run_config(station_id, run_id, h5_path, sample_rate)


def main():
    test_cas04()


if __name__ == "__main__":
    main()
