import json

from aurora.config import BANDS_DEFAULT_FILE
from aurora.config import BANDS_256_FILE
from aurora.config.metadata import Run
from aurora.config import Station
from aurora.config.config_creator import ConfigCreator
from aurora.general_helper_functions import TEST_PATH
from aurora.tf_kernel.dataset import DatasetDefinition
from mth5.utils.helpers import initialize_mth5

CONFIG_PATH = TEST_PATH.joinpath("synthetic", "config")
DATA_PATH = TEST_PATH.joinpath("synthetic", "data")


def create_test_run_config(test_case_id, matlab_or_fortran=""):
    """
    Use config creator to generate a processing config file for the synthetic data.  
    
    Parameters
    ----------
    test_case_id: string
        "test1", "test2", "test12rr"
    matlab_or_fortran: str
        "", "matlab", "fortran"

    Returns
    -------

    When creating the dataset dataframe, make it have these columns:
    [
            "station_id",
            "run_id",
            "start",
            "end",
            "mth5_path",
            "sample_rate",
            "input_channels",
            "output_channels",
            "remote",
        ] 
    
    """
    mth5_path = DATA_PATH.joinpath(f"{test_case_id}.h5")

    estimation_engine = "RME"
    local_station_id = test_case_id
    reference_station_id = ""
    reference_channels = []
    if test_case_id == "test1r2":
        estimation_engine = "RME_RR"
        reference_channels = ["hx", "hy"]
        local_station_id = "test1"
        reference_station_id = "test2"
        mth5_path = DATA_PATH.joinpath("test12rr.h5")
    if test_case_id == "test2r1":
        estimation_engine = "RME_RR"
        reference_channels = ["hx", "hy"]
        local_station_id = "test2"
        reference_station_id = "test1"
        mth5_path = DATA_PATH.joinpath("test12rr.h5")

    if matlab_or_fortran == "matlab":
        emtf_band_setup_file = BANDS_256_FILE
        num_samples_window = 256
        num_samples_overlap = 64
        config_id = f"{local_station_id}-{matlab_or_fortran}"
    elif matlab_or_fortran == "fortran":
        emtf_band_setup_file = BANDS_DEFAULT_FILE
        num_samples_window = 128
        num_samples_overlap = 32
        config_id = f"{local_station_id}-{matlab_or_fortran}"
    else:
        emtf_band_setup_file = BANDS_DEFAULT_FILE
        num_samples_window = 128
        num_samples_overlap = 32
        config_id = f"{local_station_id}"


    mth5_obj = initialize_mth5(mth5_path, mode="a")
    ch_summary = mth5_obj.channel_summary
    dataset_definition = DatasetDefinition()
    dataset_definition.from_mth5_channel_summary(ch_summary)
    dd_df = dataset_definition.df
    #channel_summary_to_dataset_definition2(ch_summary)
    dd_df["mth5_path"] = str(mth5_path)
    mth5_obj.close_mth5()
    cc = ConfigCreator(config_path=CONFIG_PATH)
    
    if test_case_id=="test1":
        p = cc.create_run_processing_object(emtf_band_file=emtf_band_setup_file)
        p.id = config_id
        dd_df["remote"] = False
        p.stations.from_dataset_dataframe(dd_df)

        for decimation in p.decimations:
            decimation.estimator.engine = estimation_engine
            decimation.window.type = "hamming"
            decimation.window.num_samples = num_samples_window
            decimation.window.overlap = num_samples_overlap
            decimation.regression.max_redescending_iterations = 2
    
        cc.to_json(p)
    elif test_case_id=="test2r1":
        dd_df["remote"] = [True, False]
        p = cc.create_run_processing_object(emtf_band_file=emtf_band_setup_file)
        p.id = config_id
        p.stations.from_dataset_dataframe(dd_df)
        # rr = Station()
        # rr.id = "test1"
        # rr.mth5_path = p.stations.local.mth5_path
        # p.stations.add_remote(rr)
        # p.stations.remote[0].id

        #run_list can also just be hard-coded here as ['001',]
        # run_list = dataset_definition.df.run_id.unique().tolist()
        # p.stations.local.runs = run_list
        # p.stations.remote[0].runs = run_list
        # for run_id in run_list:
        #     p.stations.local.run_dict[run_id].input_channels = ["hx", "hy"]
        #     p.stations.local.run_dict[run_id].output_channels = ["ex", "ey", "hz"]

        for decimation in p.decimations:
            decimation.estimator.engine = estimation_engine
            decimation.window.type = "hamming"
            decimation.window.num_samples = num_samples_window
            decimation.window.overlap = num_samples_overlap
            decimation.regression.max_redescending_iterations = 2

    return p, dd_df




def main():
    create_run_config_for_test_case("test1")
    create_run_config_for_test_case("test2")
    create_run_config_for_test_case("test1r2")
    create_run_config_for_test_case("test2r1")

if __name__ == "__main__":
    main()
