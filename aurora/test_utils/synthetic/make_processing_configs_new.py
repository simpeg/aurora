from aurora.config import BANDS_DEFAULT_FILE
from aurora.config import BANDS_256_FILE
from aurora.config.config_creator import ConfigCreator
from aurora.general_helper_functions import TEST_PATH

CONFIG_PATH = TEST_PATH.joinpath("synthetic", "config")


def create_test_run_config(test_case_id, dd_df, matlab_or_fortran=""):
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

    
    """
    estimation_engine = "RME"
    local_station_id = test_case_id
    reference_station_id = ""
    reference_channels = []
    if test_case_id == "test1r2":
        estimation_engine = "RME_RR"
        reference_channels = ["hx", "hy"]
        local_station_id = "test1"
        reference_station_id = "test2"
    if test_case_id == "test2r1":
        estimation_engine = "RME_RR"
        reference_channels = ["hx", "hy"]
        local_station_id = "test2"
        reference_station_id = "test1"

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

    cc = ConfigCreator(config_path=CONFIG_PATH)
    
    if test_case_id=="test1":
        p = cc.create_run_processing_object(emtf_band_file=emtf_band_setup_file)
        p.id = config_id
        p.stations.from_dataset_dataframe(dd_df)

        for decimation in p.decimations:
            decimation.estimator.engine = estimation_engine
            decimation.window.type = "hamming"
            decimation.window.num_samples = num_samples_window
            decimation.window.overlap = num_samples_overlap
            decimation.regression.max_redescending_iterations = 2

        p.drop_reference_channels()
        cc.to_json(p)
    elif test_case_id=="test2r1":
        p = cc.create_run_processing_object(emtf_band_file=emtf_band_setup_file)
        p.id = config_id
        p.stations.from_dataset_dataframe(dd_df)

        for decimation in p.decimations:
            decimation.estimator.engine = estimation_engine
            decimation.window.type = "hamming"
            decimation.window.num_samples = num_samples_window
            decimation.window.overlap = num_samples_overlap
            decimation.regression.max_redescending_iterations = 2

    return p




def main():
    create_run_config_for_test_case("test1")
    create_run_config_for_test_case("test2")
    create_run_config_for_test_case("test1r2")
    create_run_config_for_test_case("test2r1")

if __name__ == "__main__":
    main()
