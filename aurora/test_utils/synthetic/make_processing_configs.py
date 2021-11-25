from aurora.config.config_creator import ConfigCreator
from aurora.general_helper_functions import BAND_SETUP_PATH
from aurora.general_helper_functions import TEST_PATH

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

    """
    mth5_path = DATA_PATH.joinpath(f"{test_case_id}.h5")

    cc = ConfigCreator(config_path=CONFIG_PATH)
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
        band_setup_file = BAND_SETUP_PATH.joinpath("bs_256_26.cfg")
        num_samples_window = 256
        num_samples_overlap = 64
        config_id = f"{local_station_id}-{matlab_or_fortran}"
    elif matlab_or_fortran == "fortran":
        band_setup_file = BAND_SETUP_PATH.joinpath("bs_test.cfg")
        num_samples_window = 128
        num_samples_overlap = 32
        config_id = f"{local_station_id}-{matlab_or_fortran}"
    else:
        band_setup_file = BAND_SETUP_PATH.joinpath("bs_test.cfg")
        num_samples_window = 128
        num_samples_overlap = 32
        config_id = f"{local_station_id}"

    run_config_path = cc.create_run_config(
        station_id=local_station_id,
        mth5_path=mth5_path,
        sample_rate=1.0,
        num_samples_window=num_samples_window,
        num_samples_overlap=num_samples_overlap,
        config_id=config_id,
        output_channels=["hz", "ex", "ey"],
        reference_station_id=reference_station_id,
        reference_channels=reference_channels,
        band_setup_file = str(band_setup_file),
        estimation_engine=estimation_engine,
    )
    return run_config_path




def main():
    create_run_config_for_test_case("test1")
    create_run_config_for_test_case("test2")
    create_run_config_for_test_case("test1r2")
    create_run_config_for_test_case("test2r1")

if __name__ == "__main__":
    main()
