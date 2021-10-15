from aurora.general_helper_functions import TEST_PATH
from aurora.pipelines.process_mth5 import process_mth5_run

from make_mth5_from_asc import create_test1_h5
from make_mth5_from_asc import create_test2_h5
from make_mth5_from_asc import create_test12rr_h5
from make_synthetic_processing_configs import create_run_config_for_test_case
from synthetic_station_config import STATION_01_CFG


CONFIG_PATH = TEST_PATH.joinpath("synthetic", "config")


def test_create_mth5():
    create_test1_h5()
    create_test2_h5()
    create_test12rr_h5()


def test_create_run_configs():
    create_run_config_for_test_case("test1")
    create_run_config_for_test_case("test2")
    create_run_config_for_test_case("test12rr")


def process_synthetic_1_underdetermined():
    """
    Just like process_synthetic_1, but the window is ridiculously long so that we
    encounter the underdetermined problem. We actually pass that test but in testing
    I found that at the next band over, which has more data because there are multipe
    FCs the sigma in TRME comes out as negative. see issue #4 and issue #55.
    Returns
    -------

    """
    test_config = CONFIG_PATH.joinpath("test1_run_config_underdetermined.json")
    # test_config = Path("config", "test1_run_config_underdetermined.json")
    run_id = "001"
    process_mth5_run(test_config, run_id, units="MT")


def process_synthetic_1_with_nans():
    """

    Returns
    -------

    """
    test_config = CONFIG_PATH.joinpath("test1_run_config_nan.json")
    #    test_config = Path("config", "test1_run_config_nan.json")
    run_id = "001"
    process_mth5_run(test_config, run_id, units="MT")


def process_synthetic_1():
    test_config = CONFIG_PATH.joinpath("test1_run_config.json")
    # test_config = Path("config", "test1_run_config.json")
    run_id = "001"
    process_mth5_run(test_config, run_id, units="MT")


def process_synthetic_2():
    test_config = CONFIG_PATH.joinpath("test2_run_config.json")
    # test_config = Path("config", "test2_run_config.json")
    run_id = "001"
    process_mth5_run(test_config, run_id, units="MT")


def process_synthetic_rr12():
    test_config = CONFIG_PATH.joinpath("test12rr-RR_test2_run_config.json")
    # test_config = Path("config", "test12rr_run_config.json")
    run_id = STATION_01_CFG["run_id"]
    process_mth5_run(test_config, run_id, units="MT", show_plot=False)


def test_process_mth5():
    # process_synthetic_1_underdetermined()
    # process_synthetic_1_with_nans()
    # create_run_config_for_test_case("test1")
    process_synthetic_1()
    process_synthetic_2()
    process_synthetic_rr12()


def main():
    test_create_mth5()
    test_create_run_configs()
    test_process_mth5()


if __name__ == "__main__":
    main()
