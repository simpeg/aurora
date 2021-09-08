from pathlib import Path

from aurora.pipelines.process_mth5 import process_mth5_run

from make_mth5_from_asc import create_mth5_synthetic_file
from make_mth5_from_asc import create_mth5_synthetic_file_for_array
from make_synthetic_processing_configs import create_run_config_for_test_case
from process_synthetic_data_standard import test_process_synthetic_1_standard
from synthetic_station_config import STATION_01_CFG
from synthetic_station_config import STATION_02_CFG


def test_create_mth5():
    create_mth5_synthetic_file(STATION_01_CFG, plot=False)
    create_mth5_synthetic_file(STATION_02_CFG)
    create_mth5_synthetic_file_for_array([STATION_01_CFG, STATION_02_CFG])


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
    test_config = Path("config", "test1_run_config_underdetermined.json")
    run_id = "001"
    process_mth5_run(test_config, run_id, units="MT")


def process_synthetic_1_with_nans():
    """

    Returns
    -------

    """
    test_config = Path("config", "test1_run_config_nan.json")
    run_id = "001"
    process_mth5_run(test_config, run_id, units="MT")


def process_synthetic_1():
    test_config = Path("config", "test1_run_config.json")
    run_id = "001"
    process_mth5_run(test_config, run_id, units="MT")


def process_synthetic_2():
    test_config = Path("config", "test2_run_config.json")
    run_id = "001"
    process_mth5_run(test_config, run_id, units="MT")


def process_synthetic_rr12():
    test_config = Path("config", "test12rr_run_config.json")
    run_id = STATION_01_CFG["run_id"]
    process_mth5_run(test_config, run_id, units="MT")


def test_process_mth5():
    # create_mth5_synthetic_file(STATION_01_CFG, plot=False)
    # process_synthetic_1_underdetermined()
    # process_synthetic_1_with_nans()
    test_process_synthetic_1_standard(
        assert_compare_result=False,
        make_rho_phi_plot=True,
        show_rho_phi_plot=False,
        use_subtitle=True,
        compare_against="matlab",
    )
    test_process_synthetic_1_standard(
        assert_compare_result=True,
        make_rho_phi_plot=True,
        show_rho_phi_plot=False,
        use_subtitle=True,
        compare_against="fortran",
    )
    # create_run_config_for_test_case("test1")
    # process_synthetic_1()
    # process_synthetic_2()
    # process_synthetic_rr12()


def main():
    # test_create_mth5()
    # test_create_run_configs()
    test_process_mth5()


if __name__ == "__main__":
    main()
