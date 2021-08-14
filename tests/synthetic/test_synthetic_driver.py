from pathlib import Path

from aurora.pipelines.process_mth5 import process_mth5_run
from aurora.transfer_function.emtf_z_file_helpers import (
    merge_tf_collection_to_match_z_file,
)
from make_mth5_from_asc import create_mth5_synthetic_file
from make_mth5_from_asc import create_mth5_synthetic_file_for_array
from make_processing_configs import create_run_config_for_test_case
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


def process_synthetic_1_standard(assert_compare_result=True, show_rho_phi_plot=False):
    """
    Just like the normal test runs, but this uses a previously committed json
    file and has a known result.  The results are plotted and stored and
    checked against a standard result calculated originally in August 2021.

    Could probably factor this into its own module and import it.
    Returns
    -------

    """
    test_config = Path("config", "test1_run_config_standard.json")
    z_file_path = Path("test1_aurora.zss")
    z_file_path = z_file_path.absolute()
    run_id = "001"
    tf_collection = process_mth5_run(
        test_config, run_id, units="MT", show_plot=False, z_file_path=z_file_path
    )
    from aurora.sandbox.io_helpers.zfile_murphy import test_reader
    import numpy as np

    aux_data = test_reader()

    (
        aurora_rxy,
        aurora_ryx,
        aurora_pxy,
        aurora_pyx,
    ) = merge_tf_collection_to_match_z_file(aux_data, tf_collection)

    xy_or_yx = "xy"
    rho_rms_aurora = np.sqrt(np.mean((aurora_rxy - 100) ** 2))
    print(f"rho_rms_aurora {rho_rms_aurora}")
    phi_rms_aurora = np.sqrt(np.mean((aurora_pxy - 45) ** 2))
    print(f"phi_rms_aurora {phi_rms_aurora}")
    if assert_compare_result:
        assert np.isclose(rho_rms_aurora - 4.357440052843646, 0)
        assert np.isclose(phi_rms_aurora - 0.8846010204039002, 0)
    rho_rms_emtf = np.sqrt(np.mean((aux_data.rxy - 100) ** 2))
    phi_rms_emtf = np.sqrt(np.mean((aux_data.pxy - 45) ** 2))
    ttl_str = ""
    ttl_str += f"\n rho rms_aurora {rho_rms_aurora:.1f} rms_emtf {rho_rms_emtf:.1f}"
    ttl_str += f"\n phi rms_aurora {phi_rms_aurora:.1f} rms_emtf" f" {phi_rms_emtf:.1f}"
    print(f"{xy_or_yx} rho_rms_aurora {rho_rms_aurora} rho_rms_emtf" f" {rho_rms_emtf}")
    print(f"{xy_or_yx} phi_rms_aurora {phi_rms_aurora} phi_rms_emtf" f" {phi_rms_emtf}")
    if show_rho_phi_plot:
        tf_collection.rho_phi_plot(
            aux_data=aux_data, xy_or_yx=xy_or_yx, ttl_str=ttl_str
        )

    xy_or_yx = "yx"
    rho_rms_aurora = np.sqrt(np.mean((aurora_ryx - 100) ** 2))
    print(f"rho_rms_aurora {rho_rms_aurora}")
    phi_rms_aurora = np.sqrt(np.mean((aurora_pyx - 45) ** 2))
    print(f"phi_rms_aurora {phi_rms_aurora}")
    if assert_compare_result:
        assert np.isclose(rho_rms_aurora - 3.501146955985312, 0)
        assert np.isclose(phi_rms_aurora - 0.8086584775474835, 0)
    rho_rms_emtf = np.sqrt(np.mean((aux_data.ryx - 100) ** 2))
    phi_rms_emtf = np.sqrt(np.mean((aux_data.pyx - 45) ** 2))
    ttl_str = ""
    ttl_str += f"\n rho rms_aurora {rho_rms_aurora:.1f} rms_emtf {rho_rms_emtf:.1f}"
    ttl_str += f"\n phi rms_aurora {phi_rms_aurora:.1f} rms_emtf {phi_rms_emtf:.1f}"
    print(f"{xy_or_yx} rho_rms_aurora {rho_rms_aurora} rho_rms_emtf " f"{rho_rms_emtf}")
    print(f"{xy_or_yx} phi_rms_aurora {phi_rms_aurora} phi_rms_emtf " f"{phi_rms_emtf}")
    if show_rho_phi_plot:
        tf_collection.rho_phi_plot(
            aux_data=aux_data, xy_or_yx=xy_or_yx, ttl_str=ttl_str
        )
    print("OK")
    return


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
    process_synthetic_1_standard(show_rho_phi_plot=False)
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
