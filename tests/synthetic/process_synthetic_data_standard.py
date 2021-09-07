import numpy as np
from pathlib import Path

from aurora.config.processing_config import RunConfig
from aurora.general_helper_functions import TEST_PATH
from aurora.general_helper_functions import BAND_SETUP_PATH
from aurora.pipelines.process_mth5 import process_mth5_run
from aurora.sandbox.io_helpers.zfile_murphy import read_z_file
from aurora.transfer_function.emtf_z_file_helpers import (
    merge_tf_collection_to_match_z_file,
)

SYNTHETIC_PATH = TEST_PATH.joinpath("synthetic")
CONFIG_PATH = SYNTHETIC_PATH.joinpath("config")


def create_config_file():
    pass


def test_process_synthetic_1_standard(
    assert_compare_result=True,
    make_rho_phi_plot=True,
    show_rho_phi_plot=False,
    use_subtitle=True,
    compare_against="matlab",  # "fortran" or "matlab"
):
    """
    Just like the normal test runs, but this uses a previously committed json file
    and has a known result.  The results are plotted and stored and checked against a
    standard result calculated originally in August 2021.

    Want to support two cases of comparison here.  In one case we compare against
    the committed .zss file in the EMTF repository, and in the other case we compare
    against a committed .mat file created by the matlab codes.


    Returns
    -------

    """
    if compare_against == "fortran":
        test_config = CONFIG_PATH.joinpath("test1_run_config_standard.json")
        # test_config = Path("config", "test1_run_config_standard.json")
        auxilliary_z_file = TEST_PATH.joinpath("synthetic", "emtf_output", "test1.zss")
        expected_rms_rho_xy = 4.357440
        expected_rms_phi_xy = 0.884601
    elif compare_against == "matlab":
        auxilliary_z_file = TEST_PATH.joinpath(
            "synthetic", "emtf_output", "from_matlab_256_26.zss"
        )
        test_config = CONFIG_PATH.joinpath("test1_run_config_standard.json")
        config = RunConfig()
        config.from_json(test_config)
        band_setup_file = BAND_SETUP_PATH.joinpath("bs_256_26.cfg")
        for i_dec in config.decimation_level_ids:
            config.decimation_level_configs[i_dec].num_samples_window = 256
            config.decimation_level_configs[i_dec].num_samples_overlap = 64
            config.decimation_level_configs[
                i_dec
            ].emtf_band_setup_file = band_setup_file
        print("overwrite")
        test_config = config
    # </MATLAB>

    z_file_path = Path("test1_aurora.zss")
    z_file_path = z_file_path.absolute()
    run_id = "001"
    tf_collection = process_mth5_run(
        test_config, run_id, units="MT", show_plot=False, z_file_path=z_file_path
    )

    aux_data = read_z_file(auxilliary_z_file)

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
        assert np.isclose(rho_rms_aurora - expected_rms_rho_xy, 0, atol=1e-4)
        assert np.isclose(phi_rms_aurora - expected_rms_phi_xy, 0, atol=1e-4)
    rho_rms_emtf = np.sqrt(np.mean((aux_data.rxy - 100) ** 2))
    phi_rms_emtf = np.sqrt(np.mean((aux_data.pxy - 45) ** 2))
    ttl_str = ""
    if use_subtitle:
        ttl_str += f"\n rho rms_aurora {rho_rms_aurora:.1f} rms_emtf {rho_rms_emtf:.1f}"
        ttl_str += (
            f"\n phi rms_aurora {phi_rms_aurora:.1f} rms_emtf" f" {phi_rms_emtf:.1f}"
        )
    print(f"{xy_or_yx} rho_rms_aurora {rho_rms_aurora} rho_rms_emtf" f" {rho_rms_emtf}")
    print(f"{xy_or_yx} phi_rms_aurora {phi_rms_aurora} phi_rms_emtf" f" {phi_rms_emtf}")
    if make_rho_phi_plot:
        tf_collection.rho_phi_plot(
            aux_data=aux_data,
            xy_or_yx=xy_or_yx,
            ttl_str=ttl_str,
            show=show_rho_phi_plot,
        )

    xy_or_yx = "yx"
    rho_rms_aurora = np.sqrt(np.mean((aurora_ryx - 100) ** 2))
    print(f"rho_rms_aurora {rho_rms_aurora}")
    phi_rms_aurora = np.sqrt(np.mean((aurora_pyx - 45) ** 2))
    print(f"phi_rms_aurora {phi_rms_aurora}")
    if assert_compare_result:
        assert np.isclose(rho_rms_aurora - 3.501146, 0, atol=2e-3)
        assert np.isclose(phi_rms_aurora - 0.808658, 0, atol=1e-3)
    rho_rms_emtf = np.sqrt(np.mean((aux_data.ryx - 100) ** 2))
    phi_rms_emtf = np.sqrt(np.mean((aux_data.pyx - 45) ** 2))
    ttl_str = ""
    if use_subtitle:
        ttl_str += f"\n rho rms_aurora {rho_rms_aurora:.1f} rms_emtf {rho_rms_emtf:.1f}"
        ttl_str += f"\n phi rms_aurora {phi_rms_aurora:.1f} rms_emtf {phi_rms_emtf:.1f}"
    print(f"{xy_or_yx} rho_rms_aurora {rho_rms_aurora} rho_rms_emtf " f"{rho_rms_emtf}")
    print(f"{xy_or_yx} phi_rms_aurora {phi_rms_aurora} phi_rms_emtf " f"{phi_rms_emtf}")
    if make_rho_phi_plot:
        tf_collection.rho_phi_plot(
            aux_data=aux_data,
            xy_or_yx=xy_or_yx,
            ttl_str=ttl_str,
            show=show_rho_phi_plot,
        )

    return


def main():
    create_config_file()
    test_process_synthetic_1_standard(
        assert_compare_result=True, compare_against="fortran"
    )
    test_process_synthetic_1_standard(
        assert_compare_result=False, compare_against="matlab"
    )
    print("success")


if __name__ == "__main__":
    main()
