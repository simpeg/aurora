import numpy as np
from pathlib import Path

from aurora.config.config_creator_dev import ConfigCreator
from aurora.general_helper_functions import TEST_PATH
from aurora.general_helper_functions import BAND_SETUP_PATH
from aurora.pipelines.process_mth5 import process_mth5_run
from aurora.sandbox.io_helpers.zfile_murphy import read_z_file
from aurora.transfer_function.emtf_z_file_helpers import (
    merge_tf_collection_to_match_z_file,
)

SYNTHETIC_PATH = TEST_PATH.joinpath("synthetic")
CONFIG_PATH = SYNTHETIC_PATH.joinpath("config")
DATA_PATH = SYNTHETIC_PATH.joinpath("data")


def create_config_file(matlab_or_fortran):
    cc = ConfigCreator(config_path=CONFIG_PATH)
    mth5_path = DATA_PATH.joinpath("test1.h5")
    config_id = f"test1-{matlab_or_fortran}"
    if matlab_or_fortran == "matlab":
        band_setup_file = BAND_SETUP_PATH.joinpath("bs_256_26.cfg")
        num_samples_window = 256
        num_samples_overlap = 64
    elif matlab_or_fortran == "fortran":
        band_setup_file = BAND_SETUP_PATH.joinpath("bs_test.cfg")
        num_samples_window = 128
        num_samples_overlap = 32

    run_config_path = cc.create_run_config(
        station_id="test1",
        mth5_path=mth5_path,
        sample_rate=1.0,
        num_samples_window=num_samples_window,
        num_samples_overlap=num_samples_overlap,
        band_setup_file=str(band_setup_file),
        config_id=config_id,
        output_channels=["hz", "ex", "ey"],
    )
    return run_config_path


def process_synthetic_1_standard(
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
    test_config = CONFIG_PATH.joinpath(f"test1-{compare_against}_run_config.json")
    if compare_against == "fortran":
        auxilliary_z_file = TEST_PATH.joinpath("synthetic", "emtf_output", "test1.zss")
        expected_rms_rho_xy = 4.357440
        expected_rms_phi_xy = 0.884601
    elif compare_against == "matlab":
        auxilliary_z_file = TEST_PATH.joinpath(
            "synthetic", "emtf_output", "from_matlab_256_26.zss"
        )

    # </MATLAB>
    z_file_path = Path(f"test1_aurora_{compare_against}.zss")
    # z_file_path = Path("test1_aurora.zss")
    z_file_path = z_file_path.absolute()
    run_id = "001"
    tf_collection = process_mth5_run(
        test_config, run_id, units="MT", show_plot=False, z_file_path=z_file_path
    )
    tf_collection.merge_decimation_levels()
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


def compare_vs_fortran_output():
    compare_against = "fortran"
    create_config_file(compare_against)
    process_synthetic_1_standard(
        assert_compare_result=True,
        compare_against=compare_against,
        make_rho_phi_plot=True,
        show_rho_phi_plot=False,
        use_subtitle=True,
    )


def compare_vs_matlab_output():
    compare_against = "matlab"
    create_config_file(compare_against)
    process_synthetic_1_standard(
        assert_compare_result=False,
        compare_against=compare_against,
        make_rho_phi_plot=True,
        show_rho_phi_plot=False,
        use_subtitle=True,
    )


def main():
    compare_vs_fortran_output()
    compare_vs_matlab_output()
    print("success")


if __name__ == "__main__":
    main()
