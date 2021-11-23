"""
Working Notes:
1. Need to create a processing config for the remote reference case (done but not
tested)
2. need to run the rr processing
3. Check if the previously committed json config is being used by the single station
tests, it looks like maybe it is not anymore.
It is being used in the stft test, but maybe that can be made to depend on the
FORTRAN config file
4. create_run_config_for_test_case() is in make_synthetic_processing_configs.
Why are there two places?  Can we replace
create_config_file(matlab_or_fortran, SS_or_RR="SS") here with that version?
"""
import numpy as np
from pathlib import Path

from aurora.config.config_creator import ConfigCreator
from aurora.general_helper_functions import TEST_PATH
from aurora.general_helper_functions import BAND_SETUP_PATH
from aurora.pipelines.process_mth5 import process_mth5_run
from aurora.sandbox.io_helpers.zfile_murphy import read_z_file
from aurora.transfer_function.emtf_z_file_helpers import (
    merge_tf_collection_to_match_z_file,
)

from make_mth5_from_asc import create_test1_h5
from make_mth5_from_asc import create_test12rr_h5
from make_synthetic_processing_configs import create_run_config_for_test_case

SYNTHETIC_PATH = TEST_PATH.joinpath("synthetic")
CONFIG_PATH = SYNTHETIC_PATH.joinpath("config")
DATA_PATH = SYNTHETIC_PATH.joinpath("data")
AURORA_RESULTS_PATH = SYNTHETIC_PATH.joinpath("aurora_results")
AURORA_RESULTS_PATH.mkdir(exist_ok=True)



def compute_rms(rho, phi, model_rho_a=100.0, model_phi=45.0):
    rho_rms = np.sqrt(np.mean((rho - model_rho_a) ** 2))
    phi_rms = np.sqrt(np.mean((phi - model_phi) ** 2))
    return rho_rms, phi_rms

def make_subtitle(rho_rms_aurora, rho_rms_emtf,
                  phi_rms_aurora, phi_rms_emtf,
                  matlab_or_fortran, ttl_str=""):

    ttl_str += (
        f"\n rho rms_aurora {rho_rms_aurora:.1f} rms_{matlab_or_fortran}"
        f" {rho_rms_emtf:.1f}"
    )
    ttl_str += (
        f"\n phi rms_aurora {phi_rms_aurora:.1f} rms_{matlab_or_fortran}"
        f" {phi_rms_emtf:.1f}"
    )
    return ttl_str


def make_figure_basename(local_station_id,
                         reference_station_id,
                         xy_or_yx,
                         matlab_or_fortran):
    station_string = f"{local_station_id}"
    if reference_station_id:
        station_string = f"{station_string}_rr{reference_station_id}"
    figure_basename = (
        f"synthetic_{station_string}_{xy_or_yx}_{matlab_or_fortran}.png"
    )
    return figure_basename

def plot_rho_phi(xy_or_yx,
                 tf_collection,
                 rho_rms_aurora,
                 rho_rms_emtf,
                 phi_rms_aurora,
                 phi_rms_emtf,
                 matlab_or_fortran,
                 aux_data=None,
                 use_subtitle=True,
                 show_plot=False):
    ttl_str = ""
    if use_subtitle:
        ttl_str = make_subtitle(rho_rms_aurora, rho_rms_emtf,
                                phi_rms_aurora, phi_rms_emtf,
                                matlab_or_fortran)

    figure_basename = make_figure_basename(tf_collection.local_station_id,
                                           tf_collection.reference_station_id,
                                           xy_or_yx,
                                           matlab_or_fortran)
    tf_collection.rho_phi_plot(
        aux_data=aux_data,
        xy_or_yx=xy_or_yx,
        ttl_str=ttl_str,
        show=show_plot,
        figure_basename=figure_basename,
        figure_path=AURORA_RESULTS_PATH,
    )
    return




def process_synthetic_1_standard(
    assert_compare_result=True,
    make_rho_phi_plot=True,
    show_rho_phi_plot=False,
    use_subtitle=True,
    compare_against="matlab",
):
    """

    Parameters
    ----------
    assert_compare_result
    make_rho_phi_plot
    show_rho_phi_plot
    use_subtitle
    compare_against: string
        "fortran" or "matlab"

    Returns
    -------


    Just like the normal test runs, but this uses a previously committed json file
    and has a known result.  The results are plotted and stored and checked against a
    standard result calculated originally in August 2021.

    Want to support two cases of comparison here.  In one case we compare against
    the committed .zss file in the EMTF repository, and in the other case we compare
    against a committed .mat file created by the matlab codes.

    Note that the comparison values got slightly worse since the original commit.
    It turns out that we can recover the original values by setting beta to the old
    formula, where beta is .8843, not .7769.

    Returns
    -------

    """
    test_config = CONFIG_PATH.joinpath(f"test1-{compare_against}_run_config.json")
    if compare_against == "fortran":
        auxilliary_z_file = SYNTHETIC_PATH.joinpath("emtf_output", "test1.zss")
        expected_rms_rho_xy = 4.380757  # 4.357440
        expected_rms_phi_xy = 0.871609  # 0.884601
        expected_rms_rho_yx = 3.551043  # 3.501146
        expected_rms_phi_yx = 0.812733  # 0.808658
    elif compare_against == "matlab":
        auxilliary_z_file = SYNTHETIC_PATH.joinpath("emtf_output",
                                                    "from_matlab_256_26.zss")


    z_file_base = f"test1_aurora_{compare_against}.zss"
    z_file_path = AURORA_RESULTS_PATH.joinpath(z_file_base)

    run_id = "001"
    tf_collection = process_mth5_run(
        test_config, run_id, units="MT", show_plot=False, z_file_path=z_file_path
    )
    tf_collection._merge_decimation_levels()

    #END THE NORMAL PROCESSING TEST

    aux_data = read_z_file(auxilliary_z_file)

    (
        aurora_rxy,
        aurora_ryx,
        aurora_pxy,
        aurora_pyx,
    ) = merge_tf_collection_to_match_z_file(aux_data, tf_collection)

    xy_or_yx = "xy"
    rho_rms_aurora, phi_rms_aurora = compute_rms(aurora_rxy, aurora_pxy)
    rho_rms_emtf, phi_rms_emtf = compute_rms(aux_data.rxy, aux_data.pxy)

    if assert_compare_result:
        print(f"expected_rms_rho_xy {expected_rms_rho_xy}")
        print(f"expected_rms_phi_xy {expected_rms_phi_xy}")
        assert np.isclose(rho_rms_aurora - expected_rms_rho_xy, 0, atol=1e-4)
        assert np.isclose(phi_rms_aurora - expected_rms_phi_xy, 0, atol=1e-4)

    if make_rho_phi_plot:
        plot_rho_phi(xy_or_yx,
                     tf_collection,
                     rho_rms_aurora,
                     rho_rms_emtf,
                     phi_rms_aurora,
                     phi_rms_emtf,
                     compare_against,
                     aux_data=aux_data,
                     use_subtitle=use_subtitle,
                     show_plot=show_rho_phi_plot)

    xy_or_yx = "yx"
    rho_rms_aurora, phi_rms_aurora = compute_rms(aurora_ryx, aurora_pyx)
    rho_rms_emtf, phi_rms_emtf = compute_rms(aux_data.ryx, aux_data.pyx)

    if assert_compare_result:
        print(f"expected_rms_rho_yx {expected_rms_rho_yx}")
        print(f"expected_rms_phi_yx {expected_rms_phi_yx}")
        assert np.isclose(rho_rms_aurora - expected_rms_rho_yx, 0, atol=2e-3)
        assert np.isclose(phi_rms_aurora - expected_rms_phi_yx, 0, atol=1e-3)

    if make_rho_phi_plot:
        plot_rho_phi(xy_or_yx,
                     tf_collection,
                     rho_rms_aurora,
                     rho_rms_emtf,
                     phi_rms_aurora,
                     phi_rms_emtf,
                     compare_against,
                     aux_data=aux_data,
                     use_subtitle=use_subtitle,
                     show_plot=show_rho_phi_plot)

    return


def aurora_vs_emtf(test_case_id, matlab_or_fortran, assert_compare=False):
    create_run_config_for_test_case(test_case_id, matlab_or_fortran=matlab_or_fortran)
    process_synthetic_1_standard(
        assert_compare_result=assert_compare,
        compare_against=matlab_or_fortran,
        make_rho_phi_plot=True,
        show_rho_phi_plot=False,
        use_subtitle=True,
    )




def test():
    create_test1_h5()
    create_test12rr_h5()
    aurora_vs_emtf("test1", "fortran", assert_compare=True)
    aurora_vs_emtf("test1", "matlab", assert_compare=False)
    print("success")


def main():
    test()


if __name__ == "__main__":
    main()
