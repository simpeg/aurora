"""
Working Notes:
1. Need to create a processing config for the remote reference case and run the RR
processing.  Use the examples from test_synthetic_driver
2. Get baseline RR values and use them in an assertion test here

3. Check if the previously committed json config is being used by the single station
tests, it looks like maybe it is not anymore.
It is being used in the stft test, but maybe that can be made to depend on the
FORTRAN config file

"""
import numpy as np
from pathlib import Path

from aurora.general_helper_functions import TEST_PATH
from aurora.pipelines.process_mth5 import process_mth5_run
from aurora.sandbox.io_helpers.zfile_murphy import read_z_file
from aurora.test_utils.synthetic.make_processing_configs import create_test_run_config
from aurora.transfer_function.emtf_z_file_helpers import (
    merge_tf_collection_to_match_z_file,
)

from make_mth5_from_asc import create_test1_h5
from make_mth5_from_asc import create_test12rr_h5

SYNTHETIC_PATH = TEST_PATH.joinpath("synthetic")
CONFIG_PATH = SYNTHETIC_PATH.joinpath("config")
DATA_PATH = SYNTHETIC_PATH.joinpath("data")
EMTF_OUTPUT_PATH = SYNTHETIC_PATH.joinpath("emtf_output")
AURORA_RESULTS_PATH = SYNTHETIC_PATH.joinpath("aurora_results")
AURORA_RESULTS_PATH.mkdir(exist_ok=True)

EXPECTED_RMS_MISFIT = {}
EXPECTED_RMS_MISFIT["test1"] = {}
EXPECTED_RMS_MISFIT["test1"]["rho"] = {}
EXPECTED_RMS_MISFIT["test1"]["phi"] = {}
EXPECTED_RMS_MISFIT["test1"]["rho"]["xy"] = 4.406358  #4.380757  # 4.357440
EXPECTED_RMS_MISFIT["test1"]["phi"]["xy"] = 0.862902  #0.871609  # 0.884601
EXPECTED_RMS_MISFIT["test1"]["rho"]["yx"] = 3.625859  #3.551043  # 3.501146
EXPECTED_RMS_MISFIT["test1"]["phi"]["yx"] = 0.840394  #0.812733  # 0.808658
EXPECTED_RMS_MISFIT["test2r1"] = {}
EXPECTED_RMS_MISFIT["test2r1"]["rho"] = {}
EXPECTED_RMS_MISFIT["test2r1"]["phi"] = {}
EXPECTED_RMS_MISFIT["test2r1"]["rho"]["xy"] = 3.940519  #3.949857  #3.949919
EXPECTED_RMS_MISFIT["test2r1"]["phi"]["xy"] = 0.959861  #0.962837  #0.957675
EXPECTED_RMS_MISFIT["test2r1"]["rho"]["yx"] = 4.136467  #4.121772  #4.117700
EXPECTED_RMS_MISFIT["test2r1"]["phi"]["yx"] = 1.635570  #1.637581  #1.629026

def compute_rms(rho, phi, model_rho_a=100.0, model_phi=45.0, verbose=False):
    """
    This function being used to make comparative plots for synthetic data.  Could be 
    used in general to compare different processing results.  For example by replacing 
    model_rho_a and model_phi with other processing results, or other (
    non-uniform) model results. 
    
    Parameters
    ----------
    rho: numpy.ndarray
        1D array of computed apparent resistivities (expected in Ohmm)
    phi: numpy.ndarrayx
        1D array of computed phases (expected in degrees)
    model_rho_a: float or numpy array
        if numpy array must be the same shape as rho
    model_phi: float or numpy array
        if numpy array must be the same shape as phi.
    Returns
    -------
    rho_rms: float
        rms misfit between the model apparent resistivity and the computed resistivity
    phi_rms: float
        rms misfit between the model phase (or phases) and the computed phase
    """
    rho_rms = np.sqrt(np.mean((rho - model_rho_a) ** 2))
    phi_rms = np.sqrt(np.mean((phi - model_phi) ** 2))
    if verbose:
        print(f"rho_rms = {rho_rms}")
        print(f"phi_rms = {phi_rms}")
    return rho_rms, phi_rms

def make_subtitle(rho_rms_aurora, rho_rms_emtf,
                  phi_rms_aurora, phi_rms_emtf,
                  matlab_or_fortran, ttl_str=""):
    """
    
    Parameters
    ----------
    rho_rms_aurora: float
        rho_rms for aurora data differenced against a model. comes from compute_rms
    rho_rms_emtf:
        rho_rms for emtf data differenced against a model. comes from compute_rms
    phi_rms_aurora:
        phi_rms for aurora data differenced against a model. comes from compute_rms
    phi_rms_emtf:
        phi_rms for emtf data differenced against a model. comes from compute_rms
    matlab_or_fortran: str
        "matlab" or "fortran".  A specifer for the version of emtf.
    ttl_str: str
        string onto which we add the subtitle

    Returns
    -------
    ttl_str: str
        Figure title with subtitle

    """
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
    """
    
    Parameters
    ----------
    local_station_id: str
        station label
    reference_station_id: str
        remote reference station label
    xy_or_yx: str
        mode: "xy" or "yx"
    matlab_or_fortran: str
        "matlab" or "fortran".  A specifer for the version of emtf.

    Returns
    -------
    figure_basename: str
        filename for figure

    """
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
    """
    Could be made into a method of TF Collection
    Parameters
    ----------
    xy_or_yx
    tf_collection
    rho_rms_aurora
    rho_rms_emtf
    phi_rms_aurora
    phi_rms_emtf
    matlab_or_fortran
    aux_data
    use_subtitle
    show_plot

    Returns
    -------

    """
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


def assert_rms_misfit_ok(expected_rms_misfit, xy_or_yx, rho_rms_aurora,
                         phi_rms_aurora, rho_tol=1e-4, phi_tol=1e-4):
    """

    Parameters
    ----------
    expected_rms_misfit: dictionary
        precomputed RMS misfits for test data in rho and phi
    xy_or_yx: str
        mode
    rho_rms_aurora: float
    phi_rms_aurora: float

    Returns
    -------

    """
    expected_rms_rho = expected_rms_misfit['rho'][xy_or_yx]
    expected_rms_phi = expected_rms_misfit['phi'][xy_or_yx]
    print(f"expected_rms_rho_xy {expected_rms_rho}")
    print(f"expected_rms_rho_xy {expected_rms_phi}")
    assert np.isclose(rho_rms_aurora - expected_rms_rho, 0, atol=rho_tol)
    assert np.isclose(phi_rms_aurora - expected_rms_phi, 0, atol=phi_tol)
    return


def process_synthetic_1_standard(
        processing_config_path,
        auxilliary_z_file,
        z_file_base,
        expected_rms_misfit=None,
        make_rho_phi_plot=True,
        show_rho_phi_plot=False,
        use_subtitle=True,
        emtf_version="matlab",
    ):
    """

    Parameters
    ----------
    processing_config_path: str or Path
        where the processing configuration file is found
    expected_rms_misfit: dict
        see description in assert_rms_misfit_ok
    make_rho_phi_plot
    show_rho_phi_plot
    use_subtitle
    emtf_version: string
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
    z_file_path = AURORA_RESULTS_PATH.joinpath(z_file_base)

    run_id = "001"
    tf_collection = process_mth5_run(
        processing_config_path, run_id, units="MT", show_plot=False,
        z_file_path=z_file_path
    )
    tf_collection._merge_decimation_levels()

    #END THE NORMAL PROCESSING TEST

    aux_data = read_z_file(auxilliary_z_file)

    aurora_rho_phi = merge_tf_collection_to_match_z_file(aux_data, tf_collection)

    for xy_or_yx in ["xy", "yx"]:
        aurora_rho = aurora_rho_phi["rho"][xy_or_yx]
        aurora_phi = aurora_rho_phi["phi"][xy_or_yx]
        aux_rho = aux_data.rho(xy_or_yx)
        aux_phi = aux_data.phi(xy_or_yx)
        rho_rms_aurora, phi_rms_aurora = compute_rms(aurora_rho, aurora_phi, verbose=True)
        rho_rms_emtf, phi_rms_emtf = compute_rms(aux_rho, aux_phi)

        if expected_rms_misfit is not None:
            assert_rms_misfit_ok(expected_rms_misfit,
                                 xy_or_yx,
                                 rho_rms_aurora,
                                 phi_rms_aurora)

        if make_rho_phi_plot:
            plot_rho_phi(xy_or_yx,
                         tf_collection,
                         rho_rms_aurora,
                         rho_rms_emtf,
                         phi_rms_aurora,
                         phi_rms_emtf,
                         emtf_version,
                         aux_data=aux_data,
                         use_subtitle=use_subtitle,
                         show_plot=show_rho_phi_plot)
    return


def aurora_vs_emtf(test_case_id, emtf_version, auxilliary_z_file, z_file_base,
                   expected_rms_misfit=None):
    processing_config_path = create_test_run_config(test_case_id,
                                                    matlab_or_fortran=emtf_version)
    process_synthetic_1_standard(
        processing_config_path,
        auxilliary_z_file,
        z_file_base,
        expected_rms_misfit=expected_rms_misfit,
        emtf_version=emtf_version,
        make_rho_phi_plot=True,
        show_rho_phi_plot=False,
        use_subtitle=True,
    )


def run_test1(emtf_version, expected_rms_misfit=None):
    print(f"Test1 vs {emtf_version}")
    test_case_id = "test1"
    auxilliary_z_file = EMTF_OUTPUT_PATH.joinpath("test1.zss")
    z_file_base = f"{test_case_id}_aurora_{emtf_version}.zss"
    aurora_vs_emtf(test_case_id, emtf_version, auxilliary_z_file, z_file_base,
                   expected_rms_misfit=expected_rms_misfit)
    return

def run_test2r1():
    print(f"Test2r1")
    test_case_id = "test2r1"
    emtf_version = "fortran"
    auxilliary_z_file = EMTF_OUTPUT_PATH.joinpath("test2r1.zrr")
    z_file_base = f"{test_case_id}_aurora_{emtf_version}.zrr"
    aurora_vs_emtf(test_case_id, emtf_version, auxilliary_z_file, z_file_base,
                   expected_rms_misfit=EXPECTED_RMS_MISFIT[test_case_id])
    return


def test():
    create_test1_h5()
    create_test12rr_h5()
    run_test1("fortran", expected_rms_misfit=EXPECTED_RMS_MISFIT["test1"])
    run_test1("matlab")
    run_test2r1()
    print("success")


def main():
    test()


if __name__ == "__main__":
    main()
