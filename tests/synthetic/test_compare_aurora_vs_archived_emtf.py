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

import pandas as pd
from aurora.config.metadata.processing import Processing
from aurora.config.config_creator import ConfigCreator
from aurora.general_helper_functions import TEST_PATH
from aurora.pipelines.helpers import initialize_config
from aurora.pipelines.process_mth5 import process_mth5_run
from aurora.pipelines.process_mth5_dev import process_mth5_from_dataset_definition
from aurora.sandbox.io_helpers.zfile_murphy import read_z_file
from aurora.test_utils.synthetic.make_processing_configs_new import create_test_run_config
from aurora.tf_kernel.dataset import DatasetDefinition
from aurora.tf_kernel.helpers import extract_run_summaries_from_mth5s
from aurora.transfer_function.emtf_z_file_helpers import (
    merge_tf_collection_to_match_z_file,
)
from mth5.utils.helpers import initialize_mth5

from make_mth5_from_asc import create_test1_h5
from make_mth5_from_asc import create_test12rr_h5
from plot_helpers_synthetic import plot_rho_phi

SYNTHETIC_PATH = TEST_PATH.joinpath("synthetic")
CONFIG_PATH = SYNTHETIC_PATH.joinpath("config")
DATA_PATH = SYNTHETIC_PATH.joinpath("data")
EMTF_OUTPUT_PATH = SYNTHETIC_PATH.joinpath("emtf_output")
AURORA_RESULTS_PATH = SYNTHETIC_PATH.joinpath("aurora_results")
AURORA_RESULTS_PATH.mkdir(exist_ok=True)

def get_expected_rms_misfit(test_case_id, emtf_version=None):
    expected_rms_misfit = {}
    expected_rms_misfit["rho"] = {}
    expected_rms_misfit["phi"] = {}
    if test_case_id == "test1":
        if emtf_version == "fortran":
            expected_rms_misfit["rho"]["xy"] = 4.406358  #4.380757  # 4.357440
            expected_rms_misfit["phi"]["xy"] = 0.862902  #0.871609  # 0.884601
            expected_rms_misfit["rho"]["yx"] = 3.625859  #3.551043  # 3.501146
            expected_rms_misfit["phi"]["yx"] = 0.840394  #0.812733  # 0.808658
        elif emtf_version == "matlab":
            expected_rms_misfit["rho"]["xy"] = 2.691072
            expected_rms_misfit["phi"]["xy"] = 0.780713
            expected_rms_misfit["rho"]["yx"] = 3.676269
            expected_rms_misfit["phi"]["yx"] = 1.392265

    elif test_case_id == "test2r1":
        expected_rms_misfit["rho"]["xy"] = 3.940519  #3.949857  #3.949919
        expected_rms_misfit["phi"]["xy"] = 0.959861  #0.962837  #0.957675
        expected_rms_misfit["rho"]["yx"] = 4.136467  #4.121772  #4.117700
        expected_rms_misfit["phi"]["yx"] = 1.635570  #1.637581  #1.629026
    return expected_rms_misfit

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
    print(f"expected_rms_rho_{xy_or_yx} {expected_rms_rho}")
    print(f"expected_rms_phi_{xy_or_yx} {expected_rms_phi}")
    assert np.isclose(rho_rms_aurora - expected_rms_rho, 0, atol=rho_tol)
    assert np.isclose(phi_rms_aurora - expected_rms_phi, 0, atol=phi_tol)
    return


def process_synthetic_1_standard(
        processing_config,
        auxilliary_z_file,
        z_file_base,
        ds_df = None,
        expected_rms_misfit={},
        make_rho_phi_plot=True,
        show_rho_phi_plot=False,
        use_subtitle=True,
        emtf_version="matlab",
        **kwargs
    ):
    """

    Parameters
    ----------
    processing_config: str or Path, or a Processing() object
        where the processing configuration file is found
    expected_rms_misfit: dict
        has expected values for the RMS misfits for the TF quantities rho_xy, rho_yx,
        phi_xy, phi_yx. These are used to validate that the processing results don't
        change much.
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

    cond1 = isinstance(processing_config, str)
    cond2 = isinstance(processing_config, Path)
    if (cond1 or cond2):
        print("This needs to be updated to work with new mt_metadata Processing object")
        #load from a json path or string
        config = initialize_config(processing_config)
    elif isinstance(processing_config, Processing):
        config = processing_config
        mth5_path = config.stations.local.mth5_path
    else:
        print(f"processing_config has unexpected type {type(processing_config)}")
        raise Exception

    dataset_definition = DatasetDefinition()
    dataset_definition.df = ds_df

    tf_collection = process_mth5_from_dataset_definition(config,
                                               dataset_definition,
                                               units="MT",
                                               z_file_path=z_file_path)


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
                         show_plot=show_rho_phi_plot,
                         output_path=AURORA_RESULTS_PATH)
    return


def aurora_vs_emtf(test_case_id, emtf_version, auxilliary_z_file, z_file_base, ds_df):
    """

    Parameters
    ----------
    test_case_id: str
        one of ["test1", "test2r1"].  "test1" is associated with single station
        processing. "test2r1" is remote refernce processing
    emtf_version: str
        one of ["fortran", "matlab"]
    auxilliary_z_file: str or pathlib.Path
        points to a .zss, .zrr or .zmm that EMTF produced that will be compared
        against the python aurora output
    z_file_base: str
        This is the z_file that aurora will write its output to



    Returns
    -------

    """
    processing_config = create_test_run_config(test_case_id, ds_df,
                                                      matlab_or_fortran=emtf_version)


    expected_rms_misfit = get_expected_rms_misfit(test_case_id, emtf_version)

    process_synthetic_1_standard(
        processing_config,
        auxilliary_z_file,
        z_file_base,
        ds_df=ds_df,
        expected_rms_misfit=expected_rms_misfit,
        emtf_version=emtf_version,
        make_rho_phi_plot=True,
        show_rho_phi_plot=False,
        use_subtitle=True,
    )
    return


def run_test1(emtf_version, ds_df):
    """

    Parameters
    ----------
    emtf_version
    ds_df

    Returns
    -------

    """
    print(f"Test1 vs {emtf_version}")
    test_case_id = "test1"
    auxilliary_z_file = EMTF_OUTPUT_PATH.joinpath("test1.zss")
    z_file_base = f"{test_case_id}_aurora_{emtf_version}.zss"
    aurora_vs_emtf(test_case_id, emtf_version, auxilliary_z_file, z_file_base, ds_df)
    return

def run_test2r1(ds_df):
    print(f"Test2r1")
    test_case_id = "test2r1"
    emtf_version = "fortran"
    auxilliary_z_file = EMTF_OUTPUT_PATH.joinpath("test2r1.zrr")
    z_file_base = f"{test_case_id}_aurora_{emtf_version}.zrr"
    aurora_vs_emtf(test_case_id, emtf_version, auxilliary_z_file, z_file_base, ds_df)
    return

def make_mth5s():
    mth5_path_1 = create_test1_h5()
    mth5_path_2 = create_test12rr_h5()
    return [mth5_path_1, mth5_path_2]


def test():
    mth5_paths = make_mth5s()
    
    super_summary = extract_run_summaries_from_mth5s([mth5_paths[1],])

    dataset_df = super_summary[super_summary.station_id=="test1"]
    dataset_df["remote"] = False
    run_test1("fortran", dataset_df)
    run_test1("matlab", dataset_df)
    
    dataset_df = super_summary.copy(deep=True)
    dataset_df["remote"] = [True, False]
    run_test2r1(dataset_df)
    print("success")


def main():
    test()


if __name__ == "__main__":
    main()
