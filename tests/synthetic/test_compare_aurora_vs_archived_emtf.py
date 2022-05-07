from pathlib import Path

from aurora.general_helper_functions import TEST_PATH
from aurora.sandbox.io_helpers.zfile_murphy import read_z_file
from aurora.test_utils.synthetic.make_mth5_from_asc import create_test1_h5
from aurora.test_utils.synthetic.make_mth5_from_asc import create_test2_h5
from aurora.test_utils.synthetic.make_mth5_from_asc import create_test12rr_h5
from aurora.test_utils.synthetic.make_processing_configs import create_test_run_config
from aurora.test_utils.synthetic.processing_helpers import process_sythetic_data
from aurora.test_utils.synthetic.rms_helpers import assert_rms_misfit_ok
from aurora.test_utils.synthetic.rms_helpers import compute_rms
from aurora.test_utils.synthetic.rms_helpers import get_expected_rms_misfit
from aurora.tf_kernel.dataset import DatasetDefinition
from aurora.tf_kernel.helpers import extract_run_summaries_from_mth5s
from aurora.transfer_function.emtf_z_file_helpers import (
    merge_tf_collection_to_match_z_file,
)


from plot_helpers_synthetic import plot_rho_phi

SYNTHETIC_PATH = TEST_PATH.joinpath("synthetic")
CONFIG_PATH = SYNTHETIC_PATH.joinpath("config")
DATA_PATH = SYNTHETIC_PATH.joinpath("data")
EMTF_OUTPUT_PATH = SYNTHETIC_PATH.joinpath("emtf_output")
AURORA_RESULTS_PATH = SYNTHETIC_PATH.joinpath("aurora_results")
AURORA_RESULTS_PATH.mkdir(exist_ok=True)





def aurora_vs_emtf(test_case_id,
                   emtf_version,
                   auxilliary_z_file,
                   z_file_base,
                   ds_df,
                   make_rho_phi_plot=True,
                   show_rho_phi_plot=False,
                   use_subtitle=True,):
    """
    ToDo: Consider storing the processing config for this case as a json file,
    committed with the code.

    Just like a normal test of processing synthetic data, but this uses a
    known processing configuration and has a known result.  The results are plotted and
    stored and checked against a standard result calculated originally in August 2021.

    There are two cases of comparisons here.  In one case we compare against
    the committed .zss file in the EMTF repository, and in the other case we compare
    against a committed .mat file created by the matlab codes.

    Note that the comparison values got slightly worse since the original commit.
    It turns out that we can recover the original values by setting beta to the old
    formula, where beta is .8843, not .7769.

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
    ds_df
    make_rho_phi_plot: bool
    show_rho_phi_plot: bool
    use_subtitle: bool


    Returns
    -------

    """
    dataset_definition = DatasetDefinition()
    dataset_definition.df = ds_df
    processing_config = create_test_run_config(test_case_id, ds_df,
                                                      matlab_or_fortran=emtf_version)


    expected_rms_misfit = get_expected_rms_misfit(test_case_id, emtf_version)
    z_file_path  = AURORA_RESULTS_PATH.joinpath(z_file_base)

    tf_collection = process_sythetic_data(processing_config,
                                          dataset_definition,
                                          z_file_path=z_file_path)

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


def run_test1(emtf_version, ds_df):
    """

    Parameters
    ----------
    emtf_version : string
        "matlab", or "fortran"
    ds_df : pandas.DataFrame
        Basically a run_summary dataframe

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
    """

    Parameters
    ----------
    ds_df : pandas.DataFrame
        Basically a run_summary dataframe
    Returns
    -------

    """
    print(f"Test2r1")
    test_case_id = "test2r1"
    emtf_version = "fortran"
    auxilliary_z_file = EMTF_OUTPUT_PATH.joinpath("test2r1.zrr")
    z_file_base = f"{test_case_id}_aurora_{emtf_version}.zrr"
    aurora_vs_emtf(test_case_id, emtf_version, auxilliary_z_file, z_file_base, ds_df)
    return

def make_mth5s(merged=True):
    """
    Returns
    -------
    mth5_paths: list of Path objs or str(Path)
    """
    if merged:
        mth5_path = create_test12rr_h5()
        mth5_paths = [mth5_path,]
    else:
        mth5_path_1 = create_test1_h5()
        mth5_path_2 = create_test2_h5()
        mth5_paths = [mth5_path_1, mth5_path_2]
    return mth5_paths


def test_pipeline(merged=True):
    """

    Parameters
    ----------
    merged: bool
        If true, summarise two separate mth5 files and merge their run summaries
        If False, use an already-merged mth5

    Returns
    -------

    """
    mth5_paths = make_mth5s(merged=merged)
    super_summary = extract_run_summaries_from_mth5s(mth5_paths)

    dataset_df = super_summary[super_summary.station_id=="test1"]
    dataset_df["remote"] = False
    run_test1("fortran", dataset_df)
    run_test1("matlab", dataset_df)
    
    dataset_df = super_summary.copy(deep=True)
    dataset_df["remote"] = [True, False]
    run_test2r1(dataset_df)



def test():
    test_pipeline(merged=False)
    test_pipeline(merged=True)


def main():
    test()


if __name__ == "__main__":
    main()
