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
from pathlib import Path

from aurora.config.metadata.processing import Processing
from aurora.general_helper_functions import TEST_PATH
from aurora.pipelines.helpers import initialize_config
from aurora.pipelines.process_mth5_dev import process_mth5_from_dataset_definition
from aurora.sandbox.io_helpers.zfile_murphy import read_z_file
from aurora.test_utils.synthetic.make_processing_configs_new import create_test_run_config
from aurora.test_utils.synthetic.rms_helpers import assert_rms_misfit_ok
from aurora.test_utils.synthetic.rms_helpers import compute_rms
from aurora.test_utils.synthetic.rms_helpers import get_expected_rms_misfit
from aurora.tf_kernel.dataset import DatasetDefinition
from aurora.tf_kernel.helpers import extract_run_summaries_from_mth5s
from aurora.transfer_function.emtf_z_file_helpers import (
    merge_tf_collection_to_match_z_file,
)

from make_mth5_from_asc import create_test1_h5
from make_mth5_from_asc import create_test2_h5
from make_mth5_from_asc import create_test12rr_h5
from plot_helpers_synthetic import plot_rho_phi

SYNTHETIC_PATH = TEST_PATH.joinpath("synthetic")
CONFIG_PATH = SYNTHETIC_PATH.joinpath("config")
DATA_PATH = SYNTHETIC_PATH.joinpath("data")
EMTF_OUTPUT_PATH = SYNTHETIC_PATH.joinpath("emtf_output")
AURORA_RESULTS_PATH = SYNTHETIC_PATH.joinpath("aurora_results")
AURORA_RESULTS_PATH.mkdir(exist_ok=True)



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
    print("success")


def main():
    test()


if __name__ == "__main__":
    main()
