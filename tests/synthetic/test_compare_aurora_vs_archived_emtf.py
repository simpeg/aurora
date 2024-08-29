from aurora.pipelines.process_mth5 import process_mth5
from aurora.sandbox.io_helpers.zfile_murphy import read_z_file
from mth5.data.make_mth5_from_asc import create_test1_h5
from mth5.data.make_mth5_from_asc import create_test2_h5
from mth5.data.make_mth5_from_asc import create_test12rr_h5
from aurora.test_utils.synthetic.make_processing_configs import (
    create_test_run_config,
)
from aurora.test_utils.synthetic.paths import SyntheticTestPaths
from aurora.test_utils.synthetic.rms_helpers import assert_rms_misfit_ok
from aurora.test_utils.synthetic.rms_helpers import compute_rms
from aurora.test_utils.synthetic.rms_helpers import get_expected_rms_misfit
from aurora.transfer_function.emtf_z_file_helpers import (
    merge_tf_collection_to_match_z_file,
)

# from mtpy-v2
from mtpy.processing import RunSummary, KernelDataset

from plot_helpers_synthetic import plot_rho_phi
from loguru import logger
from mth5.helpers import close_open_files

synthetic_test_paths = SyntheticTestPaths()
synthetic_test_paths.mkdirs()
AURORA_RESULTS_PATH = synthetic_test_paths.aurora_results_path
EMTF_RESULTS_PATH = synthetic_test_paths.emtf_results_path


def aurora_vs_emtf(
    test_case_id,
    emtf_version,
    auxilliary_z_file,
    z_file_base,
    tfk_dataset,
    make_rho_phi_plot=True,
    show_rho_phi_plot=False,
    use_subtitle=True,
):
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
    tfk_dataset: aurora.transfer_function.kernel_dataset.KernelDataset
        Info about the data to process
    make_rho_phi_plot: bool
    show_rho_phi_plot: bool
    use_subtitle: bool
    """
    processing_config = create_test_run_config(
        test_case_id, tfk_dataset, matlab_or_fortran=emtf_version
    )

    expected_rms_misfit = get_expected_rms_misfit(test_case_id, emtf_version)
    z_file_path = AURORA_RESULTS_PATH.joinpath(z_file_base)

    tf_collection = process_mth5(
        processing_config,
        tfk_dataset=tfk_dataset,
        z_file_path=z_file_path,
        return_collection=True,
    )

    aux_data = read_z_file(auxilliary_z_file)
    aurora_rho_phi = merge_tf_collection_to_match_z_file(
        aux_data, tf_collection
    )
    data_dict = {}
    data_dict["period"] = aux_data.periods
    data_dict["emtf_rho_xy"] = aux_data.rxy
    data_dict["emtf_phi_xy"] = aux_data.pxy
    for xy_or_yx in ["xy", "yx"]:
        aurora_rho = aurora_rho_phi["rho"][xy_or_yx]
        aurora_phi = aurora_rho_phi["phi"][xy_or_yx]
        aux_rho = aux_data.rho(xy_or_yx)
        aux_phi = aux_data.phi(xy_or_yx)
        rho_rms_aurora, phi_rms_aurora = compute_rms(
            aurora_rho, aurora_phi, verbose=True
        )
        rho_rms_emtf, phi_rms_emtf = compute_rms(aux_rho, aux_phi)
        data_dict["aurora_rho_xy"] = aurora_rho
        data_dict["aurora_phi_xy"] = aurora_phi
        if expected_rms_misfit is not None:
            assert_rms_misfit_ok(
                expected_rms_misfit, xy_or_yx, rho_rms_aurora, phi_rms_aurora
            )

        if make_rho_phi_plot:
            plot_rho_phi(
                xy_or_yx,
                tf_collection,
                rho_rms_aurora,
                rho_rms_emtf,
                phi_rms_aurora,
                phi_rms_emtf,
                emtf_version,
                aux_data=aux_data,
                use_subtitle=use_subtitle,
                show_plot=show_rho_phi_plot,
                output_path=AURORA_RESULTS_PATH,
            )

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
    logger.info(f"Test1 vs {emtf_version}")
    test_case_id = "test1"
    auxilliary_z_file = EMTF_RESULTS_PATH.joinpath("test1.zss")
    z_file_base = f"{test_case_id}_aurora_{emtf_version}.zss"
    aurora_vs_emtf(
        test_case_id, emtf_version, auxilliary_z_file, z_file_base, ds_df
    )
    return


def run_test2r1(tfk_dataset):
    """

    Parameters
    ----------
    ds_df : pandas.DataFrame
        Basically a run_summary dataframe
    Returns
    -------

    """
    logger.info("Test2r1")
    test_case_id = "test2r1"
    emtf_version = "fortran"
    auxilliary_z_file = EMTF_RESULTS_PATH.joinpath("test2r1.zrr")
    z_file_base = f"{test_case_id}_aurora_{emtf_version}.zrr"
    aurora_vs_emtf(
        test_case_id, emtf_version, auxilliary_z_file, z_file_base, tfk_dataset
    )
    return


def make_mth5s(merged=True):
    """
    Returns
    -------
    mth5_paths: list of Path objs or str(Path)
    """
    if merged:
        mth5_path = create_test12rr_h5()
        mth5_paths = [
            mth5_path,
        ]
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
    close_open_files()

    mth5_paths = make_mth5s(merged=merged)
    run_summary = RunSummary()
    run_summary.from_mth5s(mth5_paths)
    tfk_dataset = KernelDataset()
    tfk_dataset.from_run_summary(run_summary, "test1")

    run_test1("fortran", tfk_dataset)
    run_test1("matlab", tfk_dataset)

    tfk_dataset = KernelDataset()
    tfk_dataset.from_run_summary(run_summary, "test2", "test1")
    # Uncomment to sanity check the problem is linear
    # scale_factors = {
    #     "ex": 20.0,
    #     "ey": 20.0,
    #     "hx": 20.0,
    #     "hy": 20.0,
    #     "hz": 20.0,
    # }
    # tfk_dataset.df["channel_scale_factors"].at[0] = scale_factors
    # tfk_dataset.df["channel_scale_factors"].at[1] = scale_factors
    run_test2r1(tfk_dataset)


def test():
    import logging

    logging.getLogger("matplotlib.font_manager").disabled = True
    logging.getLogger("matplotlib.ticker").disabled = True

    test_pipeline(merged=False)
    test_pipeline(merged=True)


def main():
    test()


if __name__ == "__main__":
    main()
