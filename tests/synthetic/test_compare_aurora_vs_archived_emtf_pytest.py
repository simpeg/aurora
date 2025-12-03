from loguru import logger
from mth5.data.make_mth5_from_asc import (
    create_test1_h5,
    create_test2_h5,
    create_test12rr_h5,
)
from mth5.helpers import close_open_files
from mth5.processing import KernelDataset, RunSummary

from aurora.general_helper_functions import DATA_PATH
from aurora.pipelines.process_mth5 import process_mth5
from aurora.sandbox.io_helpers.zfile_murphy import read_z_file
from aurora.test_utils.synthetic.make_processing_configs import create_test_run_config
from aurora.test_utils.synthetic.plot_helpers_synthetic import plot_rho_phi
from aurora.test_utils.synthetic.rms_helpers import (
    assert_rms_misfit_ok,
    compute_rms,
    get_expected_rms_misfit,
)
from aurora.transfer_function.emtf_z_file_helpers import (
    merge_tf_collection_to_match_z_file,
)


# Path to baseline EMTF results in source tree
BASELINE_EMTF_PATH = DATA_PATH.joinpath("synthetic", "emtf_results")


def aurora_vs_emtf(
    synthetic_test_paths,
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
    Compare aurora processing results against EMTF baseline.

    Parameters
    ----------
    synthetic_test_paths : SyntheticTestPaths
        Path fixture for test directories
    test_case_id: str
        one of ["test1", "test2r1"]. "test1" is single station, "test2r1" is remote reference
    emtf_version: str
        one of ["fortran", "matlab"]
    auxilliary_z_file: str or pathlib.Path
        points to a .zss, .zrr or .zmm that EMTF produced
    z_file_base: str
        z_file basename for aurora output
    tfk_dataset: aurora.transfer_function.kernel_dataset.KernelDataset
        Info about data to process
    make_rho_phi_plot: bool
    show_rho_phi_plot: bool
    use_subtitle: bool
    """
    AURORA_RESULTS_PATH = synthetic_test_paths.aurora_results_path

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
    aurora_rho_phi = merge_tf_collection_to_match_z_file(aux_data, tf_collection)
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


def test_pipeline_merged(synthetic_test_paths, subtests):
    """Test aurora vs EMTF comparison with merged mth5."""
    close_open_files()

    # Create merged mth5
    mth5_path = create_test12rr_h5()
    mth5_paths = [mth5_path]

    run_summary = RunSummary()
    run_summary.from_mth5s(mth5_paths)

    # Test1 vs fortran
    with subtests.test(case="test1", version="fortran"):
        logger.info("Test1 vs fortran")
        tfk_dataset = KernelDataset()
        tfk_dataset.from_run_summary(run_summary, "test1")
        auxilliary_z_file = BASELINE_EMTF_PATH.joinpath("test1.zss")
        z_file_base = "test1_aurora_fortran.zss"
        aurora_vs_emtf(
            synthetic_test_paths,
            "test1",
            "fortran",
            auxilliary_z_file,
            z_file_base,
            tfk_dataset,
        )

    # Test1 vs matlab
    with subtests.test(case="test1", version="matlab"):
        logger.info("Test1 vs matlab")
        tfk_dataset = KernelDataset()
        tfk_dataset.from_run_summary(run_summary, "test1")
        auxilliary_z_file = BASELINE_EMTF_PATH.joinpath("test1.zss")
        z_file_base = "test1_aurora_matlab.zss"
        aurora_vs_emtf(
            synthetic_test_paths,
            "test1",
            "matlab",
            auxilliary_z_file,
            z_file_base,
            tfk_dataset,
        )

    # Test2r1 vs fortran
    with subtests.test(case="test2r1", version="fortran"):
        logger.info("Test2r1")
        tfk_dataset = KernelDataset()
        tfk_dataset.from_run_summary(run_summary, "test2", "test1")
        auxilliary_z_file = BASELINE_EMTF_PATH.joinpath("test2r1.zrr")
        z_file_base = "test2r1_aurora_fortran.zrr"
        aurora_vs_emtf(
            synthetic_test_paths,
            "test2r1",
            "fortran",
            auxilliary_z_file,
            z_file_base,
            tfk_dataset,
        )


def test_pipeline_separate(synthetic_test_paths, subtests):
    """Test aurora vs EMTF comparison with separate mth5 files."""
    close_open_files()

    # Create separate mth5 files
    mth5_path_1 = create_test1_h5()
    mth5_path_2 = create_test2_h5()
    mth5_paths = [mth5_path_1, mth5_path_2]

    run_summary = RunSummary()
    run_summary.from_mth5s(mth5_paths)

    # Test1 vs fortran
    with subtests.test(case="test1", version="fortran"):
        logger.info("Test1 vs fortran")
        tfk_dataset = KernelDataset()
        tfk_dataset.from_run_summary(run_summary, "test1")
        auxilliary_z_file = BASELINE_EMTF_PATH.joinpath("test1.zss")
        z_file_base = "test1_aurora_fortran.zss"
        aurora_vs_emtf(
            synthetic_test_paths,
            "test1",
            "fortran",
            auxilliary_z_file,
            z_file_base,
            tfk_dataset,
        )

    # Test1 vs matlab
    with subtests.test(case="test1", version="matlab"):
        logger.info("Test1 vs matlab")
        tfk_dataset = KernelDataset()
        tfk_dataset.from_run_summary(run_summary, "test1")
        auxilliary_z_file = BASELINE_EMTF_PATH.joinpath("test1.zss")
        z_file_base = "test1_aurora_matlab.zss"
        aurora_vs_emtf(
            synthetic_test_paths,
            "test1",
            "matlab",
            auxilliary_z_file,
            z_file_base,
            tfk_dataset,
        )

    # Test2r1 vs fortran
    with subtests.test(case="test2r1", version="fortran"):
        logger.info("Test2r1")
        tfk_dataset = KernelDataset()
        tfk_dataset.from_run_summary(run_summary, "test2", "test1")
        auxilliary_z_file = BASELINE_EMTF_PATH.joinpath("test2r1.zrr")
        z_file_base = "test2r1_aurora_fortran.zrr"
        aurora_vs_emtf(
            synthetic_test_paths,
            "test2r1",
            "fortran",
            auxilliary_z_file,
            z_file_base,
            tfk_dataset,
        )
