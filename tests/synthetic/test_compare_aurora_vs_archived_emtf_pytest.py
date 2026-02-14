"""
Tests comparing aurora processing results against archived EMTF results.

Development Notes:
- In the early days of development these tests were useful to check that
the same results were obtained from processing an mth5 with both stations in it
and two separate mth5 files.  This could probably be made into a simpler test in mth5
that checks that the data are the same in the two files.
- This used to make a homebrew resistivity and phase plot for comparison between
archived emtf z-files, but has been replaced with mt_metadata methods.
- TODO: Check phases in these plots -- they are off by 180 so there may be a sign error
in the data, or maybe the emtf results are using a different convention.  Need to investigate.
- The comparison with the matlab emtf results uses a slighly different windowing method.

"""
import numpy as np
import pytest
from loguru import logger
from mth5.helpers import close_open_files
from mth5.processing import KernelDataset, RunSummary

from aurora.general_helper_functions import DATA_PATH
from aurora.pipelines.process_mth5 import process_mth5
from aurora.test_utils.synthetic.make_processing_configs import create_test_run_config
from aurora.transfer_function.compare import CompareTF


# Path to baseline EMTF results in source tree
BASELINE_EMTF_PATH = DATA_PATH.joinpath("synthetic", "emtf_results")


def aurora_vs_emtf(
    synthetic_test_paths,
    test_case_id,
    emtf_version,
    auxilliary_z_file,
    z_file_base,
    tfk_dataset,
    atol_phase=4.0,
    make_rho_phi_plot=True,
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

    z_file_path = AURORA_RESULTS_PATH.joinpath(z_file_base)

    process_mth5(
        processing_config,
        tfk_dataset=tfk_dataset,
        z_file_path=z_file_path,
        return_collection=True,
    )
    comparator = CompareTF(tf_01=z_file_path, tf_02=auxilliary_z_file)
    result = comparator.compare_transfer_functions(atol_phase=4.0)
    assert np.isclose(result["impedance_ratio"]["Z_10"], 1.0, atol=1e-2)
    assert np.isclose(result["impedance_ratio"]["Z_01"], 1.0, atol=1e-2)
    assert result["impedance_phase_close"]
    if make_rho_phi_plot:
        comparator.plot_two_transfer_functions(
            save_plot_path=AURORA_RESULTS_PATH.joinpath(
                f"{test_case_id}_aurora_vs_emtf_{emtf_version}_tf_compare.png"
            ),
            rho_xy_ylims=(10.0, 1000.0),
            rho_yx_ylims=(10.0, 1000.0),
        )


@pytest.mark.slow
def test_pipeline_merged(synthetic_test_paths, subtests, worker_safe_test12rr_h5):
    """Test aurora vs EMTF comparison with merged mth5."""
    close_open_files()

    # Create merged mth5
    mth5_path = worker_safe_test12rr_h5
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


@pytest.mark.slow
def test_pipeline_separate(
    synthetic_test_paths, subtests, worker_safe_test1_h5, worker_safe_test2_h5
):
    """Test aurora vs EMTF comparison with separate mth5 files."""
    close_open_files()

    # Create separate mth5 files
    mth5_path_1 = worker_safe_test1_h5
    mth5_path_2 = worker_safe_test2_h5
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
