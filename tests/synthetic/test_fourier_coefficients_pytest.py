import pytest
from loguru import logger
from mth5.data.make_mth5_from_asc import (
    create_test1_h5,
    create_test2_h5,
    create_test3_h5,
    create_test12rr_h5,
)
from mth5.helpers import close_open_files
from mth5.processing import KernelDataset, RunSummary
from mth5.timeseries.spectre.helpers import (
    add_fcs_to_mth5,
    fc_decimations_creator,
    read_back_fcs,
)

from aurora.config.config_creator import ConfigCreator
from aurora.pipelines.process_mth5 import process_mth5
from aurora.pipelines.transfer_function_kernel import TransferFunctionKernel
from aurora.test_utils.synthetic.make_processing_configs import (
    create_test_run_config,
    make_processing_config_and_kernel_dataset,
)
from aurora.test_utils.synthetic.processing_helpers import process_synthetic_2
from aurora.test_utils.synthetic.triage import tfs_nearly_equal


@pytest.fixture(scope="module")
def mth5_test_files():
    """Create synthetic MTH5 test files."""
    logger.info("Making synthetic data")
    close_open_files()
    file_version = "0.1.0"
    mth5_path_1 = create_test1_h5(file_version=file_version)
    mth5_path_2 = create_test2_h5(file_version=file_version)
    mth5_path_3 = create_test3_h5(file_version=file_version)
    mth5_path_12rr = create_test12rr_h5(file_version=file_version)

    return {
        "paths": [mth5_path_1, mth5_path_2, mth5_path_3, mth5_path_12rr],
        "path_2": mth5_path_2,
    }


def test_add_fcs_to_all_synthetic_files(mth5_test_files, subtests):
    """Test adding Fourier Coefficients to each synthetic file.

    Uses the to_fc_decimation() method of AuroraDecimationLevel.
    """
    for mth5_path in mth5_test_files["paths"]:
        with subtests.test(file=mth5_path.stem):
            mth5_paths = [mth5_path]
            run_summary = RunSummary()
            run_summary.from_mth5s(mth5_paths)
            tfk_dataset = KernelDataset()

            # Get Processing Config
            if mth5_path.stem in ["test1", "test2"]:
                station_id = mth5_path.stem
                tfk_dataset.from_run_summary(run_summary, station_id)
                processing_config = create_test_run_config(station_id, tfk_dataset)
            elif mth5_path.stem in ["test3"]:
                station_id = "test3"
                tfk_dataset.from_run_summary(run_summary, station_id)
                cc = ConfigCreator()
                processing_config = cc.create_from_kernel_dataset(tfk_dataset)
            elif mth5_path.stem in ["test12rr"]:
                tfk_dataset.from_run_summary(run_summary, "test1", "test2")
                cc = ConfigCreator()
                processing_config = cc.create_from_kernel_dataset(tfk_dataset)

            # Extract FC decimations from processing config and build the layer
            fc_decimations = [
                x.to_fc_decimation() for x in processing_config.decimations
            ]
            # For code coverage, have a case where fc_decimations is None
            # This also (indirectly) tests a different FCDecimation object.
            if mth5_path.stem == "test1":
                fc_decimations = None

            add_fcs_to_mth5(mth5_path, fc_decimations=fc_decimations)
            read_back_fcs(mth5_path)

            # Confirm the file still processes fine with the fcs inside
            tfc = process_mth5(processing_config, tfk_dataset=tfk_dataset)
            assert tfc is not None


def test_fc_decimations_creator():
    """Test fc_decimations_creator utility function."""
    cfgs = fc_decimations_creator(initial_sample_rate=1.0)
    assert cfgs is not None

    # test time period must be of correct type
    with pytest.raises(NotImplementedError):
        time_period = ["2023-01-01T17:48:29", "2023-01-09T08:54:08"]
        fc_decimations_creator(1.0, time_period=time_period)


@pytest.mark.xfail(
    reason="TypeError in mt_metadata decimation_level.py line 535 - harmonic_indices is None on pydantic branch"
)
def test_create_then_use_stored_fcs_for_processing(
    mth5_test_files, synthetic_test_paths
):
    """Test creating and using stored Fourier Coefficients for processing."""
    AURORA_RESULTS_PATH = synthetic_test_paths.aurora_results_path
    mth5_path_2 = mth5_test_files["path_2"]

    z_file_path_1 = AURORA_RESULTS_PATH.joinpath("test2.zss")
    z_file_path_2 = AURORA_RESULTS_PATH.joinpath("test2_from_stored_fc.zss")
    tf1 = process_synthetic_2(
        force_make_mth5=True, z_file_path=z_file_path_1, save_fc=True
    )
    tfk_dataset, processing_config = make_processing_config_and_kernel_dataset(
        config_keyword="test2",
        station_id="test2",
        remote_id=None,
        mth5s=[mth5_path_2],
        channel_nomenclature="default",
    )

    # Initialize a TF kernel to check for FCs
    original_window = processing_config.decimations[0].stft.window.type

    tfk = TransferFunctionKernel(dataset=tfk_dataset, config=processing_config)
    tfk.update_processing_summary()
    tfk.check_if_fcs_already_exist()
    assert (
        tfk.dataset_df.fc.all()
    )  # assert fcs True in dataframe -- i.e. they were detected.

    # now change the window type and show that FCs are not detected
    for decimation in processing_config.decimations:
        decimation.stft.window.type = "hamming"
    tfk = TransferFunctionKernel(dataset=tfk_dataset, config=processing_config)
    tfk.update_processing_summary()
    tfk.check_if_fcs_already_exist()
    assert not (
        tfk.dataset_df.fc.all()
    )  # assert fcs False in dataframe -- i.e. they were detected.

    # Now reprocess with the FCs
    for decimation in processing_config.decimations:
        decimation.stft.window.type = original_window
    tfk = TransferFunctionKernel(dataset=tfk_dataset, config=processing_config)
    tfk.update_processing_summary()
    tfk.check_if_fcs_already_exist()
    assert (
        tfk.dataset_df.fc.all()
    )  # assert fcs True in dataframe -- i.e. they were detected.

    tf2 = process_synthetic_2(force_make_mth5=False, z_file_path=z_file_path_2)
    assert tfs_nearly_equal(tf1, tf2)
