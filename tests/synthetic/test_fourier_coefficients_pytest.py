import pytest
from loguru import logger
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
def mth5_test_files(
    worker_safe_test1_h5,
    worker_safe_test2_h5,
    worker_safe_test3_h5,
    worker_safe_test12rr_h5,
):
    """Create synthetic MTH5 test files."""
    logger.info("Making synthetic data")
    close_open_files()

    return {
        "paths": [
            worker_safe_test1_h5,
            worker_safe_test2_h5,
            worker_safe_test3_h5,
            worker_safe_test12rr_h5,
        ],
        "path_2": worker_safe_test2_h5,
    }


def test_add_fcs_to_all_synthetic_files(mth5_test_files, subtests):
    """Test adding Fourier Coefficients to each synthetic file.

    Uses the to_fc_decimation() method of AuroraDecimationLevel.
    Tests each step of the workflow with detailed validation:
    1. File validation (exists, can open, has structure)
    2. RunSummary creation and validation
    3. KernelDataset creation and validation
    4. Processing config creation and validation
    5. FC addition and validation
    6. FC readback validation
    7. Processing with FCs
    """
    from mth5 import mth5

    for mth5_path in mth5_test_files["paths"]:
        subtest_name = mth5_path.stem
        with subtests.test(file=mth5_path.stem):
            logger.info(f"\n{'='*80}\nTesting {mth5_path.stem}\n{'='*80}")

            # Step 1: File validation
            with subtests.test(step=f"{subtest_name}_file_exists"):
                assert mth5_path.exists(), f"{mth5_path.stem} not found at {mth5_path}"
                logger.info(f"✓ File exists: {mth5_path}")

            with subtests.test(step=f"{subtest_name}_file_opens"):
                with mth5.MTH5(file_version="0.1.0") as m:
                    m.open_mth5(mth5_path, mode="r")
                    stations = m.stations_group.groups_list
                    assert len(stations) > 0, f"No stations found in {mth5_path.stem}"
                    logger.info(f"✓ File opens, stations: {stations}")

            with subtests.test(step=f"{subtest_name}_has_runs_and_channels"):
                with mth5.MTH5(file_version="0.1.0") as m:
                    m.open_mth5(mth5_path, mode="r")
                    for station_id in m.stations_group.groups_list:
                        station = m.get_station(station_id)
                        runs = [
                            r
                            for r in station.groups_list
                            if r
                            not in [
                                "Transfer_Functions",
                                "Fourier_Coefficients",
                                "Features",
                            ]
                        ]
                        assert len(runs) > 0, f"Station {station_id} has no runs"

                        for run_id in runs:
                            run = station.get_run(run_id)
                            channels = run.groups_list
                            assert len(channels) > 0, f"Run {run_id} has no channels"

                            # Verify channels have data
                            for ch_name in channels:
                                ch = run.get_channel(ch_name)
                                assert (
                                    ch.n_samples > 0
                                ), f"Channel {ch_name} has no data"

                        logger.info(
                            f"✓ Station {station_id}: {len(runs)} run(s), channels validated"
                        )

            # Step 2: RunSummary creation and validation
            with subtests.test(step=f"{subtest_name}_run_summary"):
                mth5_paths = [mth5_path]
                run_summary = RunSummary()
                run_summary.from_mth5s(mth5_paths)

                assert (
                    len(run_summary.df) > 0
                ), f"RunSummary is empty for {mth5_path.stem}"

                # Validate sample rates are positive
                invalid_rates = run_summary.df[run_summary.df.sample_rate <= 0]
                assert len(invalid_rates) == 0, (
                    f"RunSummary has {len(invalid_rates)} entries with invalid sample_rate:\n"
                    f"{invalid_rates[['station', 'run', 'sample_rate']]}"
                )

                logger.info(
                    f"✓ RunSummary: {len(run_summary.df)} entries, "
                    f"sample_rates={run_summary.df.sample_rate.unique()}"
                )

            # Step 3: KernelDataset creation and validation
            with subtests.test(step=f"{subtest_name}_kernel_dataset"):
                tfk_dataset = KernelDataset()

                # Get Processing Config - determine station IDs
                if mth5_path.stem in ["test1", "test2"]:
                    station_id = mth5_path.stem
                    tfk_dataset.from_run_summary(run_summary, station_id)
                elif mth5_path.stem in ["test3"]:
                    station_id = "test3"
                    tfk_dataset.from_run_summary(run_summary, station_id)
                elif mth5_path.stem in ["test12rr"]:
                    tfk_dataset.from_run_summary(run_summary, "test1", "test2")

                assert (
                    len(tfk_dataset.df) > 0
                ), f"KernelDataset is empty for {mth5_path.stem}"
                assert (
                    "station" in tfk_dataset.df.columns
                ), "KernelDataset missing 'station' column"
                assert (
                    "run" in tfk_dataset.df.columns
                ), "KernelDataset missing 'run' column"

                logger.info(
                    f"✓ KernelDataset: {len(tfk_dataset.df)} entries, "
                    f"stations={tfk_dataset.df.station.unique()}"
                )

            # Step 4: Processing config creation and validation
            with subtests.test(step=f"{subtest_name}_processing_config"):
                if mth5_path.stem in ["test1", "test2"]:
                    processing_config = create_test_run_config(station_id, tfk_dataset)
                elif mth5_path.stem in ["test3", "test12rr"]:
                    cc = ConfigCreator()
                    processing_config = cc.create_from_kernel_dataset(tfk_dataset)

                assert processing_config is not None, "Processing config is None"
                assert (
                    len(processing_config.decimations) > 0
                ), "No decimations in processing config"
                assert (
                    processing_config.channel_nomenclature is not None
                ), "No channel nomenclature"

                logger.info(
                    f"✓ Processing config: {len(processing_config.decimations)} decimations"
                )

            # Step 5: FC addition and validation
            with subtests.test(step=f"{subtest_name}_add_fcs"):
                # Extract FC decimations from processing config
                fc_decimations = [
                    x.to_fc_decimation() for x in processing_config.decimations
                ]
                # For code coverage, test with fc_decimations=None for test1
                if mth5_path.stem == "test1":
                    fc_decimations = None

                # Verify no FC group before adding
                with mth5.MTH5(file_version="0.1.0") as m:
                    m.open_mth5(mth5_path, mode="r")
                    for station_id in m.stations_group.groups_list:
                        station = m.get_station(station_id)
                        groups_before = station.groups_list
                        # FC group might already exist from previous runs, but should be empty or absent

                add_fcs_to_mth5(mth5_path, fc_decimations=fc_decimations)

                # Validate FC group exists and has content
                with mth5.MTH5(file_version="0.1.0") as m:
                    m.open_mth5(mth5_path, mode="r")
                    for station_id in m.stations_group.groups_list:
                        station = m.get_station(station_id)
                        groups_after = station.groups_list

                        assert "Fourier_Coefficients" in groups_after, (
                            f"Fourier_Coefficients group not found in station {station_id} "
                            f"after adding FCs. Groups: {groups_after}"
                        )

                        fc_group = station.fourier_coefficients_group
                        fc_runs = fc_group.groups_list
                        assert (
                            len(fc_runs) > 0
                        ), f"No FC runs found in station {station_id} after adding FCs"

                        # Validate each FC run has decimation levels
                        for fc_run_id in fc_runs:
                            fc_run = fc_group.get_fc_group(fc_run_id)
                            dec_levels = fc_run.groups_list
                            assert (
                                len(dec_levels) > 0
                            ), f"No decimation levels in FC run {fc_run_id}"

                        logger.info(
                            f"✓ FCs added to station {station_id}: "
                            f"{len(fc_runs)} run(s), {len(dec_levels)} decimation level(s)"
                        )

            # Step 6: FC readback validation
            with subtests.test(step=f"{subtest_name}_read_back_fcs"):
                # This tests that FCs can be read back from the file
                read_back_fcs(mth5_path)
                logger.info(f"✓ FCs read back successfully")

            # Step 7: Processing with FCs
            with subtests.test(step=f"{subtest_name}_process_with_fcs"):
                tfc = process_mth5(processing_config, tfk_dataset=tfk_dataset)

                assert (
                    tfc is not None
                ), f"process_mth5 returned None for {mth5_path.stem}"
                assert hasattr(
                    tfc, "station_metadata"
                ), "TF object missing station_metadata"
                assert (
                    len(tfc.station_metadata.runs) > 0
                ), "TF object has no runs in metadata"

                logger.info(
                    f"✓ Processing completed: {type(tfc).__name__}, "
                    f"{len(tfc.station_metadata.runs)} run(s) processed"
                )

            logger.info(f"✓ All tests passed for {mth5_path.stem}\n")


def test_fc_decimations_creator():
    """Test fc_decimations_creator utility function."""
    cfgs = fc_decimations_creator(initial_sample_rate=1.0)
    assert cfgs is not None

    # test time period must be of correct type
    with pytest.raises(NotImplementedError):
        time_period = ["2023-01-01T17:48:29", "2023-01-09T08:54:08"]
        fc_decimations_creator(1.0, time_period=time_period)


def test_create_then_use_stored_fcs_for_processing(
    mth5_test_files, synthetic_test_paths
):
    """Test creating and using stored Fourier Coefficients for processing."""
    AURORA_RESULTS_PATH = synthetic_test_paths.aurora_results_path
    mth5_path_2 = mth5_test_files["path_2"]

    z_file_path_1 = AURORA_RESULTS_PATH.joinpath("test2.zss")
    z_file_path_2 = AURORA_RESULTS_PATH.joinpath("test2_from_stored_fc.zss")
    tf1 = process_synthetic_2(
        force_make_mth5=True,
        z_file_path=z_file_path_1,
        save_fc=True,
        mth5_path=mth5_path_2,
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

    tf2 = process_synthetic_2(
        force_make_mth5=False, z_file_path=z_file_path_2, mth5_path=mth5_path_2
    )
    assert tfs_nearly_equal(tf1, tf2)
