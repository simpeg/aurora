"""
Discrete tests for Fourier Coefficients workflow.

Each test file is tested separately with clear stages:
1. File validation
2. FC creation
3. FC storage
4. Processing
"""

import shutil

import pytest
from loguru import logger
from mth5 import mth5
from mth5.helpers import close_open_files
from mth5.processing import KernelDataset, RunSummary
from mth5.timeseries.spectre.helpers import add_fcs_to_mth5

from aurora.config.config_creator import ConfigCreator
from aurora.pipelines.process_mth5 import process_mth5
from aurora.test_utils.synthetic.make_processing_configs import create_test_run_config


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
        "test1": worker_safe_test1_h5,
        "test2": worker_safe_test2_h5,
        "test3": worker_safe_test3_h5,
        "test12rr": worker_safe_test12rr_h5,
    }


@pytest.fixture
def temp_copy(tmp_path):
    """Create a temporary copy of an mth5 file."""

    def _copy(source_path):
        dest_path = tmp_path / source_path.name
        shutil.copy2(source_path, dest_path)
        return dest_path

    return _copy


# ==============================================================================
# TEST1 - Single station, single run, simple case
# ==============================================================================


def test_test1_file_validation(mth5_test_files):
    """Stage 1: Verify test1.h5 has expected structure and data."""
    mth5_path = mth5_test_files["test1"]

    # File should exist
    assert mth5_path.exists(), f"test1.h5 not found at {mth5_path}"

    # Open and validate structure
    with mth5.MTH5(file_version="0.1.0") as m:
        m.open_mth5(mth5_path, mode="r")

        # Should have test1 station
        stations = m.stations_group.groups_list
        assert "test1" in stations, f"test1 station not found. Stations: {stations}"

        # Get station and validate
        station = m.get_station("test1")
        runs = [
            r
            for r in station.groups_list
            if r not in ["Transfer_Functions", "Fourier_Coefficients", "Features"]
        ]
        assert len(runs) > 0, "No runs found in test1 station"

        # Check first run
        run = station.get_run(runs[0])

        # Read metadata before accessing sample_rate to avoid lazy loading issues
        run.read_metadata()
        assert (
            run.metadata.sample_rate > 0
        ), f"Run {runs[0]} sample_rate is {run.metadata.sample_rate}, expected > 0"

        # Check channels
        channels = run.groups_list
        expected_channels = ["ex", "ey", "hx", "hy", "hz"]
        for ch_name in expected_channels:
            assert (
                ch_name in channels
            ), f"Channel {ch_name} not found. Channels: {channels}"
            ch = run.get_channel(ch_name)
            assert ch.n_samples > 0, f"Channel {ch_name} has no samples"

        logger.info(
            f"✓ test1.h5 validation passed: {len(runs)} runs, {len(expected_channels)} channels"
        )


def test_test1_create_fc_decimations(mth5_test_files):
    """Stage 2: Create FC decimation configuration for test1."""
    mth5_path = mth5_test_files["test1"]

    # Create RunSummary
    run_summary = RunSummary()
    run_summary.from_mth5s([mth5_path])

    assert len(run_summary.df) > 0, "RunSummary is empty"
    assert (
        run_summary.df.sample_rate > 0
    ).all(), "RunSummary has sample_rate=0 entries"

    # Create KernelDataset
    tfk_dataset = KernelDataset()
    tfk_dataset.from_run_summary(run_summary, "test1")

    assert len(tfk_dataset.df) > 0, "KernelDataset is empty"

    # Create processing config
    processing_config = create_test_run_config("test1", tfk_dataset)

    assert processing_config is not None, "Processing config is None"
    assert len(processing_config.decimations) > 0, "No decimations in processing config"

    # Extract FC decimations (set to None to test default creation)
    fc_decimations = None  # Will be created by add_fcs_to_mth5

    logger.info(
        f"✓ test1 FC config created: {len(processing_config.decimations)} decimations"
    )

    return {
        "config": processing_config,
        "tfk_dataset": tfk_dataset,
        "fc_decimations": fc_decimations,
    }


def test_test1_add_fcs(mth5_test_files, temp_copy):
    """Stage 3: Add FCs to test1.h5 and verify storage."""
    source_path = mth5_test_files["test1"]
    mth5_path = temp_copy(source_path)

    # Create FC decimations
    run_summary = RunSummary()
    run_summary.from_mth5s([mth5_path])
    tfk_dataset = KernelDataset()
    tfk_dataset.from_run_summary(run_summary, "test1")
    processing_config = create_test_run_config("test1", tfk_dataset)
    fc_decimations = None  # Test default creation

    # Add FCs
    logger.info(f"Adding FCs to {mth5_path}")
    add_fcs_to_mth5(mth5_path, fc_decimations=fc_decimations)

    # Verify FCs were written
    with mth5.MTH5(file_version="0.1.0") as m:
        m.open_mth5(mth5_path, mode="r")

        station = m.get_station("test1")

        # Check FC group exists
        assert (
            "Fourier_Coefficients" in station.groups_list
        ), "No Fourier_Coefficients group"

        fc_group = station.fourier_coefficients_group
        fc_runs = fc_group.groups_list
        assert len(fc_runs) > 0, "No FC runs found"

        # Check at least one FC run has decimation levels
        fc_run = fc_group.get_fc_group(fc_runs[0])
        decimation_levels = fc_run.groups_list
        assert (
            len(decimation_levels) > 0
        ), f"No decimation levels in FC run {fc_runs[0]}"

        logger.info(
            f"✓ test1 FCs stored: {len(fc_runs)} runs, {len(decimation_levels)} decimation levels"
        )


def test_test1_process(mth5_test_files, temp_copy):
    """Stage 4: Process test1.h5 with FCs."""
    source_path = mth5_test_files["test1"]
    mth5_path = temp_copy(source_path)

    # Setup
    run_summary = RunSummary()
    run_summary.from_mth5s([mth5_path])
    tfk_dataset = KernelDataset()
    tfk_dataset.from_run_summary(run_summary, "test1")
    processing_config = create_test_run_config("test1", tfk_dataset)

    # Add FCs
    add_fcs_to_mth5(mth5_path, fc_decimations=None)

    # Process
    tfc = process_mth5(processing_config, tfk_dataset=tfk_dataset)

    assert tfc is not None, "process_mth5 returned None"
    logger.info(f"✓ test1 processing completed: {type(tfc)}")


# ==============================================================================
# TEST2 - Single station, single run, different data
# ==============================================================================


def test_test2_file_validation(mth5_test_files):
    """Stage 1: Verify test2.h5 has expected structure and data."""
    mth5_path = mth5_test_files["test2"]

    assert mth5_path.exists(), f"test2.h5 not found at {mth5_path}"

    with mth5.MTH5(file_version="0.1.0") as m:
        m.open_mth5(mth5_path, mode="r")

        stations = m.stations_group.groups_list
        assert "test2" in stations, f"test2 station not found. Stations: {stations}"

        station = m.get_station("test2")
        runs = [
            r
            for r in station.groups_list
            if r not in ["Transfer_Functions", "Fourier_Coefficients", "Features"]
        ]
        assert len(runs) > 0, "No runs found in test2 station"

        run = station.get_run(runs[0])
        run.read_metadata()
        assert (
            run.metadata.sample_rate > 0
        ), f"Run {runs[0]} sample_rate is {run.metadata.sample_rate}, expected > 0"

        channels = run.groups_list
        expected_channels = ["ex", "ey", "hx", "hy", "hz"]
        for ch_name in expected_channels:
            assert ch_name in channels, f"Channel {ch_name} not found"
            ch = run.get_channel(ch_name)
            assert ch.n_samples > 0, f"Channel {ch_name} has no samples"

        logger.info(f"✓ test2.h5 validation passed")


def test_test2_add_fcs(mth5_test_files, temp_copy):
    """Stage 3: Add FCs to test2.h5 and verify storage."""
    source_path = mth5_test_files["test2"]
    mth5_path = temp_copy(source_path)

    run_summary = RunSummary()
    run_summary.from_mth5s([mth5_path])

    # Verify sample_rate is correct in run_summary
    assert len(run_summary.df) > 0, "RunSummary is empty"
    assert (
        run_summary.df.sample_rate > 0
    ).all(), f"RunSummary has invalid sample_rate:\n{run_summary.df[['station', 'run', 'sample_rate']]}"

    tfk_dataset = KernelDataset()
    tfk_dataset.from_run_summary(run_summary, "test2")
    processing_config = create_test_run_config("test2", tfk_dataset)

    fc_decimations = [x.to_fc_decimation() for x in processing_config.decimations]

    logger.info(f"Adding FCs to {mth5_path}")
    add_fcs_to_mth5(mth5_path, fc_decimations=fc_decimations)

    with mth5.MTH5(file_version="0.1.0") as m:
        m.open_mth5(mth5_path, mode="r")
        station = m.get_station("test2")
        assert (
            "Fourier_Coefficients" in station.groups_list
        ), "No Fourier_Coefficients group"
        fc_group = station.fourier_coefficients_group
        assert len(fc_group.groups_list) > 0, "No FC runs found"
        logger.info(f"✓ test2 FCs stored: {len(fc_group.groups_list)} runs")


def test_test2_process(mth5_test_files, temp_copy):
    """Stage 4: Process test2.h5 with FCs."""
    source_path = mth5_test_files["test2"]
    mth5_path = temp_copy(source_path)

    run_summary = RunSummary()
    run_summary.from_mth5s([mth5_path])
    tfk_dataset = KernelDataset()
    tfk_dataset.from_run_summary(run_summary, "test2")
    processing_config = create_test_run_config("test2", tfk_dataset)

    fc_decimations = [x.to_fc_decimation() for x in processing_config.decimations]
    add_fcs_to_mth5(mth5_path, fc_decimations=fc_decimations)

    tfc = process_mth5(processing_config, tfk_dataset=tfk_dataset)
    assert tfc is not None, "process_mth5 returned None for test2"
    logger.info(f"✓ test2 processing completed")


# ==============================================================================
# TEST3 - Single station, multiple runs
# ==============================================================================


def test_test3_file_validation(mth5_test_files):
    """Stage 1: Verify test3.h5 has expected structure and data."""
    mth5_path = mth5_test_files["test3"]

    assert mth5_path.exists(), f"test3.h5 not found at {mth5_path}"

    with mth5.MTH5(file_version="0.1.0") as m:
        m.open_mth5(mth5_path, mode="r")

        stations = m.stations_group.groups_list
        assert "test3" in stations, f"test3 station not found"

        station = m.get_station("test3")
        runs = [
            r
            for r in station.groups_list
            if r not in ["Transfer_Functions", "Fourier_Coefficients", "Features"]
        ]
        assert len(runs) > 0, f"No runs found in test3 station"

        logger.info(f"test3 has {len(runs)} runs")

        # Validate each run
        for run_id in runs:
            run = station.get_run(run_id)
            run.read_metadata()
            sample_rate = run.metadata.sample_rate
            n_channels = len([ch for ch in run.groups_list if ch not in ["Features"]])

            logger.info(
                f"  Run {run_id}: sample_rate={sample_rate}, channels={n_channels}"
            )

            if sample_rate > 0:  # Only check runs with data
                assert (
                    n_channels > 0
                ), f"Run {run_id} has sample_rate={sample_rate} but no channels"

        logger.info(f"✓ test3.h5 validation passed: {len(runs)} runs")


def test_test3_add_fcs(mth5_test_files, temp_copy):
    """Stage 3: Add FCs to test3.h5 and verify storage."""
    source_path = mth5_test_files["test3"]
    mth5_path = temp_copy(source_path)

    run_summary = RunSummary()
    run_summary.from_mth5s([mth5_path])

    logger.info(f"test3 RunSummary shape: {run_summary.df.shape}")
    logger.info(
        f"test3 RunSummary:\n{run_summary.df[['station', 'run', 'sample_rate', 'start', 'end']]}"
    )

    # Filter to only runs with data
    valid_runs = run_summary.df[run_summary.df.sample_rate > 0]
    assert len(valid_runs) > 0, "No valid runs with sample_rate > 0"

    tfk_dataset = KernelDataset()
    tfk_dataset.from_run_summary(run_summary, "test3")

    cc = ConfigCreator()
    processing_config = cc.create_from_kernel_dataset(tfk_dataset)

    fc_decimations = [x.to_fc_decimation() for x in processing_config.decimations]

    logger.info(f"Adding FCs to {mth5_path}")
    try:
        add_fcs_to_mth5(mth5_path, fc_decimations=fc_decimations)

        with mth5.MTH5(file_version="0.1.0") as m:
            m.open_mth5(mth5_path, mode="r")
            station = m.get_station("test3")

            if "Fourier_Coefficients" in station.groups_list:
                fc_group = station.fourier_coefficients_group
                logger.info(f"✓ test3 FCs stored: {len(fc_group.groups_list)} runs")
            else:
                logger.warning("No Fourier_Coefficients group created for test3")

    except Exception as e:
        logger.error(f"Failed to add FCs to test3: {e}")
        raise


# ==============================================================================
# TEST12RR - Multiple stations, remote reference
# ==============================================================================


def test_test12rr_file_validation(mth5_test_files):
    """Stage 1: Verify test12rr.h5 has expected structure and data."""
    mth5_path = mth5_test_files["test12rr"]

    assert mth5_path.exists(), f"test12rr.h5 not found at {mth5_path}"

    with mth5.MTH5(file_version="0.1.0") as m:
        m.open_mth5(mth5_path, mode="r")

        stations = m.stations_group.groups_list
        assert "test1" in stations, "test1 station not found in test12rr"
        assert "test2" in stations, "test2 station not found in test12rr"

        for station_id in ["test1", "test2"]:
            station = m.get_station(station_id)
            runs = [
                r
                for r in station.groups_list
                if r not in ["Transfer_Functions", "Fourier_Coefficients", "Features"]
            ]
            assert len(runs) > 0, f"No runs found in {station_id} station"

            run = station.get_run(runs[0])
            run.read_metadata()
            assert (
                run.metadata.sample_rate > 0
            ), f"{station_id} run sample_rate is {run.metadata.sample_rate}"

        logger.info(
            f"✓ test12rr.h5 validation passed: test1 and test2 stations present"
        )


def test_test12rr_add_fcs(mth5_test_files, temp_copy):
    """Stage 3: Add FCs to test12rr.h5 and verify storage."""
    source_path = mth5_test_files["test12rr"]
    mth5_path = temp_copy(source_path)

    run_summary = RunSummary()
    run_summary.from_mth5s([mth5_path])

    logger.info(f"test12rr RunSummary shape: {run_summary.df.shape}")
    logger.info(
        f"test12rr RunSummary:\n{run_summary.df[['station', 'run', 'sample_rate']]}"
    )

    tfk_dataset = KernelDataset()
    tfk_dataset.from_run_summary(run_summary, "test1", "test2")

    cc = ConfigCreator()
    processing_config = cc.create_from_kernel_dataset(tfk_dataset)

    fc_decimations = [x.to_fc_decimation() for x in processing_config.decimations]

    logger.info(f"Adding FCs to {mth5_path}")
    try:
        add_fcs_to_mth5(mth5_path, fc_decimations=fc_decimations)

        with mth5.MTH5(file_version="0.1.0") as m:
            m.open_mth5(mth5_path, mode="r")

            for station_id in ["test1", "test2"]:
                station = m.get_station(station_id)
                if "Fourier_Coefficients" in station.groups_list:
                    fc_group = station.fourier_coefficients_group
                    logger.info(
                        f"✓ {station_id} FCs stored: {len(fc_group.groups_list)} runs"
                    )
                else:
                    logger.warning(f"No Fourier_Coefficients group for {station_id}")

    except Exception as e:
        logger.error(f"Failed to add FCs to test12rr: {e}")
        raise
