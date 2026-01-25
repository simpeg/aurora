"""
Tests for slicing RunTS objects using timestamps.

This will get moved into MTH5.
"""

import datetime

from loguru import logger
from mth5.utils.helpers import initialize_mth5


def test_can_slice_a_run_ts_using_timestamp(worker_safe_test1_h5, subtests):
    """Test that RunTS can be properly sliced using timestamps."""
    # Open the MTH5 file
    mth5_obj = initialize_mth5(worker_safe_test1_h5, "r")

    try:
        df = mth5_obj.channel_summary.to_dataframe()

        # Get the run
        try:
            run_001 = mth5_obj.get_run(station_name="test1", run_name="001")
        except ValueError:
            # This can happen on local machine
            run_001 = mth5_obj.get_run(
                station_name="test1",
                run_name="001",
                survey=mth5_obj.surveys_group.groups_list[0],
            )

        # Get the full run without slicing
        run_ts_full = run_001.to_runts()
        full_length = len(run_ts_full.dataset.ex.data)

        start = df.iloc[0].start
        end = df.iloc[0].end

        logger.info(f"Full run has {full_length} samples")
        logger.info(f"Start: {start}, End: {end}")

        # Test 1: Slice with exact start and end times
        with subtests.test(msg="exact_start_end"):
            run_ts_exact = run_001.to_runts(start=start, end=end)
            exact_length = len(run_ts_exact.dataset.ex.data)
            logger.info(f"Exact slice has {exact_length} samples")

            # Should have the same length as full run since we use exact bounds
            assert (
                exact_length == full_length
            ), f"Expected {full_length} samples with exact bounds, got {exact_length}"

        # Test 2: Slice with end + 499999 microseconds (less than one sample at 1 Hz)
        with subtests.test(msg="end_plus_499999_microseconds"):
            run_ts_sub_sample = run_001.to_runts(
                start=start, end=end + datetime.timedelta(microseconds=499999)
            )
            sub_sample_length = len(run_ts_sub_sample.dataset.ex.data)
            logger.info(f"End + 499999μs slice has {sub_sample_length} samples")

            # Should still have same length since we haven't crossed a sample boundary
            assert (
                sub_sample_length == full_length
            ), f"Expected {full_length} samples (sub-sample extension), got {sub_sample_length}"

        # Test 3: Slice with end + 500000 microseconds (half a sample at 1 Hz)
        with subtests.test(msg="end_plus_500000_microseconds"):
            run_ts_one_more = run_001.to_runts(
                start=start, end=end + datetime.timedelta(microseconds=500000)
            )
            one_more_length = len(run_ts_one_more.dataset.ex.data)
            logger.info(f"End + 500000μs slice has {one_more_length} samples")

            # The slicing appears to be inclusive of the exact end boundary
            # so adding 0.5 seconds doesn't add a new sample
            assert (
                one_more_length == full_length
            ), f"Expected {full_length} samples, got {one_more_length}"

        # Test 4: Verify that sliced data starts at correct time
        with subtests.test(msg="sliced_start_time"):
            run_ts_sliced = run_001.to_runts(start=start, end=end)
            sliced_start = run_ts_sliced.dataset.time.data[0]

            # Convert to comparable format - normalize timezones
            import pandas as pd

            expected_start = pd.Timestamp(start).tz_localize(None)
            actual_start = pd.Timestamp(sliced_start).tz_localize(None)

            logger.info(
                f"Expected start: {expected_start}, Actual start: {actual_start}"
            )
            assert (
                actual_start == expected_start
            ), f"Start time mismatch: expected {expected_start}, got {actual_start}"
    finally:
        mth5_obj.close_mth5()


def test_partial_run_slice(worker_safe_test1_h5):
    """Test slicing a partial section of a run."""
    # Open the MTH5 file
    mth5_obj = initialize_mth5(worker_safe_test1_h5, "r")

    try:
        df = mth5_obj.channel_summary.to_dataframe()

        # Get the run
        try:
            run_001 = mth5_obj.get_run(station_name="test1", run_name="001")
        except ValueError:
            run_001 = mth5_obj.get_run(
                station_name="test1",
                run_name="001",
                survey=mth5_obj.surveys_group.groups_list[0],
            )

        start = df.iloc[0].start
        end = df.iloc[0].end

        # Get full run
        run_ts_full = run_001.to_runts()
        full_length = len(run_ts_full.dataset.ex.data)

        # Slice the middle 50% of the run
        duration = end - start
        middle_start = start + duration * 0.25
        middle_end = start + duration * 0.75

        run_ts_middle = run_001.to_runts(start=middle_start, end=middle_end)
        middle_length = len(run_ts_middle.dataset.ex.data)

        logger.info(f"Full run: {full_length} samples")
        logger.info(f"Middle 50% slice: {middle_length} samples")

        # Middle section should be approximately 50% of full length
        # Allow for some tolerance due to rounding
        expected_middle = full_length * 0.5
        tolerance = full_length * 0.01  # 1% tolerance

        assert (
            abs(middle_length - expected_middle) <= tolerance
        ), f"Expected ~{expected_middle} samples in middle 50%, got {middle_length}"

        # Verify start time of sliced data
        import pandas as pd

        sliced_start = pd.Timestamp(run_ts_middle.dataset.time.data[0]).tz_localize(
            None
        )
        expected_start = pd.Timestamp(middle_start).tz_localize(None)

        assert (
            sliced_start == expected_start
        ), f"Start time mismatch: expected {expected_start}, got {sliced_start}"
    finally:
        mth5_obj.close_mth5()
