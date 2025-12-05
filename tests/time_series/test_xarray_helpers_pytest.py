"""
Pytest suite for testing xarray_helpers module.

Tests cover:
- nan_to_mean: Replacing NaN values with channel means
- handle_nan: Dropping NaN values across multiple datasets
- time_axis_match: Checking time coordinate alignment
- Edge cases and parameter variations

Optimized for pytest-xdist parallel execution.
"""

import numpy as np
import pytest
import xarray as xr

from aurora.time_series.xarray_helpers import handle_nan, nan_to_mean, time_axis_match


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def basic_times():
    """Basic time coordinate array."""
    return np.array([0, 1, 2, 3])


@pytest.fixture
def extended_times():
    """Extended time coordinate array for edge case testing."""
    return np.array([0, 1, 2, 3, 4])


@pytest.fixture
def single_channel_dataset_with_nan(basic_times):
    """Dataset with single channel containing NaN values."""
    data = np.array([1.0, np.nan, 3.0, 4.0])
    return xr.Dataset({"hx": ("time", data)}, coords={"time": basic_times})


@pytest.fixture
def multi_channel_dataset_with_nan(basic_times):
    """Dataset with multiple channels containing NaN values in different locations."""
    data_hx = np.array([1.0, np.nan, 3.0, 4.0])
    data_hy = np.array([np.nan, 2.0, 3.0, 4.0])
    return xr.Dataset(
        {
            "hx": ("time", data_hx),
            "hy": ("time", data_hy),
        },
        coords={"time": basic_times},
    )


@pytest.fixture
def dataset_no_nan(basic_times):
    """Dataset without any NaN values."""
    data = np.array([1.0, 2.0, 3.0, 4.0])
    return xr.Dataset({"hx": ("time", data)}, coords={"time": basic_times})


@pytest.fixture
def dataset_all_nan(basic_times):
    """Dataset with all NaN values."""
    data = np.array([np.nan, np.nan, np.nan, np.nan])
    return xr.Dataset({"hx": ("time", data)}, coords={"time": basic_times})


# =============================================================================
# Test Classes
# =============================================================================


class TestNanToMean:
    """Test nan_to_mean function."""

    def test_nan_to_mean_basic(self, single_channel_dataset_with_nan):
        """Test nan_to_mean replaces NaNs with mean per channel."""
        ds_filled = nan_to_mean(single_channel_dataset_with_nan.copy())

        # The mean ignoring NaN is (1+3+4)/3 = 2.666...
        expected = np.array([1.0, 2.66666667, 3.0, 4.0])
        assert np.allclose(ds_filled.hx.values, expected)

        # No NaNs should remain
        assert not np.any(np.isnan(ds_filled.hx.values))

    def test_nan_to_mean_multiple_channels(self, multi_channel_dataset_with_nan):
        """Test nan_to_mean with multiple channels and NaNs in different places."""
        ds_filled = nan_to_mean(multi_channel_dataset_with_nan.copy())

        expected_hx = np.array([1.0, 2.66666667, 3.0, 4.0])
        expected_hy = np.array([3.0, 2.0, 3.0, 4.0])

        assert np.allclose(ds_filled.hx.values, expected_hx)
        assert np.allclose(ds_filled.hy.values, expected_hy)
        assert not np.any(np.isnan(ds_filled.hx.values))
        assert not np.any(np.isnan(ds_filled.hy.values))

    def test_nan_to_mean_no_nans(self, dataset_no_nan):
        """Test nan_to_mean with dataset containing no NaN values."""
        original_data = dataset_no_nan.hx.values.copy()
        ds_filled = nan_to_mean(dataset_no_nan.copy())

        # Data should remain unchanged
        assert np.allclose(ds_filled.hx.values, original_data)
        assert not np.any(np.isnan(ds_filled.hx.values))

    def test_nan_to_mean_all_nans(self, dataset_all_nan):
        """Test nan_to_mean with dataset containing all NaN values."""
        ds_filled = nan_to_mean(dataset_all_nan.copy())

        # Should replace with 0 (from np.nan_to_num of nanmean)
        assert np.allclose(ds_filled.hx.values, 0.0)

    def test_nan_to_mean_preserves_structure(self, multi_channel_dataset_with_nan):
        """Test that nan_to_mean preserves dataset structure."""
        ds_filled = nan_to_mean(multi_channel_dataset_with_nan.copy())

        # Check that coordinates are preserved
        assert np.allclose(
            ds_filled.time.values, multi_channel_dataset_with_nan.time.values
        )

        # Check that channels are preserved
        assert set(ds_filled.data_vars) == set(multi_channel_dataset_with_nan.data_vars)

    def test_nan_to_mean_single_nan_at_edges(self, subtests):
        """Test nan_to_mean with NaN at beginning and end."""
        times = np.array([0, 1, 2, 3, 4])

        test_cases = [
            (
                "nan_at_start",
                np.array([np.nan, 2.0, 3.0, 4.0, 5.0]),
                np.array([3.5, 2.0, 3.0, 4.0, 5.0]),
            ),
            (
                "nan_at_end",
                np.array([1.0, 2.0, 3.0, 4.0, np.nan]),
                np.array([1.0, 2.0, 3.0, 4.0, 2.5]),
            ),
            (
                "nan_at_both",
                np.array([np.nan, 2.0, 3.0, 4.0, np.nan]),
                np.array([3.0, 2.0, 3.0, 4.0, 3.0]),
            ),
        ]

        for name, data, expected in test_cases:
            with subtests.test(case=name):
                ds = xr.Dataset({"hx": ("time", data)}, coords={"time": times})
                ds_filled = nan_to_mean(ds.copy())
                assert np.allclose(ds_filled.hx.values, expected)


class TestHandleNanBasic:
    """Test basic handle_nan functionality."""

    def test_handle_nan_basic(self, extended_times):
        """Test basic functionality of handle_nan with NaN values."""
        data_x = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        data_y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])

        X = xr.Dataset({"hx": ("time", data_x)}, coords={"time": extended_times})
        Y = xr.Dataset({"ex": ("time", data_y)}, coords={"time": extended_times})

        # Test with X and Y only
        X_clean, Y_clean, _ = handle_nan(X, Y, None, drop_dim="time")

        # Check that NaN values were dropped
        assert len(X_clean.time) == 3
        assert len(Y_clean.time) == 3
        assert not np.any(np.isnan(X_clean.hx.values))
        assert not np.any(np.isnan(Y_clean.ex.values))

        # Check that correct values remain
        expected_times = np.array([0, 3, 4])
        assert np.allclose(X_clean.time.values, expected_times)
        assert np.allclose(Y_clean.time.values, expected_times)

    def test_handle_nan_x_only(self):
        """Test handle_nan with only X dataset (Y empty, RR None)."""
        times = np.array([0, 1, 2, 3])
        data_x = np.array([1.0, np.nan, 3.0, 4.0])

        X = xr.Dataset({"hx": ("time", data_x)}, coords={"time": times})
        # Empty dataset with matching time coordinate
        Y = xr.Dataset(coords={"time": times})

        X_clean, Y_clean, RR_clean = handle_nan(X, Y, None, drop_dim="time")

        # Check that NaN was dropped from X
        assert len(X_clean.time) == 3
        assert not np.any(np.isnan(X_clean.hx.values))

        # Y and RR should be empty datasets
        assert len(Y_clean.data_vars) == 0
        assert len(RR_clean.data_vars) == 0

    def test_handle_nan_no_nans(self):
        """Test handle_nan with datasets containing no NaN values."""
        times = np.array([0, 1, 2, 3])
        data_x = np.array([1.0, 2.0, 3.0, 4.0])
        data_y = np.array([1.0, 2.0, 3.0, 4.0])

        X = xr.Dataset({"hx": ("time", data_x)}, coords={"time": times})
        Y = xr.Dataset({"ex": ("time", data_y)}, coords={"time": times})

        X_clean, Y_clean, _ = handle_nan(X, Y, None, drop_dim="time")

        # All data should be preserved
        assert len(X_clean.time) == 4
        assert len(Y_clean.time) == 4
        assert np.allclose(X_clean.hx.values, data_x)
        assert np.allclose(Y_clean.ex.values, data_y)


class TestHandleNanRemoteReference:
    """Test handle_nan with remote reference data."""

    def test_handle_nan_with_remote_reference(self):
        """Test handle_nan with remote reference data."""
        times = np.array([0, 1, 2, 3])
        data_x = np.array([1.0, np.nan, 3.0, 4.0])
        data_y = np.array([1.0, 2.0, 3.0, 4.0])
        data_rr = np.array([1.0, 2.0, np.nan, 4.0])

        X = xr.Dataset({"hx": ("time", data_x)}, coords={"time": times})
        Y = xr.Dataset({"ex": ("time", data_y)}, coords={"time": times})
        RR = xr.Dataset({"hx": ("time", data_rr)}, coords={"time": times})

        # Test with all datasets
        X_clean, Y_clean, RR_clean = handle_nan(X, Y, RR, drop_dim="time")

        # Check that NaN values were dropped
        assert len(X_clean.time) == 2
        assert len(Y_clean.time) == 2
        assert len(RR_clean.time) == 2
        assert not np.any(np.isnan(X_clean.hx.values))
        assert not np.any(np.isnan(Y_clean.ex.values))
        assert not np.any(np.isnan(RR_clean.hx.values))

        # Check that the values are correct
        expected_times = np.array([0, 3])
        assert np.allclose(X_clean.time.values, expected_times)
        assert np.allclose(Y_clean.time.values, expected_times)
        assert np.allclose(RR_clean.time.values, expected_times)
        assert np.allclose(X_clean.hx.values, np.array([1.0, 4.0]))
        assert np.allclose(Y_clean.ex.values, np.array([1.0, 4.0]))
        assert np.allclose(RR_clean.hx.values, np.array([1.0, 4.0]))

    def test_handle_nan_remote_reference_only(self):
        """Test handle_nan with only remote reference having NaN."""
        times = np.array([0, 1, 2, 3])
        data_x = np.array([1.0, 2.0, 3.0, 4.0])
        data_y = np.array([1.0, 2.0, 3.0, 4.0])
        data_rr = np.array([1.0, np.nan, 3.0, 4.0])

        X = xr.Dataset({"hx": ("time", data_x)}, coords={"time": times})
        Y = xr.Dataset({"ex": ("time", data_y)}, coords={"time": times})
        RR = xr.Dataset({"hy": ("time", data_rr)}, coords={"time": times})

        X_clean, Y_clean, RR_clean = handle_nan(X, Y, RR, drop_dim="time")

        # Only time index 1 should be dropped
        assert len(X_clean.time) == 3
        assert len(Y_clean.time) == 3
        assert len(RR_clean.time) == 3

        expected_times = np.array([0, 2, 3])
        assert np.allclose(X_clean.time.values, expected_times)

    def test_handle_nan_channel_name_preservation(self):
        """Test that channel names are preserved correctly with RR."""
        times = np.array([0, 1, 2])
        data = np.array([1.0, 2.0, 3.0])

        X = xr.Dataset({"hx": ("time", data)}, coords={"time": times})
        Y = xr.Dataset({"ex": ("time", data)}, coords={"time": times})
        RR = xr.Dataset(
            {"hx": ("time", data), "hy": ("time", data)}, coords={"time": times}
        )

        X_clean, Y_clean, RR_clean = handle_nan(X, Y, RR, drop_dim="time")

        # Check channel names
        assert "hx" in X_clean.data_vars
        assert "ex" in Y_clean.data_vars
        assert "hx" in RR_clean.data_vars
        assert "hy" in RR_clean.data_vars

        # RR channels should not have "remote_" prefix in output
        assert "remote_hx" not in RR_clean.data_vars


class TestHandleNanTimeMismatch:
    """Test handle_nan with time coordinate mismatches."""

    def test_handle_nan_time_mismatch(self):
        """Test handle_nan with time coordinate mismatches."""
        times_x = np.array([0, 1, 2, 3])
        times_rr = times_x + 0.1  # Small offset
        data_x = np.array([1.0, 2.0, 3.0, 4.0])
        data_rr = np.array([1.0, 2.0, 3.0, 4.0])

        X = xr.Dataset({"hx": ("time", data_x)}, coords={"time": times_x})
        RR = xr.Dataset({"hx": ("time", data_rr)}, coords={"time": times_rr})

        # Test handling of time mismatch
        X_clean, _, RR_clean = handle_nan(X, None, RR, drop_dim="time")

        # Check that data was preserved despite time mismatch
        assert len(X_clean.time) == 4
        assert "hx" in RR_clean.data_vars
        assert np.allclose(RR_clean.hx.values, data_rr)

        # Check that the time values match X's time values
        assert np.allclose(RR_clean.time.values, X_clean.time.values)

    def test_handle_nan_partial_time_mismatch(self):
        """Test handle_nan when only some time coordinates mismatch."""
        times_x = np.array([0.0, 1.0, 2.0, 3.0])
        times_rr = np.array([0.0, 1.0, 2.0001, 3.0])  # Slight mismatch at index 2
        data_x = np.array([1.0, 2.0, 3.0, 4.0])
        data_rr = np.array([1.0, 2.0, 3.0, 4.0])

        X = xr.Dataset({"hx": ("time", data_x)}, coords={"time": times_x})
        RR = xr.Dataset({"hy": ("time", data_rr)}, coords={"time": times_rr})

        # Should handle this with left join
        X_clean, _, RR_clean = handle_nan(X, None, RR, drop_dim="time")

        assert len(X_clean.time) == 4
        assert len(RR_clean.time) == 4


class TestTimeAxisMatch:
    """Test time_axis_match function."""

    def test_time_axis_match_exact(self):
        """Test time_axis_match when all axes match exactly."""
        times = np.array([0, 1, 2, 3])
        data = np.array([1.0, 2.0, 3.0, 4.0])

        X = xr.Dataset({"hx": ("time", data)}, coords={"time": times})
        Y = xr.Dataset({"ex": ("time", data)}, coords={"time": times})
        RR = xr.Dataset({"hy": ("time", data)}, coords={"time": times})

        assert time_axis_match(X, Y, RR) is True

    def test_time_axis_match_xy_only(self):
        """Test time_axis_match with only X and Y."""
        times = np.array([0, 1, 2, 3])
        data = np.array([1.0, 2.0, 3.0, 4.0])

        X = xr.Dataset({"hx": ("time", data)}, coords={"time": times})
        Y = xr.Dataset({"ex": ("time", data)}, coords={"time": times})

        assert time_axis_match(X, Y, None) is True

    def test_time_axis_match_x_rr_only(self):
        """Test time_axis_match with only X and RR."""
        times = np.array([0, 1, 2, 3])
        data = np.array([1.0, 2.0, 3.0, 4.0])

        X = xr.Dataset({"hx": ("time", data)}, coords={"time": times})
        RR = xr.Dataset({"hy": ("time", data)}, coords={"time": times})

        assert time_axis_match(X, None, RR) is True

    def test_time_axis_match_mismatch(self):
        """Test time_axis_match when axes do not match."""
        times_x = np.array([0, 1, 2, 3])
        times_rr = np.array([0, 1, 2, 4])  # Different last value
        data = np.array([1.0, 2.0, 3.0, 4.0])

        X = xr.Dataset({"hx": ("time", data)}, coords={"time": times_x})
        RR = xr.Dataset({"hy": ("time", data)}, coords={"time": times_rr})

        assert time_axis_match(X, None, RR) is False

    def test_time_axis_match_different_lengths(self):
        """Test time_axis_match with different length time axes."""
        times_x = np.array([0, 1, 2, 3])
        times_y = np.array([0, 1, 2])

        X = xr.Dataset(
            {"hx": ("time", np.array([1.0, 2.0, 3.0, 4.0]))}, coords={"time": times_x}
        )
        Y = xr.Dataset(
            {"ex": ("time", np.array([1.0, 2.0, 3.0]))}, coords={"time": times_y}
        )
        RR = xr.Dataset(
            {"hy": ("time", np.array([1.0, 2.0, 3.0, 4.0]))}, coords={"time": times_x}
        )

        # Use RR instead of None to avoid AttributeError
        assert time_axis_match(X, Y, RR) is False

    def test_time_axis_match_float_precision(self):
        """Test time_axis_match with floating point precision issues."""
        times_x = np.array([0.0, 0.1, 0.2, 0.3])
        times_rr = times_x + 1e-10  # Very small difference
        data = np.array([1.0, 2.0, 3.0, 4.0])

        X = xr.Dataset({"hx": ("time", data)}, coords={"time": times_x})
        RR = xr.Dataset({"hy": ("time", data)}, coords={"time": times_rr})

        # Should not match due to precision difference
        assert time_axis_match(X, None, RR) is False


class TestHandleNanMultipleChannels:
    """Test handle_nan with multiple channels in each dataset."""

    def test_handle_nan_multiple_channels_x_y(self):
        """Test handle_nan with multiple channels in X and Y."""
        times = np.array([0, 1, 2, 3])
        data_hx = np.array([1.0, np.nan, 3.0, 4.0])
        data_hy = np.array([1.0, 2.0, np.nan, 4.0])
        data_ex = np.array([np.nan, 2.0, 3.0, 4.0])
        data_ey = np.array([1.0, 2.0, 3.0, 4.0])

        X = xr.Dataset(
            {
                "hx": ("time", data_hx),
                "hy": ("time", data_hy),
            },
            coords={"time": times},
        )

        Y = xr.Dataset(
            {
                "ex": ("time", data_ex),
                "ey": ("time", data_ey),
            },
            coords={"time": times},
        )

        X_clean, Y_clean, _ = handle_nan(X, Y, None, drop_dim="time")

        # Only time index 3 has no NaN in any channel
        assert len(X_clean.time) == 1
        assert len(Y_clean.time) == 1
        assert X_clean.time.values[0] == 3

    def test_handle_nan_preserves_all_channels(self):
        """Test that all channels are preserved after NaN handling."""
        times = np.array([0, 1, 2])
        data = np.array([1.0, 2.0, 3.0])

        X = xr.Dataset(
            {
                "hx": ("time", data),
                "hy": ("time", data),
                "hz": ("time", data),
            },
            coords={"time": times},
        )

        Y = xr.Dataset(
            {
                "ex": ("time", data),
                "ey": ("time", data),
            },
            coords={"time": times},
        )

        X_clean, Y_clean, _ = handle_nan(X, Y, None, drop_dim="time")

        # All channels should be preserved
        assert set(X_clean.data_vars) == {"hx", "hy", "hz"}
        assert set(Y_clean.data_vars) == {"ex", "ey"}


class TestHandleNanEdgeCases:
    """Test edge cases for handle_nan."""

    def test_handle_nan_empty_dataset(self):
        """Test handle_nan with empty Y and RR."""
        times = np.array([0, 1, 2, 3])
        data = np.array([1.0, 2.0, 3.0, 4.0])

        X = xr.Dataset({"hx": ("time", data)}, coords={"time": times})
        # Empty dataset with matching time coordinate
        Y = xr.Dataset(coords={"time": times})

        X_clean, Y_clean, RR_clean = handle_nan(X, Y, None, drop_dim="time")

        # X should be unchanged
        assert len(X_clean.time) == 4
        assert np.allclose(X_clean.hx.values, data)

        # Y and RR should be empty
        assert len(Y_clean.data_vars) == 0
        assert len(RR_clean.data_vars) == 0

    def test_handle_nan_all_nans_dropped(self):
        """Test handle_nan when all rows have at least one NaN."""
        times = np.array([0, 1, 2])
        data_x = np.array([np.nan, 2.0, 3.0])
        data_y = np.array([1.0, np.nan, 3.0])
        data_rr = np.array([1.0, 2.0, np.nan])

        X = xr.Dataset({"hx": ("time", data_x)}, coords={"time": times})
        Y = xr.Dataset({"ex": ("time", data_y)}, coords={"time": times})
        RR = xr.Dataset({"hy": ("time", data_rr)}, coords={"time": times})

        X_clean, Y_clean, RR_clean = handle_nan(X, Y, RR, drop_dim="time")

        # No rows should remain
        assert len(X_clean.time) == 0
        assert len(Y_clean.time) == 0
        assert len(RR_clean.time) == 0

    def test_handle_nan_different_drop_dim(self):
        """Test handle_nan still works when drop_dim is specified (even though time_axis_match assumes 'time')."""
        # Note: time_axis_match function assumes 'time' dimension exists, so we use 'time' here
        # but test that drop_dim parameter is respected
        times = np.array([0, 1, 2, 3])
        data_x = np.array([1.0, np.nan, 3.0, 4.0])
        data_y = np.array([1.0, 2.0, 3.0, 4.0])

        X = xr.Dataset({"hx": ("time", data_x)}, coords={"time": times})
        Y = xr.Dataset({"ex": ("time", data_y)}, coords={"time": times})

        X_clean, Y_clean, _ = handle_nan(X, Y, None, drop_dim="time")

        # NaN at index 1 should be dropped
        assert len(X_clean.time) == 3
        assert len(Y_clean.time) == 3

        expected_times = np.array([0, 2, 3])
        assert np.allclose(X_clean.time.values, expected_times)


class TestHandleNanDataIntegrity:
    """Test that handle_nan preserves data integrity."""

    def test_handle_nan_values_correctness(self):
        """Test that correct values are preserved after dropping NaNs."""
        times = np.array([0, 1, 2, 3, 4])
        data_x = np.array([10.0, np.nan, 30.0, np.nan, 50.0])
        data_y = np.array([100.0, 200.0, np.nan, 400.0, 500.0])

        X = xr.Dataset({"hx": ("time", data_x)}, coords={"time": times})
        Y = xr.Dataset({"ex": ("time", data_y)}, coords={"time": times})

        X_clean, Y_clean, _ = handle_nan(X, Y, None, drop_dim="time")

        # Only times 0 and 4 have no NaN in any channel
        expected_times = np.array([0, 4])
        expected_x = np.array([10.0, 50.0])
        expected_y = np.array([100.0, 500.0])

        assert np.allclose(X_clean.time.values, expected_times)
        assert np.allclose(X_clean.hx.values, expected_x)
        assert np.allclose(Y_clean.ex.values, expected_y)

    def test_handle_nan_original_unchanged(self):
        """Test that original datasets are not modified by handle_nan."""
        times = np.array([0, 1, 2, 3])
        data_x = np.array([1.0, np.nan, 3.0, 4.0])
        data_y = np.array([1.0, 2.0, 3.0, 4.0])

        X = xr.Dataset({"hx": ("time", data_x)}, coords={"time": times})
        Y = xr.Dataset({"ex": ("time", data_y)}, coords={"time": times})

        # Store original values
        original_x_len = len(X.time)
        original_y_len = len(Y.time)

        # Call handle_nan
        X_clean, Y_clean, _ = handle_nan(X, Y, None, drop_dim="time")

        # Original datasets should be unchanged
        assert len(X.time) == original_x_len
        assert len(Y.time) == original_y_len
        assert np.isnan(X.hx.values[1])  # NaN still present
