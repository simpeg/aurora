# -*- coding: utf-8 -*-
"""
This module contains unittests for the xarray_helpers module.
"""

import numpy as np
import xarray as xr
import pytest

from aurora.time_series.xarray_helpers import handle_nan, nan_to_mean


def test_nan_to_mean_basic():
    """Test nan_to_mean replaces NaNs with mean per channel."""
    times = np.array([0, 1, 2, 3])
    data = np.array([1.0, np.nan, 3.0, 4.0])
    ds = xr.Dataset({"hx": ("time", data)}, coords={"time": times})

    ds_filled = nan_to_mean(ds.copy())
    # The mean ignoring NaN is (1+3+4)/3 = 2.666...
    expected = np.array([1.0, 2.66666667, 3.0, 4.0])
    assert np.allclose(ds_filled.hx.values, expected)
    # No NaNs should remain
    assert not np.any(np.isnan(ds_filled.hx.values))


def test_nan_to_mean_multiple_channels():
    """Test nan_to_mean with multiple channels and NaNs in different places."""
    times = np.array([0, 1, 2, 3])
    data_hx = np.array([1.0, np.nan, 3.0, 4.0])
    data_hy = np.array([np.nan, 2.0, 3.0, 4.0])
    ds = xr.Dataset(
        {
            "hx": ("time", data_hx),
            "hy": ("time", data_hy),
        },
        coords={"time": times},
    )

    ds_filled = nan_to_mean(ds.copy())
    expected_hx = np.array([1.0, 2.66666667, 3.0, 4.0])
    expected_hy = np.array([3.0, 2.0, 3.0, 4.0])
    assert np.allclose(ds_filled.hx.values, expected_hx)
    assert np.allclose(ds_filled.hy.values, expected_hy)
    assert not np.any(np.isnan(ds_filled.hx.values))
    assert not np.any(np.isnan(ds_filled.hy.values))


def test_handle_nan_basic():
    """Test basic functionality of handle_nan with NaN values."""
    # Create sample data with NaN values
    times = np.array([0, 1, 2, 3, 4])
    data_x = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
    data_y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])

    X = xr.Dataset({"hx": ("time", data_x)}, coords={"time": times})
    Y = xr.Dataset({"ex": ("time", data_y)}, coords={"time": times})

    # Test with X and Y only
    X_clean, Y_clean, _ = handle_nan(X, Y, None, drop_dim="time")

    # Check that NaN values were dropped
    assert len(X_clean.time) == 3
    assert len(Y_clean.time) == 3
    assert not np.any(np.isnan(X_clean.hx.values))
    assert not np.any(np.isnan(Y_clean.ex.values))


def test_handle_nan_with_remote_reference():
    """Test handle_nan with remote reference data."""
    # Create sample data
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


def test_handle_nan_time_mismatch():
    """Test handle_nan with time coordinate mismatches."""
    # Create sample data with slightly different timestamps
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
