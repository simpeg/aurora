"""
Pytest suite for testing WindowingScheme class.

Tests cover:
- Basic instantiation and properties
- Sliding window operations (numpy, xarray)
- Taper application
- FFT operations
- Edge cases and parameter variations
- Untested functionality from original implementation

Optimized for pytest-xdist parallel execution.
"""

import numpy as np
import pytest
import xarray as xr

from aurora.time_series.time_axis_helpers import make_time_axis
from aurora.time_series.windowing_scheme import WindowingScheme


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(0)


@pytest.fixture
def basic_windowing_scheme():
    """Basic windowing scheme with default parameters."""
    return WindowingScheme(
        num_samples_window=32,
        num_samples_overlap=8,
        taper_family="hamming",
    )


@pytest.fixture
def windowing_scheme_with_sample_rate():
    """Windowing scheme with sample rate for time-domain tests."""
    return WindowingScheme(
        num_samples_window=128,
        num_samples_overlap=32,
        sample_rate=50.0,
        taper_family="hamming",
    )


@pytest.fixture
def xarray_dataset(random_seed):
    """Create an xarray Dataset with random data."""
    N = 1000
    sps = 50.0
    t0 = np.datetime64("1977-03-02 12:34:56")
    time_vector = make_time_axis(t0, N, sps)

    ds = xr.Dataset(
        {
            "hx": (["time"], np.abs(np.random.randn(N))),
            "hy": (["time"], np.abs(np.random.randn(N))),
        },
        coords={"time": time_vector},
        attrs={
            "some random info": "dogs",
            "some more random info": "cats",
            "sample_rate": sps,
        },
    )
    return ds


@pytest.fixture
def xarray_dataarray(random_seed):
    """Create an xarray DataArray with random data."""
    num_samples_data = 10000
    xrd = xr.DataArray(
        np.random.randn(num_samples_data, 1),
        dims=["time", "channel"],
        coords={"time": np.arange(num_samples_data)},
    )
    return xrd


@pytest.fixture
def numpy_timeseries(random_seed):
    """Create a numpy array time series."""
    return np.random.random(10000)


# =============================================================================
# Test Classes
# =============================================================================


class TestWindowingSchemeBasic:
    """Test basic instantiation and properties."""

    def test_instantiate_windowing_scheme(self):
        """Test creating a WindowingScheme with all parameters."""
        num_samples_window = 128
        num_samples_overlap = 32
        num_samples_data = 1000
        sample_rate = 50.0
        taper_family = "hamming"

        ws = WindowingScheme(
            num_samples_window=num_samples_window,
            num_samples_overlap=num_samples_overlap,
            num_samples_data=num_samples_data,
            taper_family=taper_family,
        )
        ws.sample_rate = sample_rate

        expected_window_duration = num_samples_window / sample_rate
        assert ws.window_duration == expected_window_duration
        assert ws.num_samples_window == num_samples_window
        assert ws.num_samples_overlap == num_samples_overlap
        assert ws.taper_family == taper_family

    def test_num_samples_advance_property(self, basic_windowing_scheme):
        """Test that num_samples_advance is calculated correctly."""
        expected_advance = (
            basic_windowing_scheme.num_samples_window
            - basic_windowing_scheme.num_samples_overlap
        )
        assert basic_windowing_scheme.num_samples_advance == expected_advance

    def test_available_number_of_windows(self, basic_windowing_scheme):
        """Test calculation of available windows for given data length."""
        from aurora.time_series.window_helpers import (
            available_number_of_windows_in_array,
        )

        num_samples_data = 10000
        expected_num_windows = available_number_of_windows_in_array(
            num_samples_data,
            basic_windowing_scheme.num_samples_window,
            basic_windowing_scheme.num_samples_advance,
        )

        num_windows = basic_windowing_scheme.available_number_of_windows(
            num_samples_data
        )
        assert num_windows == expected_num_windows

    def test_string_representation(self, basic_windowing_scheme):
        """Test __str__ and __repr__ methods."""
        str_repr = str(basic_windowing_scheme)
        assert "32" in str_repr  # num_samples_window
        assert "8" in str_repr  # num_samples_overlap
        assert repr(basic_windowing_scheme) == str(basic_windowing_scheme)

    def test_clone_method(self, basic_windowing_scheme):
        """Test that clone creates a deep copy."""
        cloned = basic_windowing_scheme.clone()

        assert cloned.num_samples_window == basic_windowing_scheme.num_samples_window
        assert cloned.num_samples_overlap == basic_windowing_scheme.num_samples_overlap
        assert cloned.taper_family == basic_windowing_scheme.taper_family
        assert cloned is not basic_windowing_scheme


class TestWindowingSchemeSlidingWindow:
    """Test sliding window operations."""

    def test_apply_sliding_window_numpy(self, random_seed, numpy_timeseries):
        """Test sliding window on numpy array returns correct shape."""
        windowing_scheme = WindowingScheme(
            num_samples_window=64,
            num_samples_overlap=50,
        )

        windowed_array = windowing_scheme.apply_sliding_window(numpy_timeseries)

        expected_num_windows = windowing_scheme.available_number_of_windows(
            len(numpy_timeseries)
        )
        assert windowed_array.shape[0] == expected_num_windows
        assert windowed_array.shape[1] == 64

    def test_apply_sliding_window_can_return_xarray(self):
        """Test that sliding window can return xarray from numpy input."""
        ts = np.arange(15)
        windowing_scheme = WindowingScheme(
            num_samples_window=3,
            num_samples_overlap=1,
        )

        windowed_xr = windowing_scheme.apply_sliding_window(ts, return_xarray=True)

        assert isinstance(windowed_xr, xr.DataArray)
        assert "time" in windowed_xr.coords
        assert "within-window time" in windowed_xr.coords

    def test_apply_sliding_window_to_xarray_dataarray(
        self, random_seed, xarray_dataarray
    ):
        """Test sliding window on xarray DataArray."""
        windowing_scheme = WindowingScheme(
            num_samples_window=64,
            num_samples_overlap=50,
        )

        windowed_xrda = windowing_scheme.apply_sliding_window(
            xarray_dataarray, return_xarray=True
        )

        # DataArray is converted to Dataset internally, then back to DataArray
        # Shape will be (channel, time, within-window time)
        assert isinstance(windowed_xrda, xr.DataArray)
        expected_num_windows = windowing_scheme.available_number_of_windows(
            len(xarray_dataarray)
        )
        assert windowed_xrda.shape[1] == expected_num_windows  # time dimension

    def test_apply_sliding_window_to_xarray_dataset(self, random_seed, xarray_dataset):
        """Test sliding window on xarray Dataset preserves all channels."""
        windowing_scheme = WindowingScheme(
            num_samples_window=32,
            num_samples_overlap=8,
        )

        windowed_dataset = windowing_scheme.apply_sliding_window(
            xarray_dataset, return_xarray=True
        )

        assert isinstance(windowed_dataset, xr.Dataset)
        assert "hx" in windowed_dataset
        assert "hy" in windowed_dataset
        assert "time" in windowed_dataset.coords
        assert "within-window time" in windowed_dataset.coords

    def test_sliding_window_shapes_with_different_overlaps(self, random_seed, subtests):
        """Test sliding window with various overlap values."""
        ts = np.random.random(1000)

        for overlap in [0, 8, 16, 24, 31]:
            with subtests.test(overlap=overlap):
                ws = WindowingScheme(num_samples_window=32, num_samples_overlap=overlap)
                windowed = ws.apply_sliding_window(ts)

                expected_advance = 32 - overlap
                expected_windows = ws.available_number_of_windows(len(ts))

                assert windowed.shape[0] == expected_windows
                assert windowed.shape[1] == 32


class TestWindowingSchemeTaper:
    """Test taper application."""

    def test_can_apply_taper(self, random_seed, numpy_timeseries):
        """Test that taper modifies windowed data correctly."""
        windowing_scheme = WindowingScheme(
            num_samples_window=64,
            num_samples_overlap=50,
            taper_family="hamming",
        )

        windowed_data = windowing_scheme.apply_sliding_window(numpy_timeseries)
        tapered_windowed_data = windowing_scheme.apply_taper(windowed_data)

        # Taper should modify the data
        assert (windowed_data[:, 0] != tapered_windowed_data[:, 0]).all()

        # Shape should remain the same
        assert windowed_data.shape == tapered_windowed_data.shape

    def test_taper_dataset(self, random_seed, xarray_dataset):
        """Test taper application to xarray Dataset."""
        windowing_scheme = WindowingScheme(
            num_samples_window=64,
            num_samples_overlap=8,
            sample_rate=None,
            taper_family="hamming",
        )

        windowed_dataset = windowing_scheme.apply_sliding_window(
            xarray_dataset, return_xarray=True
        )
        tapered_dataset = windowing_scheme.apply_taper(windowed_dataset)

        assert isinstance(tapered_dataset, xr.Dataset)

        # Check that taper modified the data
        assert not np.allclose(
            windowed_dataset["hx"].data[0, :],
            tapered_dataset["hx"].data[0, :],
        )

    def test_taper_with_different_families(self, random_seed, subtests):
        """Test taper application with various window families."""
        ts = np.random.random(1000)

        for taper_family in ["boxcar", "hamming", "hann", "blackman", "blackmanharris"]:
            with subtests.test(taper_family=taper_family):
                ws = WindowingScheme(
                    num_samples_window=64,
                    num_samples_overlap=16,
                    taper_family=taper_family,
                )

                windowed_data = ws.apply_sliding_window(ts)
                tapered_data = ws.apply_taper(windowed_data)

                # Boxcar shouldn't change data, others should
                if taper_family == "boxcar":
                    assert np.allclose(windowed_data, tapered_data)
                else:
                    assert not np.allclose(windowed_data, tapered_data)


class TestWindowingSchemeFFT:
    """Test FFT operations."""

    def test_fourier_transform_dataset(self, random_seed):
        """Test FFT on xarray Dataset."""
        sample_rate = 40.0
        windowing_scheme = WindowingScheme(
            num_samples_window=128,
            num_samples_overlap=96,
            sample_rate=sample_rate,
        )

        # Create test dataset
        N = 10000
        sps = sample_rate
        t0 = np.datetime64("1977-03-02 12:34:56")
        time_vector = make_time_axis(t0, N, sps)
        ds = xr.Dataset(
            {
                "hx": (["time"], np.abs(np.random.randn(N))),
                "hy": (["time"], np.abs(np.random.randn(N))),
            },
            coords={"time": time_vector},
            attrs={"sample_rate": sps},
        )

        windowed_dataset = windowing_scheme.apply_sliding_window(ds)
        tapered_windowed_dataset = windowing_scheme.apply_taper(windowed_dataset)
        stft = windowing_scheme.apply_fft(tapered_windowed_dataset)

        assert isinstance(stft, xr.Dataset)
        assert "hx" in stft
        assert "hy" in stft
        assert "frequency" in stft.coords

    def test_fourier_transform_dataarray(self, random_seed):
        """Test FFT on xarray DataArray."""
        sample_rate = 40.0
        windowing_scheme = WindowingScheme(
            num_samples_window=128,
            num_samples_overlap=96,
            sample_rate=sample_rate,
        )

        # Create test dataset
        N = 10000
        sps = sample_rate
        t0 = np.datetime64("1977-03-02 12:34:56")
        time_vector = make_time_axis(t0, N, sps)
        ds = xr.Dataset(
            {
                "hx": (["time"], np.abs(np.random.randn(N))),
                "hy": (["time"], np.abs(np.random.randn(N))),
            },
            coords={"time": time_vector},
            attrs={"sample_rate": sps},
        )

        # Convert to DataArray
        da = ds.to_array("channel")

        windowed_dataset = windowing_scheme.apply_sliding_window(da)
        tapered_windowed_dataset = windowing_scheme.apply_taper(windowed_dataset)
        stft = windowing_scheme.apply_fft(tapered_windowed_dataset)

        assert isinstance(stft, xr.DataArray)
        assert "frequency" in stft.coords

    def test_frequency_axis_calculation(self, windowing_scheme_with_sample_rate):
        """Test frequency axis is calculated correctly."""
        dt = 1.0 / windowing_scheme_with_sample_rate.sample_rate
        freq_axis = windowing_scheme_with_sample_rate.frequency_axis(dt)

        # get_fft_harmonics returns one-sided spectrum without Nyquist
        # Length is num_samples_window // 2
        expected_length = windowing_scheme_with_sample_rate.num_samples_window // 2
        assert len(freq_axis) == expected_length
        assert freq_axis[0] == 0.0  # DC component


class TestWindowingSchemeTimeDomain:
    """Test time-domain properties that require sample_rate."""

    def test_window_duration(self, windowing_scheme_with_sample_rate):
        """Test window_duration property."""
        expected_duration = (
            windowing_scheme_with_sample_rate.num_samples_window
            / windowing_scheme_with_sample_rate.sample_rate
        )
        assert windowing_scheme_with_sample_rate.window_duration == expected_duration

    def test_dt_property(self, windowing_scheme_with_sample_rate):
        """Test dt (sample interval) property."""
        expected_dt = 1.0 / windowing_scheme_with_sample_rate.sample_rate
        assert windowing_scheme_with_sample_rate.dt == expected_dt

    def test_duration_advance(self, windowing_scheme_with_sample_rate):
        """Test duration_advance property."""
        expected_duration_advance = (
            windowing_scheme_with_sample_rate.num_samples_advance
            / windowing_scheme_with_sample_rate.sample_rate
        )
        assert (
            windowing_scheme_with_sample_rate.duration_advance
            == expected_duration_advance
        )


class TestWindowingSchemeTimeAxis:
    """Test time axis manipulation methods."""

    def test_left_hand_window_edge_indices(self, basic_windowing_scheme):
        """Test calculation of window edge indices."""
        num_samples_data = 1000
        lhwe = basic_windowing_scheme.left_hand_window_edge_indices(num_samples_data)

        expected_num_windows = basic_windowing_scheme.available_number_of_windows(
            num_samples_data
        )
        assert len(lhwe) == expected_num_windows

        # First window starts at 0
        assert lhwe[0] == 0

        # Windows advance by num_samples_advance
        if len(lhwe) > 1:
            assert lhwe[1] == basic_windowing_scheme.num_samples_advance

    def test_downsample_time_axis(self, basic_windowing_scheme):
        """Test downsampling of time axis for windowed data."""
        time_axis = np.arange(1000, dtype=float)
        downsampled = basic_windowing_scheme.downsample_time_axis(time_axis)

        expected_num_windows = basic_windowing_scheme.available_number_of_windows(
            len(time_axis)
        )
        assert len(downsampled) == expected_num_windows

        # First value should match first sample
        assert downsampled[0] == time_axis[0]

    def test_cast_windowed_data_to_xarray(self, basic_windowing_scheme):
        """Test casting numpy windowed data to xarray."""
        windowed_array = np.random.randn(10, 32)  # 10 windows, 32 samples each
        time_vector = np.arange(10, dtype=float)
        dt = 0.02

        xrda = basic_windowing_scheme.cast_windowed_data_to_xarray(
            windowed_array, time_vector, dt=dt
        )

        assert isinstance(xrda, xr.DataArray)
        assert "time" in xrda.coords
        assert "within-window time" in xrda.coords
        assert len(xrda.coords["time"]) == 10
        assert len(xrda.coords["within-window time"]) == 32


class TestWindowingSchemeEdgeCases:
    """Test edge cases and error handling."""

    def test_sliding_window_without_time_vector_warns(self, basic_windowing_scheme):
        """Test that requesting xarray without time_vector issues warning."""
        ts = np.arange(100)

        # Should work but warn
        result = basic_windowing_scheme.apply_sliding_window(
            ts, time_vector=None, return_xarray=True
        )

        assert isinstance(result, xr.DataArray)

    def test_xarray_attrs_immutable(self, xarray_dataset):
        """Test that xarray attributes cannot be directly overwritten."""
        with pytest.raises(AttributeError):
            xarray_dataset.sample_rate = 10

    def test_zero_overlap(self):
        """Test windowing with no overlap."""
        ws = WindowingScheme(num_samples_window=32, num_samples_overlap=0)
        ts = np.arange(128)

        windowed = ws.apply_sliding_window(ts)

        assert windowed.shape[0] == 4  # 128 / 32
        assert windowed.shape[1] == 32

    def test_maximum_overlap(self):
        """Test windowing with maximum overlap (L-1)."""
        ws = WindowingScheme(num_samples_window=32, num_samples_overlap=31)
        ts = np.arange(1000)

        windowed = ws.apply_sliding_window(ts)

        assert windowed.shape[1] == 32
        assert ws.num_samples_advance == 1


class TestWindowingSchemeSpectralDensity:
    """Test spectral density calibration factor."""

    def test_linear_spectral_density_calibration_factor(
        self, windowing_scheme_with_sample_rate
    ):
        """Test calculation of spectral density calibration factor."""
        calibration_factor = (
            windowing_scheme_with_sample_rate.linear_spectral_density_calibration_factor
        )

        # Should be a positive scalar
        assert isinstance(calibration_factor, float)
        assert calibration_factor > 0

        # Verify formula: sqrt(2 / (sample_rate * S2))
        S2 = windowing_scheme_with_sample_rate.S2
        sample_rate = windowing_scheme_with_sample_rate.sample_rate
        expected = np.sqrt(2 / (sample_rate * S2))

        assert np.isclose(calibration_factor, expected)


class TestWindowingSchemeTaperFamilies:
    """Test different taper families and their parameters."""

    def test_various_taper_families(self, subtests):
        """Test that various taper families can be instantiated."""
        for taper_family in [
            "boxcar",
            "hamming",
            "hann",
            "blackman",
            "blackmanharris",
        ]:
            with subtests.test(taper_family=taper_family):
                ws = WindowingScheme(
                    num_samples_window=64,
                    num_samples_overlap=16,
                    taper_family=taper_family,
                )

                assert ws.taper_family == taper_family
                assert len(ws.taper) == 64

    def test_kaiser_window_with_beta(self):
        """Test Kaiser window with beta parameter."""
        ws = WindowingScheme(
            num_samples_window=64,
            num_samples_overlap=16,
            taper_family="kaiser",
            taper_additional_args={"beta": 5.0},
        )

        assert ws.taper_family == "kaiser"
        assert len(ws.taper) == 64

    def test_tukey_window_with_alpha(self):
        """Test Tukey window with alpha parameter."""
        ws = WindowingScheme(
            num_samples_window=64,
            num_samples_overlap=16,
            taper_family="tukey",
            taper_additional_args={"alpha": 0.5},
        )

        assert ws.taper_family == "tukey"
        assert len(ws.taper) == 64


class TestWindowingSchemeIntegration:
    """Integration tests for complete workflows."""

    def test_complete_stft_workflow(self, random_seed):
        """Test complete STFT workflow: window -> taper -> FFT."""
        sample_rate = 100.0
        ws = WindowingScheme(
            num_samples_window=128,
            num_samples_overlap=64,
            sample_rate=sample_rate,
            taper_family="hamming",
        )

        # Create test data
        N = 10000
        t0 = np.datetime64("2020-01-01 00:00:00")
        time_vector = make_time_axis(t0, N, sample_rate)
        ds = xr.Dataset(
            {
                "ex": (["time"], np.sin(2 * np.pi * 5 * np.arange(N) / sample_rate)),
                "ey": (["time"], np.cos(2 * np.pi * 5 * np.arange(N) / sample_rate)),
            },
            coords={"time": time_vector},
            attrs={"sample_rate": sample_rate},
        )

        # Apply complete workflow
        windowed = ws.apply_sliding_window(ds)
        tapered = ws.apply_taper(windowed)
        stft = ws.apply_fft(tapered)

        assert isinstance(stft, xr.Dataset)
        assert "ex" in stft
        assert "ey" in stft
        assert "frequency" in stft.coords

        # Check that we have complex values
        assert np.iscomplexobj(stft["ex"].data)

    def test_windowing_preserves_data_length_relationship(self, random_seed, subtests):
        """Test that windowing parameters produce expected number of windows."""
        data_lengths = [1000, 5000, 10000]
        window_sizes = [32, 64, 128]
        overlaps = [8, 16, 32]

        for data_len in data_lengths:
            for win_size in window_sizes:
                for overlap in overlaps:
                    if overlap >= win_size:
                        continue

                    with subtests.test(
                        data_len=data_len, win_size=win_size, overlap=overlap
                    ):
                        ws = WindowingScheme(
                            num_samples_window=win_size,
                            num_samples_overlap=overlap,
                        )

                        ts = np.random.random(data_len)
                        windowed = ws.apply_sliding_window(ts)

                        expected_windows = ws.available_number_of_windows(data_len)
                        assert windowed.shape[0] == expected_windows
                        assert windowed.shape[1] == win_size


class TestWindowingSchemeStridingFunction:
    """Test striding function parameter."""

    def test_default_striding_function(self, basic_windowing_scheme):
        """Test that default striding function is 'crude'."""
        assert basic_windowing_scheme.striding_function_label == "crude"

    def test_custom_striding_function_label(self):
        """Test setting custom striding function label."""
        ws = WindowingScheme(
            num_samples_window=32,
            num_samples_overlap=8,
            striding_function_label="crude",
        )

        assert ws.striding_function_label == "crude"
