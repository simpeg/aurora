import numpy as np
import xarray as xr
import unittest

from aurora.time_series.time_axis_helpers import make_time_axis
from aurora.time_series.windowing_scheme import WindowingScheme
from loguru import logger

np.random.seed(0)


# =============================================================================
#  Helper functions
# =============================================================================


def get_windowing_scheme(
    num_samples_window=32,
    num_samples_overlap=8,
    sample_rate=None,
    taper_family="hamming",
):
    windowing_scheme = WindowingScheme(
        num_samples_window=num_samples_window,
        num_samples_overlap=num_samples_overlap,
        taper_family=taper_family,
        sample_rate=sample_rate,
    )
    return windowing_scheme


def get_xarray_dataset(N=1000, sps=50.0):
    """
    make a few xarrays, then bind them into a dataset
    ToDo: Consider moving this method into test_utils/

    """
    t0 = np.datetime64("1977-03-02 12:34:56")
    time_vector = make_time_axis(t0, N, sps)
    ds = xr.Dataset(
        {
            "hx": (
                [
                    "time",
                ],
                np.abs(np.random.randn(N)),
            ),
            "hy": (
                [
                    "time",
                ],
                np.abs(np.random.randn(N)),
            ),
        },
        coords={
            "time": time_vector,
        },
        attrs={
            "some random info": "dogs",
            "some more random info": "cats",
            "sample_rate": sps,
        },
    )
    return ds


# =============================================================================
#  Tests
# =============================================================================


class TestWindowingScheme(unittest.TestCase):
    def setUp(self):
        self.defaut_num_samples_data = 10000
        self.defaut_num_samples_window = 64
        self.default_num_samples_overlap = 50

    def test_cant_write_xarray_attrs(self):
        """
        This could go into a separate module for testing xarray stuff
        """
        ds = get_xarray_dataset()
        try:
            ds.sample_rate = 10
            logger.info("was not expecting to be able to overwrite attr of xarray")
            assert False
        except AttributeError:
            assert True

    def test_instantiate_windowing_scheme(self):
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

    def test_apply_sliding_window(self):
        num_samples_data = self.defaut_num_samples_data
        num_samples_window = self.defaut_num_samples_window
        num_samples_overlap = self.default_num_samples_overlap
        ts = np.random.random(num_samples_data)
        windowing_scheme = WindowingScheme(
            num_samples_window=num_samples_window,
            num_samples_overlap=num_samples_overlap,
        )
        windowed_array = windowing_scheme.apply_sliding_window(ts)
        return windowed_array

    def test_apply_sliding_window_can_return_xarray(self):
        ts = np.arange(15)
        windowing_scheme = WindowingScheme(num_samples_window=3, num_samples_overlap=1)
        windowed_xr = windowing_scheme.apply_sliding_window(ts, return_xarray=True)
        assert isinstance(windowed_xr, xr.DataArray)
        return windowed_xr

    def test_apply_sliding_window_to_xarray(self, return_xarray=False):
        num_samples_data = self.defaut_num_samples_data
        num_samples_window = self.defaut_num_samples_window
        num_samples_overlap = self.default_num_samples_overlap
        xrd = xr.DataArray(
            np.random.randn(num_samples_data, 1),
            dims=["time", "channel"],
            coords={"time": np.arange(num_samples_data)},
        )
        windowing_scheme = WindowingScheme(
            num_samples_window=num_samples_window,
            num_samples_overlap=num_samples_overlap,
        )
        windowed_xrda = windowing_scheme.apply_sliding_window(
            xrd, return_xarray=return_xarray
        )
        return windowed_xrda

    def test_can_apply_taper(self):
        from aurora.time_series.window_helpers import (
            available_number_of_windows_in_array,
        )

        num_samples_data = self.defaut_num_samples_data
        num_samples_window = self.defaut_num_samples_window
        num_samples_overlap = self.default_num_samples_overlap
        ts = np.random.random(num_samples_data)
        windowing_scheme = WindowingScheme(
            num_samples_window=num_samples_window,
            num_samples_overlap=num_samples_overlap,
            taper_family="hamming",
        )
        expected_advance = num_samples_window - num_samples_overlap
        assert windowing_scheme.num_samples_advance == expected_advance
        expected_num_windows = available_number_of_windows_in_array(
            num_samples_data, num_samples_window, expected_advance
        )
        num_windows = windowing_scheme.available_number_of_windows(num_samples_data)
        assert num_windows == expected_num_windows
        windowed_data = windowing_scheme.apply_sliding_window(ts)
        tapered_windowed_data = windowing_scheme.apply_taper(windowed_data)
        assert (windowed_data[:, 0] != tapered_windowed_data[:, 0]).all()

        # import matplotlib.pyplot as plt
        # plt.plot(windowed_data[0],'r');plt.plot(tapered_windowed_data[0],'g')
        # plt.show()
        return

    def test_taper_dataset(self, plot=False):
        import matplotlib.pyplot as plt

        windowing_scheme = get_windowing_scheme(
            num_samples_window=64,
            num_samples_overlap=8,
            sample_rate=None,
            taper_family="hamming",
        )
        ds = get_xarray_dataset()

        windowed_dataset = windowing_scheme.apply_sliding_window(ds, return_xarray=True)
        if plot:
            fig, ax = plt.subplots()
            ax.plot(windowed_dataset["hx"].data[0, :], "r", label="window0")
            ax.plot(windowed_dataset["hx"].data[1, :], "r", label="window1")
        tapered_dataset = windowing_scheme.apply_taper(windowed_dataset)
        if plot:
            ax.plot(tapered_dataset["hx"].data[0, :], "g", label="tapered0")
            ax.plot(tapered_dataset["hx"].data[1, :], "g", label="tapered1")
            ax.legend()
            plt.show()

    def test_can_create_xarray_dataset_from_several_sliding_window_xarrays(self):
        """
        This method operates on an xarray dataset.
        Returns
        -------
        """
        windowing_scheme = get_windowing_scheme(
            num_samples_window=32, num_samples_overlap=8
        )
        ds = get_xarray_dataset()
        wds = windowing_scheme.apply_sliding_window(ds, return_xarray=True)
        return wds

    def test_fourier_transform(self):
        """
        This method gets a windowed time series, applies a taper, and fft
        """
        sample_rate = 40.0
        windowing_scheme = get_windowing_scheme(
            num_samples_window=128, num_samples_overlap=96, sample_rate=sample_rate
        )

        # Test with xr.Dataset
        ds = get_xarray_dataset(N=10000, sps=sample_rate)
        windowed_dataset = windowing_scheme.apply_sliding_window(ds)
        tapered_windowed_dataset = windowing_scheme.apply_taper(windowed_dataset)
        stft = windowing_scheme.apply_fft(tapered_windowed_dataset)
        assert isinstance(stft, xr.Dataset)

        # Test with xr.DataArray
        da = ds.to_array("channel")
        windowed_dataset = windowing_scheme.apply_sliding_window(da)
        tapered_windowed_dataset = windowing_scheme.apply_taper(windowed_dataset)
        stft = windowing_scheme.apply_fft(tapered_windowed_dataset)
        assert isinstance(stft, xr.DataArray)

        # import matplotlib.pyplot as plt
        # plt.plot(stft.frequency.data, np.abs(stft["hx"].data.mean(axis=0)))
        # plt.show()


def main():
    """
    Testing the windowing scheme
    """
    unittest.main()


if __name__ == "__main__":
    main()
