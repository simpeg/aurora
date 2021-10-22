"""
TODO: Add a test that uses detrending constant, none and linear for fft
or maybe make an fft test module?
"""
import numpy as np
import xarray as xr

from aurora.time_series.time_axis_helpers import make_time_axis
from aurora.time_series.windowing_scheme import WindowingScheme


def get_test_windowing_scheme(
    num_samples_window=32, num_samples_overlap=8, sample_rate=None
):
    windowing_scheme = WindowingScheme(
        num_samples_window=num_samples_window,
        num_samples_overlap=num_samples_overlap,
        taper_family="hamming",
        sample_rate=sample_rate,
    )
    return windowing_scheme


def get_xarray_dataset(N=1000, sps=50.0):
    t0 = np.datetime64("1977-03-02 12:34:56")
    time_vector = make_time_axis(t0, N, sps)
    print("ok, now make a few xarrays, then bind them into a dataset")
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
            "some random info": "dogs",
            "some more random info": "cats",
        },
    )
    return ds


# <TESTS>
def test_instantiate_windowing_scheme():
    ws = WindowingScheme(
        num_samples_window=128,
        num_samples_overlap=32,
        num_samples_data=1000,
        taper_family="hamming",
    )
    ws.sample_rate = 50.0
    # assert(ws.sample_rate==50)
    print(ws.window_duration)
    print(ws)
    print("assert some condtion here")
    # assert allows checking bool as well,
    # rather than just printing it will tell you if that particular statement
    # fails.
    #
    # good for manual checks of values
    # see: https://github.com/simpeg/discretize/blob/main/tests/base/test_basemesh.py
    return


def test_apply_sliding_window():
    N = 10000
    qq = np.random.random(N)
    windowing_scheme = WindowingScheme(num_samples_window=64, num_samples_overlap=50)
    print(windowing_scheme.num_samples_advance)
    print(windowing_scheme.available_number_of_windows(N))
    ww = windowing_scheme.apply_sliding_window(qq)
    return ww


def test_apply_sliding_window_can_return_xarray():
    qq = np.arange(15)
    windowing_scheme = WindowingScheme(num_samples_window=3, num_samples_overlap=1)
    windowed_xr = windowing_scheme.apply_sliding_window(qq, return_xarray=True)
    print(windowed_xr)
    return windowed_xr


def test_apply_sliding_window_to_xarray(return_xarray=False):
    N = 10000
    xrd = xr.DataArray(
        np.random.randn(N, 1), dims=["time", "channel"], coords={"time": np.arange(N)}
    )
    windowing_scheme = WindowingScheme(num_samples_window=64, num_samples_overlap=50)
    ww = windowing_scheme.apply_sliding_window(xrd, return_xarray=return_xarray)
    return ww


def test_can_apply_taper():
    N = 10000
    qq = np.random.random(N)
    windowing_scheme = WindowingScheme(
        num_samples_window=64, num_samples_overlap=50, taper_family="hamming"
    )
    print(windowing_scheme.num_samples_advance)
    print(windowing_scheme.available_number_of_windows(N))
    windowed_data = windowing_scheme.apply_sliding_window(qq)
    tapered_windowed_data = windowing_scheme.apply_taper(windowed_data)
    print(tapered_windowed_data)

    # import matplotlib.pyplot as plt
    # plt.plot(windowed_data[0],'r');plt.plot(tapered_windowed_data[0],'g')
    # plt.show()
    print("ok")
    return


def test_taper_dataset(plot=False):
    import matplotlib.pyplot as plt

    windowing_scheme = get_test_windowing_scheme(num_samples_window=64)
    ds = get_xarray_dataset()
    windowed_dataset = windowing_scheme.apply_sliding_window(ds, return_xarray=True)
    if plot:
        plt.plot(windowed_dataset["hx"].data[0, :], "r", label="window0")
        plt.plot(windowed_dataset["hx"].data[1, :], "r", label="window1")
    tapered_dataset = windowing_scheme.apply_taper(windowed_dataset)
    if plot:
        plt.plot(tapered_dataset["hx"].data[0, :], "g", label="tapered0")
        plt.plot(tapered_dataset["hx"].data[1, :], "g", label="tapered1")
        plt.legend()
        plt.show()
    print("OK")

    # windowed_dataset = test_can_create_xarray_dataset_from_several_sliding_window_xarrays()
    # tapered_windowed_dataset = win


def test_can_create_xarray_dataset_from_several_sliding_window_xarrays():
    """
    This method operates on an xarray dataset.
    Returns
    -------
    """
    windowing_scheme = get_test_windowing_scheme()
    ds = get_xarray_dataset()
    wds = windowing_scheme.apply_sliding_window(ds, return_xarray=True)
    print(wds)
    return wds


def test_fourier_transform():
    """
    This method needs to get a windowed time series, apply the taper,
    fft, scale the Fourier coefficients
    Returns
    -------

    """
    sample_rate = 40.0
    windowing_scheme = get_test_windowing_scheme(
        num_samples_window=128, num_samples_overlap=96, sample_rate=sample_rate
    )
    ds = get_xarray_dataset(N=10000, sps=sample_rate)
    windowed_dataset = windowing_scheme.apply_sliding_window(ds)
    tapered_windowed_dataset = windowing_scheme.apply_taper(windowed_dataset)
    stft = windowing_scheme.apply_fft(tapered_windowed_dataset)
    print(f"Type(stft) = {type(stft)}")
    # import matplotlib.pyplot as plt
    # plt.plot(stft.frequency.data, np.abs(stft["hx"].data.mean(axis=0)))
    # plt.show()
    print("ok")
    pass


# </TESTS>


def main():
    """
    Testing the windowing scheme
    """
    test_instantiate_windowing_scheme()
    windowed_array = test_apply_sliding_window()
    print(f"windowed_array shape = {windowed_array.shape}")
    windowed_xarray = test_apply_sliding_window_can_return_xarray()
    print(f"Type(windowed_xarray) = {type(windowed_xarray)}")
    windowed_xarray = test_apply_sliding_window_to_xarray(return_xarray=False)
    print(f"Type(windowed_xarray) = {type(windowed_xarray)}")
    windowed_xarray = test_apply_sliding_window_to_xarray(return_xarray=True)
    print(f"Type(windowed_xarray) = {type(windowed_xarray)}")
    xr_ds = test_can_create_xarray_dataset_from_several_sliding_window_xarrays()
    print(f"Type(xr_ds) = {type(xr_ds)}")
    test_can_apply_taper()
    test_taper_dataset(plot=False)
    test_fourier_transform()

    print(
        "@ToDo Insert an integrated test showing common usage of sliding window\
    for 2D arrays, for example windowing for dnff"
    )

    print("finito")


if __name__ == "__main__":
    main()
