"""

    This module contains methods associated with operating on windowed time series.
    i.e. Arrays that have been chunked so that an operator can operate chunk-wise.

    Development Notes:
     Many of the methods in this module are not currently in use.
     It looks like I once thought we should have a class to handle windowed time series, but then
     decided that static methods applied to xr.Dataset would be better.


"""
from aurora.time_series.decorators import can_use_xr_dataarray
from mt_metadata.transfer_functions.processing.aurora.decimation_level import (
    get_fft_harmonics,
)
from typing import Optional, Union
from loguru import logger
import numpy as np
import scipy.signal as ssig
import xarray as xr


class WindowedTimeSeries(object):
    """
    Time series that has been chopped into (possibly) overlapping windows.

    This is a place where we can put methods that operate on these sorts of
    objects.

    Assumes xr.Datasets keyed by "channel"

    Specific methods:
        Demean
        Detrend
        Prewhiten
        stft
        invert_prewhitening

        TODO: Consider making these @staticmethod so import WindowedTimeSeries
         and then call the static methods
    """

    def __init__(self):
        """Constructor"""
        pass

    @can_use_xr_dataarray
    @staticmethod
    def apply_taper(
        data: Union[xr.DataArray, xr.Dataset], taper: np.ndarray
    ):  # , in_place=True):
        """
        Point by point multiplication of taper against time series.

        xarray handles this very cleanly as a direct multiply operation.
        tapered_obj = windowed_obj * windowing_scheme.taper
        """
        data = data * taper
        return data

    @staticmethod
    def detrend(
        data: xr.Dataset,
        detrend_axis: Optional[Union[int, None]] = None,
        detrend_type: Optional[str] = "linear",
    ):
        """
        De-trends input data.

        Development Notes:
         It looks like this function was concerned with nan in the time series.
         Since there is no general implemented solution for this in aurora/MTH5
         it maybe better to just get rid of the nan-checking. The data can be
         pre-screened for nans if needed.


        Parameters
        ----------
        data : xarray Dataset
        detrend_axis : int
        detrend_type : string
            Controlled vocabulary ["linear", "constant"]
            This argument is provided to scipy.signal.detrend

        Returns
        -------
        data:xr.Dataset
            The input data, modified in-place with de-trending.
        """
        if detrend_axis is None:
            detrend_axis = _get_time_coordinate_axis(data)

        for channel in data.keys():
            ensembles = data[channel].data
            if np.isnan(ensembles).any():
                msg = "detrending data with nan not supported"
                logger.error(msg)
                # TODO: use nan_to_mean from time_series helpers

            try:
                ensembles = ssig.detrend(
                    ensembles, axis=detrend_axis, type=detrend_type
                )
            except ValueError as e:
                # TODO: Add doc -- this looks like it may be handling invalid detrend types
                msg = (
                    "Could not detrend "
                    f"{channel} in time range "
                    f"{data[channel].coords.indexes['time'][0].isoformat()} to "
                    f"{data[channel].coords.indexes['time'][-1].isoformat()}."
                )
                logger.error(f"{msg} with error {e}")

            data[channel].data = ensembles
        return data

    @can_use_xr_dataarray
    @staticmethod
    def apply_fft(
        data: Union[xr.DataArray, xr.Dataset],
        sample_rate: float,
        spectral_density_correction: float,
        detrend_type: Optional[str] = "linear",
    ) -> xr.Dataset:
        """
        Applies the Fourier transform to each window in the windowed time series.

        Assumes sliding window and taper already applied.

        TODO: Make this return a Specrtogram() object.

        Parameters
        ----------
        data: xarray.core.dataset.Dataset
            The windowed data to FFT
        spectral_density_correction: float
            Applies window normalization (see Heinzel et al.)

        detrend_type: string
            Passed through to scipy.signal during detrend operation.

        Returns
        -------
        spectral_ds:xr.Dataset
            Dataset same channels as input but data are now complex values Fourier coefficients.

        """
        spectral_ds = fft_xr_ds(data, sample_rate, detrend_type=detrend_type)
        spectral_ds *= spectral_density_correction

        return spectral_ds


def _get_time_coordinate_axis(dataset: xr.Dataset) -> int:
    """
    Returns an integer corresponding to the time axis of the underlying numpy array

    It is common to pass an argument to scipy.signal methods axis=int
    where that integer specifies along which dimension we are applying the
    operator.  This method helps ensure the correct axis is passed..
    Parameters
    ----------
    dataset : xarray.Dataset

    Returns
    -------
    int:
        index of time axis in dataset
    """
    coordinate_labels = list(dataset.coords.keys())

    if len(coordinate_labels) != 2:
        logger.warning("Warning - Expected two distinct coordinates")
        # raise Exception

    return coordinate_labels.index("time")


def fft_xr_ds(
    dataset: xr.Dataset,
    sample_rate: float,
    detrend_type: Optional[str] = "linear",
    # prewhitening: Optional[Union[str, None]] = None,
) -> xr.Dataset:
    """
    Apply Fourier transform to an xarray Dataset (already windowed) time series.

    Notes:
    - The returned harmonics do not include the Nyquist frequency. To modify this
    add +1 to n_fft_harmonics.  Also, only 1-sided ffts are returned.
    - For each channel within the Dataset, fft is applied along the
    within-window-time axis of the associated numpy array

    TODO: add support for prewhitening per-window

    Parameters
    ----------
    dataset : xr.Dataset
        Data are 2D (windowed univariate time series).
    sample_rate: float
        The sample rate of the time series
    detrend_type: str
        Pass-through parameter to scipy.signal.detrend
    prewhitening: Not used (yet)

    Returns
    -------
    output_ds: xr.Dataset
        The FFT of the input.  Only contains non-negative frequencies
        i.e. input dataset had coords: (time, within-window time)
        but output ds has coords (time, frequency)
    """
    # TODO: Modify this so that demeaning and detrending is happening before
    #  application of the tapering window.  Add a second demean right before the FFT

    samples_per_window = len(dataset.coords["within-window time"])
    n_fft_harmonics = int(samples_per_window / 2)  # no bin at Nyquist,
    harmonic_frequencies = get_fft_harmonics(samples_per_window, sample_rate)

    output_ds = xr.Dataset()
    time_coordinate_index = list(dataset.coords.keys()).index("time")
    dataset = WindowedTimeSeries.detrend(
        data=dataset, detrend_axis=time_coordinate_index, detrend_type=detrend_type
    )
    for channel_id in dataset.keys():
        data = dataset[channel_id].data
        # Here is where you would add segment-by-segment prewhitening
        fspec_array = np.fft.fft(data, axis=time_coordinate_index)
        fspec_array = fspec_array[:, 0:n_fft_harmonics]  # 1-sided

        xrd = xr.DataArray(
            fspec_array,
            dims=["time", "frequency"],
            coords={"frequency": harmonic_frequencies, "time": dataset.time.data},
        )
        output_ds.update({channel_id: xrd})

    return output_ds


# Not used 20240723: Commenting out.
# def delay_correction(self, dataset, run_obj):
#     """
#     Applies a time delay correction -- NOT TESTED - PSEUDOCODE ONLY
#
#     Parameters
#     ----------
#     dataset : xr.Dataset
#     run_obj :
#
#     Returns
#     -------
#
#     """
#     from scipy.interpolate import interp1d
#     for channel_id in dataset.keys():
#         mth5_channel = run_obj.get_channel(channel_id)
#         channel_filter = mth5_channel.channel_response
#         delay_in_seconds = channel_filter.total_delay
#         true_time_axis = dataset.time + delay_in_seconds
#         interpolator = interp1d(
#             true_time_axis, dataset[channel_id].data, assume_sorted=True
#         )
#         corrected_data = interpolator(dataset.time)
#         dataset[channel_id].data = corrected_data
#     return dataset

# Not used 20240723: Commenting out.
# def validate_coordinate_ordering_time_domain(dataset:xr.Dataset):
#     """
#     Check that the data dimensions are what you expect.
#
#     Not Currently Used.
#
#     Development Notes:
#     Want to make sure operating along the correct axes (demean, detrend, taper, etc.)
#
#     TODO: add this to WindowedTimeSeries as a validator.
#      But in this case make sure that the axis being operated along is the right one based
#      on its coordinate name ("within-window time" for most operations).
#
#     Parameters
#     ----------
#     dataset : xarray.Dataset
#         This is a windowed dataset, so it is expected to have coords:
#         ["time", "within-window time"]
#
#     Returns
#     -------
#     bool: True if valid, else False
#     """
#     coordinate_labels = list(dataset.coords.keys())
#     cond1 = coordinate_labels[0] == "within-window time"
#     cond2 = coordinate_labels[1] == "time"
#     if cond1 & cond2:
#         return True
#     else:
#         logger.error("Uncertain that xarray coordinates are correctly ordered")
#         return False
#         # raise ValueError
