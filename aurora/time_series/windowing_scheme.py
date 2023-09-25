#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
The windowing scheme defines the chunking and chopping of the time series for
the Short Time Fourier Transform.  Often referred to as a "sliding window" or
a "striding window".  Iin its most basic form it is a taper with a rule to
say how far to advance at each stride (or step).

To generate an array of data-windows from a data series we only need the
two parameters window_length (L) and window_overlap (V).  The parameter
"window_advance" (L-V) can be used in lieu of overlap.  Sliding windows are
normally described terms of overlap but it is cleaner to code in terms of
advance.

Choices L and V are usually made with some knowledge of time series sample
rate, duration, and the frequency band of interest.  In aurora because this is used
to prep for STFT, L is typically a power of 2.

In general we will need one instance of this class per decimation level,
but in practice often leave the windowing scheme the same for each decimation level.

This class is a key part of the "gateway" to frequency domain, so it has been given
a sampling_rate attribute.  While sampling rate is a property of the data, and not
the windowing scheme per se, it is good for this class to be aware of the sampling
rate.

Future modifications could involve:
- binding this class with a time series.
- Making a subclass with only L, V, and then having an extension with sample_rate


When 2D arrays are generated how should we index them?
[[ 0  1  2]
 [ 2  3  4]
 [ 4  5  6]
 [ 6  7  8]
 [ 8  9 10]
 [10 11 12]
 [12 13 14]]
In this example the rows are indexing the individual windows ... and so they
should be associated with the time of each window.  We will need to set a
standard for this.  Obvious options are center_time of window and time_of_first
sample. I prefer time_of_first sample.  This can always be transformed to
center time or another standard later.  We can call this the "window time
axis".  The columns are indexing "steps of delta-t".  The actual times are
different for every row, so it would be best to use something like
[0, dt, 2*dt] for that axis to keep it general.  We can call this the
"within-window sample time axis"


TODO: Regarding the optional time_vector input to self.apply_sliding_window()
... this current implementation takes as input numpy array data.  We need to
also allow for an xarray to be implemented. In the simplest case we would
take an xarray in and extract its "time" axis as time vector

20210529
This class is going to be modified to only accept xarray as input data.
We can force any incoming numpy arrays to be either xr.DataArray or xr.Dataset.
Similarly, output will be only xr.DataArray or xr.Dataset
"""

import copy
import numpy as np
import xarray as xr

from aurora.time_series.apodization_window import ApodizationWindow
from aurora.time_series.windowed_time_series import WindowedTimeSeries
from aurora.time_series.window_helpers import available_number_of_windows_in_array
from aurora.time_series.window_helpers import SLIDING_WINDOW_FUNCTIONS

from mt_metadata.transfer_functions.processing.aurora.decimation_level import (
    get_fft_harmonics,
)


class WindowingScheme(ApodizationWindow):
    """
    20210415: Casting window length, overlap, advance, etc. in terms of number
    of samples or "points" here as this is common signal processing the
    nomenclature.  We may provide an interface to define these things in terms
    of percent, duration in seconds etc. in a supporting module.

    Note that sample_rate is actually a property of the data and not of the
    window ... still not sure if we want to make sample_rate an attr here
    or if its better to put properties like window_duration() as a method of
    some composition of time series and windowing scheme.
    """

    def __init__(self, **kwargs):
        super(WindowingScheme, self).__init__(**kwargs)
        self.num_samples_overlap = kwargs.get(
            "num_samples_overlap", None
        )  # make this 75% of num_samples_window by default
        self.striding_function_label = kwargs.get("striding_function_label", "crude")
        self._left_hand_window_edge_indices = None
        self.sample_rate = kwargs.get("sample_rate", None)

    def clone(cls):
        return copy.deepcopy(cls)

    def __str__(self):
        info_string = (
            f"Window of {self.num_samples_window} samples with "
            f"overlap {self.num_samples_overlap}"
        )
        # add taper summary here?
        return info_string

    @property
    def num_samples_advance(self):
        """
        A derived property.  If we made this a fundamental defined property
        then overlap would become a derived property.  Overlap is more
        conventional than advance in the literature however so we choose it as
        our property label.
        """
        return self.num_samples_window - self.num_samples_overlap

    def available_number_of_windows(self, num_samples_data):
        """

        Parameters
        ----------
        num_samples_data : int
            The number of samples in the time series to be windowed by self.

        Returns
        -------
        number_of_windows : int
           Count of the number of windows returned from time series of
           num_samples_data.  Only take as many windows as available without
           wrapping.  Start with one window for free, move forward by
           num_samples_advance and don't walk over the cliff.
        """
        return available_number_of_windows_in_array(
            num_samples_data, self.num_samples_window, self.num_samples_advance
        )

    def apply_sliding_window(
        self, data, time_vector=None, dt=None, return_xarray=False
    ):
        """
        I would like this method to support numpy arrays as well as xarrays.

        Parameters
        ----------
        data: 1D numpy array, xr.DataArray, xr.Dataset
            The data to break into ensembles.
        time_vector: 1D numpy array
            The time axis of the data.
        dt: float
            The sample interval of the data (reciprocal of sample_rate)
        return_xarray: boolean
            If True will return an xarray object, even if the input object was a
            numpy array

        Returns
        -------
        windowed_obj: arraylike
            Normally an object of type xarray.core.dataarray.DataArray
            Could be numpy array as well.
        """
        if isinstance(data, np.ndarray):
            windowed_obj = self._apply_sliding_window_numpy(
                data, time_vector=time_vector, dt=dt, return_xarray=return_xarray
            )

        elif isinstance(data, xr.DataArray):
            # Cast DataArray to DataSet, iterate and then Dataset back to DataArray
            xrds = data.to_dataset("channel")
            windowed_obj = self.apply_sliding_window(
                xrds, time_vector=time_vector, dt=dt
            )
            windowed_obj = windowed_obj.to_array("channel")

        elif isinstance(data, xr.Dataset):
            ds = xr.Dataset()
            for key in data.keys():
                windowed_obj = self._apply_sliding_window_numpy(
                    data[key].data,
                    time_vector=data.time.data,
                    dt=dt,
                    return_xarray=True,
                )
                ds.update({key: windowed_obj})
            windowed_obj = ds

        else:
            print(f"Unexpected Data type {type(data)}")
            raise Exception
        return windowed_obj

    def _apply_sliding_window_numpy(
        self, data, time_vector=None, dt=None, return_xarray=False
    ):
        """

        Parameters
        ----------
        data: numpy.ndarray
            A channel of time series data
        time_vector: numpy.ndarray or None
            Time coordinate of xarray.  If None is passed we just assign integer counts
        dt: float or None
            Sampling interval
        return_xarray: bool
            If True an xarray is returned,
            If False we just return a numpy array of the windowed data


        Returns
        -------
        output: xr.DataArray or np.ndarray
            The windowed time series, bound to time axis or just as numpy array,
            depending on the value of return_xarray
        """
        sliding_window_function = SLIDING_WINDOW_FUNCTIONS[self.striding_function_label]
        windowed_array = sliding_window_function(
            data, self.num_samples_window, self.num_samples_advance
        )

        if return_xarray:
            # Get window_time_axis coordinate
            if time_vector is None:
                time_vector = np.arange(len(data))
            window_time_axis = self.downsample_time_axis(time_vector)

            output = self.cast_windowed_data_to_xarray(
                windowed_array, window_time_axis, dt=dt
            )
        else:
            output = windowed_array

        return output

    def cast_windowed_data_to_xarray(self, windowed_array, time_vector, dt=None):
        """
        TODO?: Factor this method to a standalone function in window_helpers?

        Parameters
        ----------
        windowed_array
        time_vector
        dt

        Returns
        -------

        """
        # Get within-window_time_axis coordinate
        if dt is None:
            print("Warning dt not defined, using dt=1")
            dt = 1.0
        within_window_time_axis = dt * np.arange(self.num_samples_window)

        # cast to xr.DataArray
        xrd = xr.DataArray(
            windowed_array,
            dims=["time", "within-window time"],
            coords={"within-window time": within_window_time_axis, "time": time_vector},
        )
        return xrd

    def compute_window_edge_indices(self, num_samples_data):
        """This has been useful in the past but maybe not needed here"""
        number_of_windows = self.available_number_of_windows(num_samples_data)
        self._left_hand_window_edge_indices = (
            np.arange(number_of_windows) * self.num_samples_advance
        )
        return

    def left_hand_window_edge_indices(self, num_samples_data):
        if self._left_hand_window_edge_indices is None:
            self.compute_window_edge_indices(num_samples_data)
        return self._left_hand_window_edge_indices

    def downsample_time_axis(self, time_axis):
        """
        Parameters
        ----------
        time_axis : arraylike
            This is the time axis associated with the time-series prior to
            the windowing operation.

        Returns
        -------
        window_time_axis : array-like
            This is a time axis for the windowed data.  Say that we had 1Hz
            data starting at t=0 and 100 samples.  Then we window,
            with window length 10, and advance 10, the window time axis is
             [0, 10, 20 , ... 90].  Say the same window length, but now
             advance is 5.  Then [0, 5, 10, 15, ... 90] is the result.


        """
        lhwe = self.left_hand_window_edge_indices(len(time_axis))
        window_time_axis = time_axis[lhwe]
        return window_time_axis

    def apply_taper(self, data):
        """
        modifies the data in place by applying a taper to each window
        TODO: consider adding an option to return a copy of the data without
        the taper applied
        """
        data = WindowedTimeSeries.apply_taper(data=data, taper=self.taper)
        return data

    def frequency_axis(self, dt):
        fft_harmonics = get_fft_harmonics(self.num_samples_window, 1.0 / dt)
        return fft_harmonics

    def apply_fft(self, data, spectral_density_correction=True, detrend_type="linear"):
        """

        Parameters
        ----------
        data: xarray.core.dataset.Dataset
        spectral_density_correction: boolean
        detrend_type: string

        Returns
        -------
        spectral_ds:


        Assume we have already applied sliding window and taper.
        Things to think about:
        We want to assign the frequency axis during this method

        """
        # ONLY SUPPORTS DATASET AT THIS POINT
        if isinstance(data, xr.Dataset):
            spectral_ds = fft_xr_ds(data, self.sample_rate, detrend_type=detrend_type)
            if spectral_density_correction:
                spectral_ds = self.apply_spectral_density_calibration(spectral_ds)
        elif isinstance(data, xr.DataArray):
            xrds = data.to_dataset("channel")
            spectral_ds = fft_xr_ds(xrds, self.sample_rate, detrend_type=detrend_type)
            spectral_ds = spectral_ds.to_array("channel")
            return spectral_ds

        else:
            print(f"fft of {type(data)} not yet supported")
            raise Exception

        return spectral_ds

    def apply_spectral_density_calibration(self, dataset):
        """
        Parameters
        ----------
        dataset

        Returns
        -------


        """
        scale_factor = self.linear_spectral_density_calibration_factor
        dataset *= scale_factor
        return dataset

    # PROPERTIES THAT NEED SAMPLING RATE
    # these may be moved elsewhere later
    @property
    def dt(self):
        """
        comes from data
        """
        return 1.0 / self.sample_rate

    @property
    def window_duration(self):
        """
        units are SI seconds assuming dt is SI seconds
        """
        return self.num_samples_window * self.dt

    @property
    def duration_advance(self):
        """ """
        return self.num_samples_advance * self.dt

    @property
    def linear_spectral_density_calibration_factor(self):
        """
        Returns
        -------
        calibration_factor : float
            Following Hienzel et al 2002, Equations 24 and 25 for Linear
            Spectral Density correction for a single sided spectrum.
        """
        return np.sqrt(2 / (self.sample_rate * self.S2))


def fft_xr_ds(dataset, sample_rate, detrend_type=None, prewhitening=None):
    """

    This should call window_helpers.apply_fft_to_windowed_array
    or get moved to window_helpers.py

    The returned harmonics do not include the Nyquist frequency. To modify this
    add +1 to n_fft_harmonics.  Also, only 1-sided ffts are returned.

    For each channel within the Dataset, fft is applied along the
    within-window-time axis of the associated numpy array

    Parameters
    ----------
    dataset : xr.Dataset
        Data are 2D (windowed univariate time series).
    sample_rate: float

    detrend_type
    prewhitening

    Returns
    -------

    """
    # TODO: Modify this so that demeaning and detrending is happening before
    # application of the tapering window.  Add a second demean right before the FFT

    samples_per_window = len(dataset.coords["within-window time"])
    n_fft_harmonics = int(samples_per_window / 2)  # no bin at Nyquist,
    harmonic_frequencies = get_fft_harmonics(samples_per_window, sample_rate)

    # CORE METHOD
    output_ds = xr.Dataset()
    # operation_axis = 1  # make this pick the "time" axis from xarray
    time_coordinate_index = list(dataset.coords.keys()).index("time")
    if detrend_type:  # neither False nor None
        dataset = WindowedTimeSeries.detrend(
            data=dataset, detrend_axis=time_coordinate_index, detrend_type="linear"
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


def window_scheme_from_decimation(decimation):
    """
    Helper function to workaround mt_metadata to not import form aurora

    Parameters
    ----------
    decimation: mt_metadata.transfer_function.processing.aurora.decimation_level
    .DecimationLevel

    Returns
    -------
        windowing_scheme aurora.time_series.windowing_scheme.WindowingScheme
    """
    from aurora.time_series.windowing_scheme import WindowingScheme

    windowing_scheme = WindowingScheme(
        taper_family=decimation.window.type,
        num_samples_window=decimation.window.num_samples,
        num_samples_overlap=decimation.window.overlap,
        taper_additional_args=decimation.window.additional_args,
        sample_rate=decimation.sample_rate_decimation,
    )
    return windowing_scheme
