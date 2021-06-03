#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
The windowing scheme defines the chunking and chopping of the time series for
the Short Time Fourier Transform.  Often referred to as a "sliding window" or
a "striding window".  It is basically a taper with a rule to say how far to
advance at each stride (or step).

To generate an array of data-windows from a data series we only need the
two parameters window_length (L) and window_overlap (V).  The parameter
"window_advance" (L-V) can be used in lieu of overlap.  Sliding windows are
normally described terms of overlap but it is cleaner to code in terms of
advance.

Choices L and V are usually made with some knowledge of time series sample
rate, duration, and the frequency band of interest.  We can create a
module that "suggests" L, V, based on these metadata to make the default
processing configuration parameters.

Note: In general we will need one instance of this class per decimation level,
but in the current implementation we will probably leave the windowing scheme
the same for each decimation level.

This class is a key part of the "gateway" to frequency domain, so what
frequency domain considerations do we want to think about here.. certainly
the window length and the sampling rate define the frequency resolution, and as
such should be considered in context of the "band averaging scheme"

Indeed the frequencies come from this class if it has a sampling rate.  While
sampling rate is a property of the data, and not the windowing scheme per se,
it is good for this class to be aware of the sampling rate.  ... or should we
push the frequency stuffs to a combination of TS plus WindowingScheme?
The latter feels more appropriate.

<20210510>
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

</20210510>

There is an open trade here about whether to embed the data length as an ivar
or a variable we pass. i.e. whether the same windowing scheme is independent
of the data length or not.

TODO: Regarding the optional time_vector input to self.apply_sliding_window()
... this current implementation takes as input numpy array data.  We need to
also allow for an xarray to be implemented. In the simplest case we would
take an xarray in and extract its "time" axis as time vector

<20210529>
This class is going to be modified to only accept xarray as input data.
We can force any incoming numpy arrays to be either xr.DataArray or xr.Dataset.
Similarly, output will be only xr.DataArray or xr.Dataset
</20210529>
"""

import copy
import numpy as np
import xarray as xr

from aurora.signal.apodization_window import ApodizationWindow

from aurora.signal.window_helpers import apply_taper_to_windowed_array
from aurora.signal.window_helpers import available_number_of_windows_in_array
from aurora.signal.window_helpers import SLIDING_WINDOW_FUNCTIONS

def cast_numpy_array_to_xarray_dataset():
    """

    Returns
    -------

    """
    pass

class WindowingScheme(ApodizationWindow):
    """
    20210415: Casting everything in terms of number of samples or "points" as
    this is the nomenclature of the signal processing cadre.  We can provide
    functions to define things like overlap in terms of percent, duration in
    seconds etc in another module.

    Note that sampling_rate is actually a property of the data and not of the
    window ... still not sure if we want to make sampling_rate an attr here
    or if its better to put properties like window_duration() as a method of
    some composition of time series and windowing scheme.

    kwargs:

    """
    def __init__(self, **kwargs):
        super(WindowingScheme, self).__init__(**kwargs)
        self.num_samples_overlap = kwargs.get("num_samples_overlap", None) #make this 75% of num_samples_window by default
        self.striding_function_label = kwargs.get("striding_function_label", "crude")
        self._left_hand_window_edge_indices = None
        self.sampling_rate = kwargs.get("sampling_rate", None)

    def clone(cls):
        return copy.deepcopy(cls)

    def __str__(self):
        info_string = f"Window of {self.num_samples_window} samples with " \
            f"overlap {self.num_samples_overlap}"
        #add taper summary here?
        return info_string

    @property
    def num_samples_advance(self):
        """
        Attributes derived property that actually could be fundamental .. if we
        defined this we would wind up deriving the overlap.  Overlap is more
        conventional than advance in the literature however so we choose this as
        our property label.
        """
        return self.num_samples_window - self.num_samples_overlap


    def available_number_of_windows(self, num_samples_data):
        """
        dont walk over the cliff.  Only take as many windows as available
        without wrapping.  Start with one window for free.

        Parameters
        ----------
        num_samples_data

        Returns
        -------

        """
        return available_number_of_windows_in_array(num_samples_data,
                                                    self.num_samples_window,
                                                    self.num_samples_advance)


    def apply_sliding_window(self, data, time_vector=None, dt=None,
                             return_xarray=False):
        """
        I would like this method to support numpy arrays as well as xarrays.
        Parameters
        ----------
        data
        time_vector
        dt
        return_xarray

        Returns windowed_obj
        -------

        """
        if isinstance(data, np.ndarray):
            print("this will only work for a 1D array, cast to dataset for "
                  "generality")
            windowed_obj = self._apply_sliding_window_numpy(data,
                                                            time_vector=time_vector,
                                                            dt=dt,
                                                            return_xarray=return_xarray)

        elif isinstance(data, xr.DataArray):
            """
            Cast DataArray to DataSet, iterate and then Dataset back to 
            DataArray
            
            """
            xrds = data.to_dataset("channel")
            windowed_obj = self.apply_sliding_window(xrds,
                                                   time_vector=time_vector, dt=dt)
            windowed_obj = windowed_obj.to_array("channel")

        elif isinstance(data, xr.Dataset):
            #return_xarray = True #only going to handle return xarray=T
            ds = xr.Dataset()
            for key in data.keys():
                print(f"key {key}")
                windowed_obj = self._apply_sliding_window_numpy(data[key].data,
                                                                time_vector=data.time.data,
                                                                dt=dt,
                                                                return_xarray=True)
                ds.update({key:windowed_obj})
            windowed_obj = ds

        else:
            print(f"Unexpected Data type {type(data)}")
            raise Exception
        return windowed_obj



    def _apply_sliding_window_numpy(self, data, time_vector=None, dt=None,
                             return_xarray=False):
        """

        Parameters
        ----------
        data
        time_vector: standin for the time coordinate of xarray.
        dt: stand in for sampling interval

        Returns
        -------

        """
        sliding_window_function = SLIDING_WINDOW_FUNCTIONS[self.striding_function_label]
        windowed_array = sliding_window_function(data, self.num_samples_window,
                                                 self.num_samples_advance)

        #<FACTOR TO ANOTHER METHOD>
        if return_xarray:
            #<Get window_time_axis coordinate>
            if time_vector is None:
                time_vector = np.arange(len(data))
            window_time_axis = self.downsample_time_axis(time_vector)
            #</Get window_time_axis coordinate>

            xrd = self.cast_windowed_data_to_xarray(windowed_array, window_time_axis, dt=dt)
            return xrd
        #</FACTOR TO ANOTHER METHOD>
        else:
            return windowed_array


    def cast_windowed_data_to_xarray(self, windowed_array, time_vector, dt=None):
        """
        TODO: FACTOR guts of this method out of class and place in window_helpers...
        Parameters
        ----------
        windowed_array
        time_vector
        dt

        Returns
        -------

        """
        # <Get within-window_time_axis coordinate>
        if dt is None:
            print("Warning dt not defined, using dt=1")
            dt = 1.0
        within_window_time_axis = dt * np.arange(self.num_samples_window)
        # <Get within-window_time_axis coordinate>

        # <Cast to xarray>
        xrd = xr.DataArray(windowed_array, dims=["time", "within-window time"],
                           coords={"within-window time": within_window_time_axis,
                                   "time": time_vector})
        # </Cast to xarray>
        return xrd

    def xarray_sliding_window(self, data, time_vector=None, dt=None):
        pass

    def compute_window_edge_indices(self, num_samples_data):
        """This has been useful in the past but maybe not needed here"""
        number_of_windows = self.available_number_of_windows(num_samples_data)
        self._left_hand_window_edge_indices = np.arange(
            number_of_windows)*self.num_samples_advance
        return

    def left_hand_window_edge_indices(self, num_samples_data):
        if self._left_hand_window_edge_indices is None:
            self.compute_window_edge_indices(num_samples_data)
        return self._left_hand_window_edge_indices

    def downsample_time_axis(self, time_axis):
        lhwe = self.left_hand_window_edge_indices(len(time_axis))
        multiple_window_time_axis = time_axis[lhwe]
        return multiple_window_time_axis

    def apply_taper(self, data):
        if isinstance(data, np.ndarray):
            tapered_windowed_data = self._apply_taper_numpy(data)
            return tapered_windowed_data
        elif isinstance(data, xr.DataArray):
            xrds = data.to_dataset("channel")
            tapered_obj = self.apply_taper(xrds)
            tapered_obj= tapered_obj.to_array("channel")

            print("not yet SOLVED")
            return tapered_obj
            #raise Exception
        elif isinstance(data, xr.Dataset):
            #overwriting
            #output_ds = xr.Dataset()
            for key in data.keys():
                print(f"key {key}")
                data[key].data = self._apply_taper_numpy(data[key].data)
                #tapered_obj = xr.DataArray()
                #output_ds.update({key: tapered_obj})
            return data
        else:
            print(f"Unexpected Data type {type(data)}")
            raise Exception



    def _apply_taper_numpy(self, data):
        """
        The axes are set up so that each rc? is tapered
        Parameters
        ----------
        data

        Returns
        -------

        """
        tapered_windowed_data = apply_taper_to_windowed_array(self.taper, data)
        return tapered_windowed_data

    def frequency_axis(self, dt):
        df = 1./(self.num_samples_window*dt)
        np.fft.fftfreq(self.num_samples_window, d=dt)
        pass

    def apply_fft(self, data, spectral_density_correction=True,
                  detrend_type="linear"):
        """
        lets assume we have already applied sliding window and taper.
        Things to think about:
        We want to assign the frequency axis during this method
        Maybe we should have
        Returns
        -------

        """
        #ONLY SUPPORTS DATASET AT THIS POINT
        if isinstance(data, xr.Dataset):
            spectral_ds = fft_xr_ds(data, self.sampling_rate,
                                    detrend_type=detrend_type)
            if spectral_density_correction:
                spectral_ds = self.apply_spectral_density_calibration(spectral_ds)
        elif isinstance(data, xr.DataArray):
            xrds = data.to_dataset("channel")
            spectral_ds = fft_xr_ds(xrds, self.sampling_rate,
                                    detrend_type=detrend_type)
            if spectral_density_correction:
                spectral_ds = self.apply_spectral_density_calibration(spectral_ds)
            spectral_ds = spectral_ds.to_array("channel")

            print("not yet SOLVED spectral")
            return spectral_ds

        else:
            print(f"fft of {type(data)} not yet supported")
            raise Exception

        return  spectral_ds

    def apply_spectral_density_calibration(self, dataset):
        """

        Parameters
        ----------
        dataset
        sample_rate

        Returns
        -------


        """
        scale_factor = self.spectral_density_calibration_factor
        if isinstance(dataset, xr.Dataset):
            for key in dataset.keys():
                print(f"applying spectral density calibration to {key}")
                dataset[key].data *= scale_factor
        else:
            print(f"scaling of {type(data)} not yet supported")
            raise Exception
        return dataset
#<PROPERTIES THAT NEED SAMPLING RATE>
#these may be moved elsewhere later
    @property
    def dt(self):
        """
        comes from data
        """
        return 1./self.sampling_rate

    @property
    def window_duration(self):
        """
        units are SI seconds assuming dt is SI seconds
        """
        return self.num_samples_window*self.dt

    @property
    def duration_advance(self):
        """
        """
        return self.num_samples_advance*self.dt

    @property
    def spectral_density_calibration_factor(self):
        factor = spectral_density_calibration_factor(self.coherent_gain, self.nenbw, self.dt, self.num_samples_window)
        return factor

#</PROPERTIES THAT NEED SAMPLING RATE>


def spectral_density_calibration_factor(coherent_gain, enbw, dt, N):
    """
    scales the spectra for the effects of the windowing, and converts to spectral density
    spectral_calibration = (1/0.54)*np.sqrt((2*0.025)/(1.36*288000)) #hamming
    Parameters
    ----------
    coherent_gain
    enbw
    dt
    N

    Returns
    -------

    """
    spectral_density_calibration_factor = (1./coherent_gain)*np.sqrt((2*dt)/(enbw*N))
    return spectral_density_calibration_factor



def fft_xr_ds(dataset, sample_rate, one_sided=True, detrend_type="linear"):
    """
    assume you have an xr.dataset or xr.DataArray.  It is 2D.
    This should call window_helpers.apply_fft_to_windowed_array
    or get moved to window_helpers.py
    Parameters
    ----------
    ds

    Returns
    -------

    TODO: Review nf_sjvk
    """
    import numpy as np
    #for each DataArray in the Dataset, we will apply fft along the within-window-time axis.
    #SET AXIS:
    operation_axis = 1
    output_ds = xr.Dataset()
    key0 = list(dataset.keys())[0]
    n_windows, samples_per_window = dataset[key0].data.shape

    dt = 1. / sample_rate
    if np.mod(samples_per_window,2)==0:
        n_fft_harmonics = int(samples_per_window / 2) + 1 #DC and Nyquist
    else:
        n_fft_harmonics = int(samples_per_window / 2) #No bin at Nyquist
    harmonic_frequencies = np.fft.fftfreq(samples_per_window, d=dt)
    if one_sided:
        harmonic_frequencies = harmonic_frequencies[0:n_fft_harmonics]
        harmonic_frequencies = np.abs(harmonic_frequencies)
        #if np.mod(samples_per_window, 2) == 0:
        #    harmonic_frequencies[-1] *= -1
        ##Nyquist is negative
    for key in dataset.keys():
        print(f"key {key}")
        data = dataset[key].data
        window_means = data.mean(axis=operation_axis)
        demeaned_data = (data.T - window_means).T
        if detrend_type: #neither False nor None
            #axis=1 should be changed to be xarray aware of the time axis
            #overwrite data=True probably best for most applications but
            #be careful with that.  Do we want to avoid this in general?
            #could we be possibly overwriting stuff on MTH5 in future?
            import scipy.signal as ssig
            ssig.detrend(demeaned_data, axis=1, overwrite_data=True,
                         type=detrend_type)

        print("MAY NEED TO ADD DETRENDING OR PREWHITENING HERE AS PREP FOR FFT")
        fspec_array = np.fft.fft(demeaned_data, axis=1)
        if one_sided:
            fspec_array = fspec_array[:,0:n_fft_harmonics]

        xrd = xr.DataArray(fspec_array, dims=["time", "frequency"],
                           coords={"frequency": harmonic_frequencies,
                                   "time": dataset.time.data})
        output_ds.update({key:xrd})
    return output_ds
