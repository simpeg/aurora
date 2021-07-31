import functools
import xarray as xr
import scipy.signal as ssig
from aurora.time_series.decorators import can_use_xr_dataarray

def schur_product_windowed_data(windowed_data, multiplier):
    """
    The axes are set up so that each window is tapered

    In particular, each "window" is a row of windowed_array.  Thus taper
    operates by multiplying, point-by-point (Schur) each row or windowed_array.

    TODO: either take an argument for which axis the taper applies along or
    make the calling function confirm that each row is a window and each
    column is a window-advance-delta-t

    Parameters
    ----------
    data

    Returns
    -------

    """
    tapered_windowed_data = windowed_array * taper  # seems to do sparse diag mult
    # time trial it against a few other methods
    return tapered_windowed_data


def validate_coordinate_ordering_time_domain(dataset):
    """
    Check that the data dimensions are what you expect.  THis may evolve some
    but for now, I just want to make sure that we are operating along the
    correct axes when we demean, detrend, taper, etc.
    Parameters
    ----------
    dataset : xarray.Dataset

    Returns
    -------

    """
    coordinate_labels = list(dataset.coords.keys())
    cond1 = coordinate_labels[0] == "within-window time"
    cond2 = coordinate_labels[1] == "time"
    if (cond1 & cond2):
        return True
    else:
        print("Uncertain that xarray coordinates are correctly ordered")
        raise Exception

def get_time_coordinate_axis(dataset):
    """
    It is common to pass an argument to scipy.signal methods axis=int
    where that integer specifies along which dimension we are applying the
    operator.  This method helps ensure that we have the correct axis.
    Parameters
    ----------
    dataset : xarray.Dataset

    Returns
    -------

    """
    coordinate_labels = list(dataset.coords.keys())

    if len(coordinate_labels) != 2:
        print("Warning - Expected two distinct coordinates")
        #raise Exception

    return coordinate_labels.index("time")
    # time_coord_indices = [ndx for x, ndx in enumerate(coordinate_labels) if
    #                                                   x=="time"]
    # if len(time_coord_indices) == 1:
    #     return time_coord_indices[0]
    # else:
    #     print("expected only one universal time coordinate")
    #     raise Exception



class WindowedTimeSeries(object):
    """
    Time series that has been chopped into (possibly) overlapping windows.

    This is a place where we can put methods that operate on these sorts of
    objects.

    The assumption is that we take xarrays keyed by "channel"

    Specific methods:
        Demean
        Detrend
        Prewhiten
        stft
        invert_prewhitening

        probably make these @staticmethod s so we import WindowedTimeSeries
        and then call the static methods
    """
    def __init__(self):
        pass

    @can_use_xr_dataarray
    @staticmethod
    def apply_taper(data=None, taper=None, in_place=True):
        """
        it turns out xarray handles this very cleanly as a direct multiply
        operation.  Initially I was looping over channels and multiplying
        each array using
        tapered_obj = WindowedTimeSeries.apply_taper(data=windowed_obj,
    #                                           taper=windowing_scheme.taper)
        but one can simply call:
        tapered_obj = windowed_obj * windowing_scheme.taper

        Thus this method will be deprecated.
        """
        data = data * taper
        # validate_coordinate_ordering_time_domain(data)
        #
        # for key in data.keys():
        #     print(f"key {key}")
        #     windowed_array = data[key].data
        #     tapered_windowed_data = windowed_array * taper
        #     data[key].data = tapered_windowed_data

        return data


    @staticmethod
    def detrend(data=None, detrend_axis=None, detrend_type=None, inplace=True):
        """
        TODO: overwrite data=True probably best for most applications but
            be careful with that.  Do we want to avoid this in general?
            could we be possibly overwriting stuff on MTH5 in future?
            Also, is overwrite even working how I think it is here?
        TODO: overwrite_data not working right in scipy.signal, dont use it
        for now
        Parameters
        ----------
        data
        detrend_axis
        detrend_type : string
            "linear" or "constant"
            This argument is provided to scipy.signal.detrend

        Returns
        -------

        """
        if detrend_axis is None:
            detrend_axis = get_time_coordinate_axis(data)
        if not inplace:
            print("deep copy dataset and then overwrite")
            raise NotImplementedError

        for key in data.keys():
            print(f"key {key}")
            print("By modifying windowed_array below am I modifying data")

            windowed_array = data[key].data

            if detrend_type: #neither False nor None
                windowed_array = ssig.detrend(windowed_array,
                                              axis=detrend_axis,
                                              type=detrend_type)
                                              #overwrite_data=True

            if inplace:
                data[key].data = windowed_array
            else:
                print("deep copy dataset and then overwrite")
                raise NotImplementedError
        return data

    @staticmethod
    def apply_stft(data=None, sampling_rate=None, detrend_type=None,
                   spectral_density_calibration=1.0, fft_axis=None):
        """
        Only supports xr.Dataset at this point

        Parameters
        ----------
        data
        sampling_rate
        detrend_type

        Returns
        -------

        """
        if fft_axis is None:
            fft_axis = get_time_coordinate_axis(data)
        spectral_ds = fft_xr_ds(data, sampling_rate,
                                    detrend_type=detrend_type)
        spectral_ds *= spectral_density_calibration
        return spectral_ds