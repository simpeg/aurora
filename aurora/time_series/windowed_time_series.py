import numpy as np
import scipy.signal as ssig

from scipy.interpolate import interp1d

from aurora.time_series.decorators import can_use_xr_dataarray


def validate_coordinate_ordering_time_domain(dataset):
    """
    Check that the data dimensions are what you expect.  This may evolve, but for now,
    just want to make sure that we are operating along the correct axes when we
    demean, detrend, taper, etc.
    Parameters
    ----------
    dataset : xarray.Dataset

    Returns
    -------

    """
    coordinate_labels = list(dataset.coords.keys())
    cond1 = coordinate_labels[0] == "within-window time"
    cond2 = coordinate_labels[1] == "time"
    if cond1 & cond2:
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
        # raise Exception

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
        Point by point multiplication of taper against time series.
        xarray handles this very cleanly as a direct multiply operation.
        tapered_obj = windowed_obj * windowing_scheme.taper
        """
        data = data * taper
        return data

    @staticmethod
    def detrend(data=None, detrend_axis=None, detrend_type=None, inplace=True):
        """
        Notes: overwrite data=True probably best for most applications but be careful
         with that.  Do we want to avoid this in general? Could we be possibly
         overwriting stuff on MTH5 in future?
         Also, is overwrite even working how I think it is here?
         Overwrite_data not working right in scipy.signal, dont use it for now

        Parameters
        ----------
        data : xarray Dataset
        detrend_axis : string
        detrend_type : string
            "linear" or "constant"
            This argument is provided to scipy.signal.detrend

        Returns
        -------

        """
        if detrend_axis is None:
            detrend_axis = get_time_coordinate_axis(data)
        if not inplace:
            raise NotImplementedError

        for channel in data.keys():

            # windowed_array = data[key].data
            nanless_data = data[channel].dropna(dim="time")
            ensembles = nanless_data.data
            if detrend_type:  # neither False nor None
                try:
                    ensembles = ssig.detrend(
                        ensembles, axis=detrend_axis, type=detrend_type
                    )
                    # overwrite_data=True
                except ValueError as error:
                    msg = (
                        "Could not detrend "
                        f"{channel} in time range "
                        f"{data[channel].coords.indexes['time'][0].isoformat()} to "
                        f"{data[channel].coords.indexes['time'][-1].isoformat()}."
                    )
                    if ensembles.size == 0:
                        print(msg + " NO DATA")
                    else:
                        print(msg + "UNKOWN REASON:" + error)

            if inplace:
                if len(nanless_data.time) < len(data[channel].time):
                    data[channel].data += np.nan
                    data[channel].loc[nanless_data.time, :] = ensembles
                    # there must be a cute way to just assign all nan to just the
                    # columns that had nan values in some places, rather than
                    # assigning nan to the whole array and then overwriting the
                    # ensembles .. something like
                    # data[channel].loc[~nanless_data.time, :] = np.nan
                else:
                    data[channel].data = ensembles
            else:
                raise NotImplementedError
        return data

    @staticmethod
    def apply_stft(
        data=None,
        sample_rate=None,
        detrend_type=None,
        spectral_density_calibration=1.0,
        fft_axis=None,
    ):
        """
        Only supports xr.Dataset at this point

        Parameters
        ----------
        data
        sample_rate
        detrend_type

        Returns
        -------

        """
        from aurora.time_series.windowing_scheme import fft_xr_ds

        if fft_axis is None:
            fft_axis = get_time_coordinate_axis(data)
        spectral_ds = fft_xr_ds(data, sample_rate, detrend_type=detrend_type)
        spectral_ds *= spectral_density_calibration
        return spectral_ds

    def delay_correction(self, dataset, run_obj):
        """

        Parameters
        ----------
        dataset : xr.Dataset
        run_obj :

        Returns
        -------

        """
        # "NOT TESTED - PSEUDOCODE ONLY"
        for channel_id in dataset.keys():
            mth5_channel = run_obj.get_channel(channel_id)
            channel_filter = mth5_channel.channel_response_filter
            delay_in_seconds = channel_filter.total_delay
            true_time_axis = dataset.time + delay_in_seconds
            interpolator = interp1d(
                true_time_axis, dataset[channel_id].data, assume_sorted=True
            )
            corrected_data = interpolator(dataset.time)
            dataset[channel_id].data = corrected_data
        return dataset
