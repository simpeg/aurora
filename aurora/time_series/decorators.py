"""
# Here is the decorator pattern
def decorator(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        # Do something before
        value = func(*args, **kwargs)
        # Do something after
        return value

    return wrapper_decorator
"""

import functools
import xarray as xr


def can_use_xr_dataarray(func):
    """
    Intended as a decorator.  Most of the windowed time series methods are
    intended to work with xarray.Dataset class.  But I would like to be able
    to pass them xarray.DataArray objects.  This class casts a DataArray to a
    Dataset, runs it through func and casts back to a DataArray.

    A simuilar decorator should be written for numpy arrays.
    Parameters
    ----------
    func

    Returns
    -------

    """

    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):

        if isinstance(kwargs["data"], xr.DataArray):
            kwargs["data"] = kwargs["data"].to_dataset("channel")
            input_was_dataarray = True
        else:
            input_was_dataarray = False

        if isinstance(func, staticmethod):
            callable_func = func.__get__(None, object)
            processed_obj = callable_func(*args, **kwargs)
        else:
            processed_obj = func(*args, **kwargs)

        if input_was_dataarray:
            processed_obj = processed_obj.to_array("channel")
        return processed_obj

    return wrapper_decorator
