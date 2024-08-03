"""

This module contains some helper functions that are used when working with sliding windows.

Development Notes:
 - Not all the functions here are needed, some of them are just examples and tests.
 - for example there are three sliding window functions that were considered.
   The idea was to compare their performance, but currently we use `sliding_window_crude`.



Notes in google doc:
https://docs.google.com/document/d/1CsRhSLXsRG8HQxM4lKNqVj-V9KA9iUQAvCOtouVzFs0/edit?usp=sharing


"""
import numpy as np
import time

from loguru import logger
from numba import jit
from numpy.lib.stride_tricks import as_strided
from typing import Optional, Union


def available_number_of_windows_in_array(
    n_samples_array: int, n_samples_window: int, n_advance: int
) -> int:
    """
    Returns the number of whole windows that can be extracted from array of length
     n_samples_array by a window of length n_samples_window, if the window advances
     by n_advance samples at each step.

    Parameters
    ----------
    n_samples_array: int
        The length of the time series
    n_samples_window: int
        The length of the window (in samples)
    n_advance: int
        The number of samples the window advances at each step

    Returns
    -------
    available_number_of_strides: int
        The number of windows the time series will yield
    """
    stridable_samples = n_samples_array - n_samples_window
    if stridable_samples < 0:
        logger.error(
            "Window is longer than the time series -- no complete windows can be returned"
        )
        return 0
    available_number_of_strides = int(np.floor(stridable_samples / n_advance))
    available_number_of_strides += 1
    return available_number_of_strides


# Sliding Window Operators
def sliding_window_crude(
    data, num_samples_window, num_samples_advance, num_windows=None
):
    """
    Reshapes input data with a sliding window.

    Parameters
    ----------
    data: np.ndarray
        The time series data to be windowed
    num_samples_window: int
        The length of the window (in samples)
    num_samples_advance: int
        The number of samples the window advances at each step
    num_windows: int
        The number of windows to "take".  Must be less or equal to the number of
        available windows.

    Returns
    -------
    output_array: numpy.ndarray
        The windowed time series
    """
    if num_windows is None:
        num_windows = available_number_of_windows_in_array(
            len(data), num_samples_window, num_samples_advance
        )
    output_array = np.full((num_windows, num_samples_window), np.nan)
    for i in range(num_windows):
        output_array[i, :] = data[
            i * num_samples_advance : i * num_samples_advance + num_samples_window
        ]

    return output_array


@jit(nopython=True)
def sliding_window_numba(data, num_samples_window, num_samples_advance, num_windows):
    """
    Reshapes input data with a sliding window.

    Parameters
    ----------
    data: np.ndarray
        The time series data to be windowed
    num_samples_window: int
        The length of the window (in samples)
    num_samples_advance: int
        The number of samples the window advances at each step
    num_windows: int
        The number of windows to "take".

    Returns
    -------
    output_array: numpy.ndarray
        The windowed time series
    """
    output_array = np.full((num_windows, num_samples_window), np.nan)
    for i in range(num_windows):
        output_array[i, :] = data[
            i * num_samples_advance : i * num_samples_advance + num_samples_window
        ]

    return output_array


def striding_window(
    data: np.ndarray,
    num_samples_window: int,
    num_samples_advance: int,
    num_windows: Optional[Union[int, None]] = None,
) -> np.ndarray:
    """
    Reshapes input data with a sliding window.

    Not currently used.

    Development Notes:
    Applies a striding window to an array.  We use 1D arrays here.
    Note that this method is extendable to N-dimensional arrays as was once shown
    at  http://www.johnvinyard.com/blog/?p=268

    Here the code is restricted to 1D.
    This is because of several warnings encountered, on the notes of stride_tricks.py,
    as well as for example here:
    https://stackoverflow.com/questions/4936620/using-strides-for-an-efficient-moving-average-filter

    While we can possibly set up Aurora so that no copies of the strided window are made
    downstream, we cannot guarantee that another user may not add methods that require
    copies.  To avoid this, use 1d implementation only for now.

    Another clean example of this method can be found in the razorback codes from brgm.

    Parameters
    ----------
    data: np.ndarray
        The time series data to be windowed
    num_samples_window: int
        The length of the window (in samples)
    num_samples_advance: int
        The number of samples the window advances at each step
    num_windows: int
        The number of windows to "take".  Must be less or equal to the number of
        available windows.

    Returns
    -------
    strided_window: numpy.ndarray
        The windowed time series.  result is 2d: result[i] is the i'th window.

    """
    if num_windows is None:
        num_windows = available_number_of_windows_in_array(
            len(data), num_samples_window, num_samples_advance
        )
    bytes_per_element = data.itemsize
    output_shape = (num_windows, num_samples_window)
    strides_shape = (num_samples_advance * bytes_per_element, bytes_per_element)
    logger.debug("strides_shape", strides_shape)
    strided_window = as_strided(
        data, shape=output_shape, strides=strides_shape
    )  # , writeable=False)
    return strided_window


# Sliding Window Functions
SLIDING_WINDOW_FUNCTIONS = {
    "crude": sliding_window_crude,
    "numba": sliding_window_numba,
    "stride": striding_window,
}


def check_all_sliding_window_functions_are_equivalent() -> None:
    """
    This is a test to see if the sliding window functions all return the same output.

    TODO: Move this into tests.

    Development Notes:
    - simple sanity check that runs each sliding window function on a small array and
    confirms the results are numerically identical.
    - Note that striding window will return int types where others return float.

    """

    N = 15  # Num samples data
    L = 3  # n_samples_window
    V = 1  # n_overlap
    A = L - V  # n_advance
    data = np.arange(N)
    n_win = available_number_of_windows_in_array(N, L, A)
    results = {}
    for function_label, function in SLIDING_WINDOW_FUNCTIONS.items():
        results[function_label] = function(data, L, A, n_win)
        logger.info(results[function_label])

    for i, function_label in enumerate(results.keys()):
        if i == 0:
            reference_result = results[function_label]
        else:
            difference = reference_result - results[function_label]
            if np.sum(np.abs(difference)) == 0:
                assert True
            else:
                assert False


def do_some_tests():
    """
    A placeholder for things that should be moved to tests/

    TODO: Move these into tests

    """
    # Set parameters for test
    N = 10000000  # num samples data
    n_samples_window = 128
    n_overlap = 96
    n_advance = n_samples_window - n_overlap

    # Test that striding window executes on a toy dataset
    sw = striding_window(np.arange(15), 3, 2, num_windows=4)
    logger.info(sw)

    # Test speed of striding_window on time series length N
    t0 = time.time()
    strided_window = striding_window(1.0 * np.arange(N), n_samples_window, n_advance)
    strided_window += 1
    logger.info("stride {}".format(time.time() - t0))
    logger.info(strided_window)

    # Test speed of sliding_window_crude on time series length N
    t0 = time.time()
    slid_window = sliding_window_crude(
        1.0 * np.arange(N), n_samples_window, n_advance
    )  # , num_windows=4)
    slid_window += 1
    logger.info("crude  {}".format(time.time() - t0))
    logger.info(slid_window)

    # Test speed of sliding_window_numba on time series length N
    num_windows = available_number_of_windows_in_array(N, n_samples_window, n_advance)
    logger.info(num_windows)
    t0 = time.time()
    numba_slid_window = sliding_window_numba(
        1.0 * np.arange(N), n_samples_window, n_advance, num_windows
    )  # , num_windows=4)
    numba_slid_window += 1
    logger.info("numba  {}".format(time.time() - t0))

    t0 = time.time()
    numba_slid_window = sliding_window_numba(
        1.0 * np.arange(N), n_samples_window, n_advance, num_windows
    )  # , num_windows=4)
    logger.info("numba  {}".format(time.time() - t0))


def test_apply_taper() -> None:
    """
    Tests that syntax to apply taper is correct.

    Makes a plit showing the first time window with and without taper

    """
    import matplotlib.pyplot as plt
    import scipy.signal as ssig

    num_samples_window = 64
    num_windows = 100
    windowed_data = np.abs(np.random.randn(num_windows, num_samples_window))
    taper = ssig.windows.hann(num_samples_window)
    tapered_windowed_data = windowed_data * taper
    plt.plot(windowed_data[0], "r", label="data")
    plt.plot(tapered_windowed_data[0], "g", label="tapered data")
    plt.legend()
    plt.show()


def main():
    """Placeholder for tests"""
    check_all_sliding_window_functions_are_equivalent()
    do_some_tests()
    test_apply_taper()


if __name__ == "__main__":
    """Allow module to be called from terminal window"""
    main()


# """
#     This function was originally copied from:
#         http://www.johnvinyard.com/blog/?p=268
# """
#
# def norm_shape(shape):
#     '''
#     Normalize numpy array shapes so they're always expressed as a tuple,
#     even for one-dimensional shapes.
#
#     Parameters
#         shape - an int, or a tuple of ints
#
#     Returns
#         a shape tuple
#     @note: 20140221, kkappler: this method taken from
#     http://www.johnvinyard.com/blog/?p=268, and used by ensemblizedTimeSeries
#     but probably will be used by other methods.
#     It is less a signal processing method and more a numpy-array-handling
#     tool, in case we ever get that specific with files/functions.
#
#     '''
#     try:
#         i = int(shape)
#         return (i, )
#     except TypeError:
#         # shape was not a number
#         pass
#
#     try:
#         t = tuple(shape)
#         return t
#     except TypeError:
#         # shape was not iterable
#         pass
#
#     raise TypeError('shape must be an int, or a tuple of ints')
#
#
# #</Adding these methods which were developed at QF and needed for first version ops>
# def sliding_window(a, ws, ss=None, flatten=True):
#     '''
#     This function was originally copied from:
#         http://www.johnvinyard.com/blog/?p=268
#
#     usage: sw = sliding_window(a,ws,ss = None,flatten = True):
#     Return a sliding window over a in any number of dimensions
#
#     Parameters:
#         a  - an n-dimensional numpy array
#         ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size
#              of each dimension of the window
#         ss - an int (a is 1D) or tuple (a is 2D or greater) representing the
#              amount to slide the window in each dimension. If not specified, it
#              defaults to ws.
#         flatten - if True, all slices are flattened, otherwise, there is an
#                   extra dimension for each dimension of the input.
#
#     Returns
#         Array, if 2d dimensions are (num_windows, n_points_per_window)
#
#         An array containing each n-dimensional window from a
#         The Ensemblized time series seems in many ways like a MVTS, but it is NOT.
#     It looks like one, because the data are a 2D array, with time moving along the
#     "row" or zero axis, but the similarities stop there.
#     The time axis is different for each row!  Each row spans the same duration,
#     but the actual interval t0+row_duration is difference for each row,, i.e. the
#     t0 is different for each row.
#
#     In the case where the window length is 1, we obntain the special case where
#     the ensemblized time series is just the transpose (not-conjugate) of the input
#     time series.
#
#     There is an excellent article on ensemblization in numpy
#     http://www.johnvinyard.com/blog/?p=268
#
#     @note: For many ensemble processing applications you are NOT obliged
#     to hold the ensembles in memory at a given time.  The old
#     ensembleEdges class was designed with this in mind and probably would
#     be slightly less RAM-heavy.  Prior to 2011 I always created ensemblizedTimeSeries
#     in my codes (even if they were not called by that name) because of the
#     inate support for vector math
#     @note: the ws here is the distance the window slides whcih is L-V, i.e. it
#     is not the Overlap, but rather its difference from L.
#     '''
#     if None is ss:
#         # ss was not provided. the windows will not overlap in any direction.
#         ss = ws
#     ws = norm_shape(ws)
#     ss = norm_shape(ss)
#
#     # convert ws, ss, and a.shape to numpy arrays so that we can do math in every
#     # dimension at once.
#     ws = np.array(ws)
#     ss = np.array(ss)
#     shape = np.array(a.shape)
#
#     # ensure that ws, ss, and a.shape all have the same number of dimensions
#     ls = [len(shape), len(ws), len(ss)]
#     if len(set(ls)) != 1:
#         raise ValueError(\
#         'a.shape, ws and ss must all have the same length. They were %s' % str(ls))
#
#     # ensure that ws is smaller than a in every dimension
#     if np.any(ws > shape):
#         raise ValueError(\
#         'ws cannot be larger than a in any dimension.\
#  a.shape was %s and ws was %s' % (str(a.shape), str(ws)))
#
#     # how many slices will there be in each dimension?
#     newshape = norm_shape(((shape - ws) // ss) + 1)
#     # the shape of the strided array will be the number of slices in each dimension
#     # plus the shape of the window (tuple addition)
#     newshape += norm_shape(ws)
#     # the strides tuple will be the array's strides multiplied by step size, plus
#     # the array's strides (tuple addition)
#     newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
#     strided = ast(a, shape=newshape, strides=newstrides)
#     if not flatten:
#         return strided
#
#     # Collapse strided so that it has one more dimension than the window.  I.e.,
#     # the new array is a flat list of slices.
#     meat = len(ws) if ws.shape else 0
#     firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
#     dim = np.array(firstdim + (newshape[-meat:]))
#     # remove any dimensions with size 1
#     dim = dim[dim != 1]
#     return strided.reshape(dim)
