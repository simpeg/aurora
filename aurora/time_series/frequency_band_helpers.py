import numpy as np
import pandas as pd

from mt_metadata.transfer_functions.processing.aurora.band import (
    FrequencyBands,
)


def extract_band(frequency_band, fft_obj, epsilon=1e-7):
    """
    This may become a method of fft_obj, or frequency band.
    For now leave as stand alone.

    Note #1: 20230902
    drop=True does not play nice with h5py and Dataset, results in a type error.
    File "stringsource", line 2, in h5py.h5r.Reference.__reduce_cython__
    TypeError: no default __reduce__ due to non-trivial __cinit__
    However, it works OK with DataArray, so maybe use data array in general

    Parameters
    ----------
    frequency_band: aurora.interval.Interval()
        interval corresponding to a frequency band
    fft_obj: xr.DataArray
        To be replaced with an fft_obj() class in future
    epsilon: float
        Use this when you are worried about missing a frequency due to
        round off error.  This is in general not needed if we use a df/2 pad
        around true harmonics.

    Returns
    -------
    band: xr.DataArray
        The frequencies within the band passed into this function
    """
    cond1 = fft_obj.frequency >= frequency_band.lower_bound - epsilon
    cond2 = fft_obj.frequency <= frequency_band.upper_bound + epsilon
    try:
        band = fft_obj.where(cond1 & cond2, drop=True)
    except TypeError:  # see Note #1
        tmp = fft_obj.to_array()
        band = tmp.where(cond1 & cond2, drop=True)
        band = band.to_dataset("variable")
    return band


def frequency_band_edges(
    f_lower_bound, f_upper_bound, num_bands_per_decade=None, num_bands=None
):
    """
    Provides logarithmically spaced fenceposts across lowest and highest
    frequencies. This is a lot like calling logspace.  The resultant gates
    have constant  Q, i.e. deltaF/f_center=Q=constant.
    where f_center is defined geometircally, i.e. sqrt(f2*f1) is the center freq
    between f1 and f2.

    TODO: Add a linear spacing option?

    Parameters
    ----------
    f_lower_bound : float
        lowest frequency under consideration
    f_upper_bound : float
        highest frequency under consideration
    num_bands_per_decade : int (TODO test, float maybe ok also.. need to test)
        number of bands per decade
    num_bands : int
        total number of bands.  This supercedes num_bands_per_decade if supplied

    Returns
    -------
    fence_posts : array
        logarithmically spaced fenceposts acoss lowest and highest
        frequencies.  These partition the frequency domain between
        f_lower_bound and f_upper_bound
    """
    if (num_bands is None) & (num_bands_per_decade is None):
        print("Specify either number_of_bands or numnerbands_per_decade")
        raise Exception

    if num_bands is None:
        number_of_decades = np.log10(f_upper_bound / f_lower_bound)
        # The number of decades spanned (use log8 for octaves)
        num_bands = round(
            number_of_decades * num_bands_per_decade
        )  # floor or ceiling here?

    base = np.exp((1.0 / num_bands) * np.log(f_upper_bound / f_lower_bound))
    # log - NOT log10!

    print(f"base = {base}")
    bases = base * np.ones(num_bands + 1)
    print(f"bases = {bases}")
    exponents = np.linspace(0, num_bands, num_bands + 1)
    print(f"exponents = {exponents}")
    fence_posts = f_lower_bound * (bases**exponents)
    print(f"fence posts = {fence_posts}")
    return fence_posts