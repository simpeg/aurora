import numpy as np



def extract_band(frequency_band, fft_obj, epsilon=1e-7):
    """
    TODO: THis may want to be a method of fft_obj, or it may want to be a
    method of frequency band.  For now leave as stand alone.
    Parameters
    ----------
    interval
    fft_obj: xr.DataArray
    epsilon: use this when you are worried about missing a frequency due to
    round off error.  THis is in general not needed if we use a df/2 pad
    around true harmonics.

    Returns xr.DataArray
    -------

    """
    cond1 = fft_obj.frequency >= frequency_band.lower_bound - epsilon
    cond2 = fft_obj.frequency <= frequency_band.upper_bound + epsilon

    band = fft_obj.where(cond1 & cond2, drop=True)
    return band

def frequency_band_edges(f_lower_bound, f_upper_bound,
                                  num_bands_per_decade=None, num_bands=None):
    """
    Provides logarithmically spaced fenceposts acoss lowest and highest
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

    Returns :
    fence_posts : array
        logarithmically spaced fenceposts acoss lowest and highest
        frequencies.  These partition the frequency domain between
        f_lower_bound and f_upper_bound
    -------



    """
    if (num_bands is None) & (num_bands_per_decade is None):
        print("Specify either number_of_bands or numnerbands_per_decade")
        raise Exception

    if num_bands is None:
        number_of_decades = np.log10(f_upper_bound/ f_lower_bound)
        # The number of decades spanned (use log8 for octaves)
        num_bands = round(number_of_decades * num_bands_per_decade) #floor or ceiling here?


    base = np.exp((1.0 / num_bands) * np.log(f_upper_bound / f_lower_bound))
    # log - NOT log10!

    print(f"base = {base}")
    bases = base * np.ones(num_bands + 1);
    print(f"bases = {bases}")
    exponents = np.linspace(0, num_bands, num_bands + 1)
    print(f"exponents = {exponents}")
    fence_posts = f_lower_bound * (bases ** exponents)
    print(f"fence posts = {fence_posts}")
    return fence_posts
