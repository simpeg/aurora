import numpy as np
from aurora.time_series.frequency_band import FrequencyBands


def extract_band(frequency_band, fft_obj, epsilon=1e-7):
    """
    TODO: This may want to be a method of fft_obj, or it may want to be a
    method of frequency band.  For now leave as stand alone.

    Parameters
    ----------
    fft_obj: xr.DataArray
        To be replaced with an fft_obj() class in future
    epsilon: float
        Use this when you are worried about missing a frequency due to
        round off error.  This is in general not needed if we use a df/2 pad
        around true harmonics.

    Returns xr.DataArray
    -------

    """
    cond1 = fft_obj.frequency >= frequency_band.lower_bound - epsilon
    cond2 = fft_obj.frequency <= frequency_band.upper_bound + epsilon

    band = fft_obj.where(cond1 & cond2, drop=True)
    return band


def frequency_band_edges(
    f_lower_bound, f_upper_bound, num_bands_per_decade=None, num_bands=None
):
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
    fence_posts = f_lower_bound * (bases ** exponents)
    print(f"fence posts = {fence_posts}")
    return fence_posts


def configure_frequency_bands(config):
    """
    May want to make config a "frequency band config object", but maybe not.
    For now just using a flat config structure (per decimation level)

    These methods could also be placed under FrequencyBands() class as
    init_from_emtf()
    init_from_bounds_array()
    init_from_default()


    Parameters
    ----------
    config : aurora.config.decimation_level_config.DecimationLevelConfig
        The configuration parameters for setting up the frequency bands.

        If config["band_setup_style"] is "EMTF" this will look for one of
        Gary's "band_setup" files and parse it.  It will look for
        config.emtf_band_setup_file.

        Other options would be :
        1. "band_edges", accepts an array of lower_bound, upper_bound pairs
        2. "logarithmic range": could accept a lower_bound, and an
        upper_bound, and a number of bands inbetween., or it could estimate
        a)lower bound from a rule about the minimum number of cycles needed
        for an estimate (say 5 or 10)
        b) upper bound from a Nyquist rule, say 80% f_Nyquist


    Returns
    -------
    frequency_bands : aurora.time_series.frequency_band.FrequencyBands
        a fully populated FrequencyBands object with all info needed to do
        band averaging.
    """
    frequency_bands = FrequencyBands()
    if config["band_setup_style"] == "EMTF":
        frequency_bands.from_emtf_band_setup(
            filepath=config.emtf_band_setup_file,
            sample_rate=config.sample_rate,
            decimation_level=config.decimation_level_id + 1,
            num_samples_window=config.num_samples_window,
        )
    elif config["band_setup_style"] == "band edges":
        frequency_bands.band_edges = config["band_edges"]
        # "Not Yet Supported"
        raise NotImplementedError
    elif config["band_setup_style"] == "logarithmic range":
        lower_bound = config["frequency_bands_lower_bound"]
        upper_bound = config["frequency_bands_upper_bound"]
        num_bands = config["num_frequency_bands"]
        if lower_bound is None:
            pass
            # suggest lower_bound from a rule
        if upper_bound is None:
            pass
            # suggest upper_bound from a rule
        if num_bands is None:
            pass
            # suggest based on num_bands per octave or decade
        # now call logspace(lower, upper, num_bands)
        raise NotImplementedError

    return frequency_bands

def df_from_bands(band_list):
    """
    Note: The decimation_level here is +1 to agree with EMTF convention. 
    Not clear this is really necessary
    
    Parameters
    ----------
    band_list: list
        obtained from aurora.config.metadata.decimation_level.DecimationLevel.bands

    Returns
    -------
        out_df: pd.Dataframe
        Same format as that generated by EMTFBandSetupFile.get_decimation_level()

    """
    import pandas as pd
    df_columns = ['decimation_level', 'lower_bound_index', 'upper_bound_index']
    n_rows = len(band_list)
    df_columns_dict = {}
    for col in df_columns:
        df_columns_dict[col] = n_rows * [None]
    for i_band, band in enumerate(band_list):
        df_columns_dict["decimation_level"][i_band] = band.decimation_level + 1
        df_columns_dict["lower_bound_index"][i_band] = band.index_min
        df_columns_dict["upper_bound_index"][i_band] = band.index_max
    out_df = pd.DataFrame(data=df_columns_dict)
    out_df.sort_values(by="lower_bound_index", inplace=True)
    return out_df
