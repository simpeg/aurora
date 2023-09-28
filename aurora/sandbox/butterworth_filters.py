"""


"""
import scipy.signal as ssig
from mt_metadata.transfer_functions.processing.aurora.decimation_level import (
    get_fft_harmonics,
)


def butter_bandpass(low_cut, high_cut, sample_rate, order=5):
    """
    https://scipy.github.io/old-wiki/pages/Cookbook/ButterworthBandpass
    """
    nyquist_frequency = 0.5 * sample_rate
    low = low_cut / nyquist_frequency
    high = high_cut / nyquist_frequency
    b, a = ssig.butter(order, [low, high], btype="band", analog=False)
    return b, a


def butter_highpass(f_cut, fs, order=5):
    """
    This is based on butter_bandpass
    """
    f_nyq = 0.5 * fs
    normalized_frequency = f_cut / f_nyq
    b, a = ssig.butter(order, normalized_frequency, btype="high", analog=False)
    return b, a


def butter_lowpass(cutoff, fs, order=5):
    """
    This is based on butter_bandpass
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = ssig.butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_highpass_by_harmonic_index(
    num_samples=128,
    sample_rate=1.0,
    n_harmonic_cutoff=3,
    filter_order=11,
):
    """
    application of the sos filter created could use for example:
    for ch in local["mvts"].keys():
        local["mvts"][ch].data = ssig.sosfilt(sos, local["mvts"][ch].data)

    Parameters
    ----------
    num_samples_window
    sample_rate
    n_harmonic_cutoff
    filter_order

    Returns
    -------

    """
    freqs = get_fft_harmonics(num_samples, sample_rate)
    sos = ssig.butter(
        filter_order,
        freqs[n_harmonic_cutoff],
        btype="highpass",
        fs=sample_rate,
        output="sos",
    )
    return sos
