import numpy as np


def get_fft_harmonics(samples_per_window, sample_rate, one_sided=True):
    """
    Works for odd and even number of points.
    Does not return Nyquist, does return DC component
    Could be midified with kwargs to support one_sided, two_sided, ignore_dc
    ignore_nyquist, and etc.  Could actally take FrequencyBands as an argument
    if we wanted as well.

    Parameters
    ----------
    samples_per_window
    sample_rate

    Returns
    -------

    """
    n_fft_harmonics = int(samples_per_window / 2)  # no bin at Nyquist,
    harmonic_frequencies = np.fft.fftfreq(samples_per_window, d=1.0 / sample_rate)
    harmonic_frequencies = harmonic_frequencies[0:n_fft_harmonics]
    return harmonic_frequencies
