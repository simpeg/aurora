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
    samples_per_window: integer
        Number of samples in a window that will be Fourier transformed.
    sample_rate: float
            Inverse of time step between samples,
            Samples per second

    Returns
    -------
    harmonic_frequencies: numpy array
        The frequencies that the fft will be computed
    """
    n_fft_harmonics = int(samples_per_window / 2)  # no bin at Nyquist,
    delta_t = 1.0 / sample_rate
    harmonic_frequencies = np.fft.fftfreq(samples_per_window, d=delta_t)
    harmonic_frequencies = harmonic_frequencies[0:n_fft_harmonics]
    return harmonic_frequencies
