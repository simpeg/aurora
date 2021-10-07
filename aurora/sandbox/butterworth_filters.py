import scipy.signal as ssig


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
