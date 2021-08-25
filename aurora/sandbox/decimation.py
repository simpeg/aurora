"""
place holder for decimation

We will start by using the built in scipy.signal.decimate function as a default
but in general we want to allow the user to select their Anti-alias filter

#<Snippet>
This snippet was used to test prefiltering TimeSeries before TF processing.  It seemed
to improve the output TFs but its not clear how it compares against prewhiteing.
# import scipy.signal as ssig
# from aurora.time_series.frequency_domain_helpers import get_fft_harmonics
# num_samples_window = 128 #config.num_samples_window
# sample_rate = 1.0 # config.sample_rate
# freqs = get_fft_harmonics(num_samples_window, sample_rate)
# sos = ssig.butter(11, freqs[3], btype='highpass', fs=sample_rate, \
#                                             output='sos')
# for ch in local["mvts"].keys():
#     local["mvts"][ch].data = ssig.sosfilt(sos, local["mvts"][ch].data)
#</Snippet>
"""


class AntiAliasFilter:
    """ """

    def __init__(self):
        """
        filter_type : ["FIR", "IIR", "default"]
        """
        self.filter_type = "default"


class DecimationConfig(object):
    """ """

    def __init__(self):
        self.decimation_factors = []
