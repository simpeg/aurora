from aurora.time_series.frequency_band_helpers import extract_band


class Spectrogram(object):
    """
    Class to contain methods for STFT objects.
    TODO: Add support for cross powers
    TODO: Add OLS Z-estimates
    TODO: Add Sims/Vozoff Z-estimates

    """

    def __init__(self, dataset=None):
        self._dataset = dataset
        self._frequency_increment = None

    @property
    def dataset(self):
        return self._dataset

    @property
    def time_axis(self):
        return self.dataset.time

    @property
    def frequency_increment(self):
        if self._frequency_increment is None:
            frequency_axis = self.dataset.frequency
            self._frequency_increment = frequency_axis.data[1] - frequency_axis.data[0]
        return self._frequency_increment

    def num_harmonics_in_band(self, frequency_band, epsilon=1e-7):
        """

        Parameters
        ----------
        band
        stft_obj

        Returns
        -------

        """
        cond1 = self._dataset.frequency >= frequency_band.lower_bound - epsilon
        cond2 = self._dataset.frequency <= frequency_band.upper_bound + epsilon
        num_harmonics = (cond1 & cond2).data.sum()
        return num_harmonics

    def extract_band(self, frequency_band, channels=[]):
        """
        TODO: Check if this should be returning a copy of the data...

        Parameters
        ----------
        frequency_band
        channels

        Returns
        -------
        spectrogram: aurora.time_series.spectrogram.Spectrogram
            Returns a Spectrogram object with only the extracted band for a dataset

        """
        extracted_band_dataset = extract_band(
            frequency_band, self.dataset, channels=channels, epsilon=1e-7
        )
        spectrogram = Spectrogram(dataset=extracted_band_dataset)
        return spectrogram

    def cross_powers(self, ch1, ch2, band=None):
        pass
