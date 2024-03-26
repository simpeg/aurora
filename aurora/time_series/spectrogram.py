import xarray as xr

from aurora.time_series.frequency_band_helpers import extract_band
from loguru import logger


class Spectrogram(object):
    """
    Class to contain methods for STFT objects.
    TODO: Add support for cross powers
    TODO: Add OLS Z-estimates
    TODO: Add Sims/Vozoff Z-estimates

    """

    def __init__(self, dataset=None):
        """

        Parameters
        ----------
        dataset: xarray.core.dataset.Dataset
            Chunk of a STFT, having dimensions time and frequency
        """
        self._dataset = dataset
        self._frequency_increment = None
        self._cross_powers = None

    @property
    def dataset(self):
        return self._dataset

    @property
    def cross_powers(self):
        if self._cross_powers is None:
            self._cross_powers = xr.Dataset(coords={"time": self.dataset.time.data})
        return self._cross_powers

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
        band: mt_metadata.transfer_functions.processing.aurora.band.Band
            The frequency band object
        epsilon: float
            Prevents numeric round-off errors checking harmonics are in band
        Returns
        -------
        num_harmonics: int
            The number of frequency bins within the frequency band
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

    def cross_power(self, ch1, ch2, augment=True):
        """
        TODO: This method should normally be used on a Spectrogram that has
        been extracted (a tf_estimation band).  Consider extending
        the class or somehow marking it that the band extraction has happened
        TODO: The cross power should get put into its own xr.Dataset, not into the spectrogram

        Convention:
        "The two-sided spectral density function between two random processes is defined
        using X*Y and _not_ YX*" -- Bendat & Perisol (1980, p54, Eqn 3.45).

        Parameters
        ----------
        ch1: str
            The channel name of the first channel in the cross-spectrum
        ch2: str
            The channel name of the second channel in the cross-spectrum


        Returns
        -------

        """
        name = f"{ch1}_{ch2}"
        try:
            return self.cross_powers[name]
        except KeyError:
            msg = f"did not find stored cross power {name}; Computing {name}"
            logger.info(msg)
        X = self.dataset[ch1]
        Y = self.dataset[ch2]
        xHy = (X.conj().transpose() * Y).sum(dim="frequency")
        if ch1 == ch2:
            xHy = xHy.__abs__()
        if augment:
            name = f"{ch1}_{ch2}"
            self.cross_powers[name] = xHy
        return xHy
