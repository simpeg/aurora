"""
 WORK IN PROGRESS (WIP): This module contains a class that represents a spectrogram,
 i.e. A 2D time series of Fourier coefficients with axes time and frequency.

"""
from aurora.time_series.frequency_band_helpers import extract_band
from typing import Optional
import xarray


class Spectrogram(object):
    """
    Class to contain methods for STFT objects.
    TODO: Add support for cross powers
    TODO: Add OLS Z-estimates
    TODO: Add Sims/Vozoff Z-estimates

    """

    def __init__(self, dataset=None):
        """Constructor"""
        self._dataset = dataset
        self._frequency_increment = None

    def _lowest_frequency(self):
        pass

    def _higest_frequency(self):
        pass

    def __str__(self) -> str:
        """Returns a Description of frequency coverage"""
        intro = "Spectrogram:"
        frequency_coverage = (
            f"{self.dataset.dims['frequency']} harmonics, {self.frequency_increment}Hz spaced \n"
            f" from {self.dataset.frequency.data[0]} to {self.dataset.frequency.data[-1]} Hz."
        )
        time_coverage = f"\n{self.dataset.dims['time']} Time observations"
        time_coverage = f"{time_coverage} \nStart: {self.dataset.time.data[0]}"
        time_coverage = f"{time_coverage} \nEnd:   {self.dataset.time.data[-1]}"

        channel_coverage = list(self.dataset.data_vars.keys())
        channel_coverage = "\n".join(channel_coverage)
        channel_coverage = f"\nChannels present: \n{channel_coverage}"
        return (
            intro
            + "\n"
            + frequency_coverage
            + "\n"
            + time_coverage
            + "\n"
            + channel_coverage
        )

    def __repr__(self):
        return self.__str__()

    @property
    def dataset(self):
        """returns the underlying xarray data"""
        return self._dataset

    @property
    def time_axis(self):
        """returns the time axis of the underlying xarray"""
        return self.dataset.time

    @property
    def frequency_increment(self):
        """
        returns the "delta f" of the frequency axis
        - assumes uniformly sampled in frequency domain
        """
        if self._frequency_increment is None:
            frequency_axis = self.dataset.frequency
            self._frequency_increment = frequency_axis.data[1] - frequency_axis.data[0]
        return self._frequency_increment

    def num_harmonics_in_band(self, frequency_band, epsilon=1e-7):
        """

        Returns the number of harmonics within the frequency band in the underlying dataset

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
        Returns another instance of Spectrogram, with the frequency axis reduced to the input band.

        TODO: Consider returning a copy of the data...

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
            frequency_band,
            self.dataset,
            channels=channels,
            epsilon=self.frequency_increment / 2.0,
        )
        spectrogram = Spectrogram(dataset=extracted_band_dataset)
        return spectrogram

    # TODO: Add cross power method
    # def cross_powers(self, ch1, ch2, band=None):
    #     pass

    def flatten(self, chunk_by: Optional[str] = "time") -> xarray.Dataset:
        """

        Returns the flattened xarray (time-chunked by default).

        Parameters
        ----------
        chunk_by: str
            Controlled vocabulary ["time", "frequency"]. Reshaping the 2D spectrogram can be done two ways
            (basically "row-major", or column-major). In xarray, but we either keep frequency constant and
            iterate over time, or keep time constant and iterate over frequency (in the inner loop).


        Returns
        -------
        xarray.Dataset : The dataset from the band spectrogram, stacked.

        Development Notes:
        The flattening used in tf calculation by default is opposite to here
        dataset.stack(observation=("frequency", "time"))
        However, for feature extraction, it may make sense to swap the order:
        xrds = band_spectrogram.dataset.stack(observation=("time", "frequency"))
        This is like chunking into time windows and allows individual features to be computed on each time window -- if desired.
        Still need to split the time series though--Splitting to time would be a reshape by (last_freq_index-first_freq_index).
        Using pure xarray this may not matter but if we drop down into numpy it could be useful.


        """
        if chunk_by == "time":
            observation = ("time", "frequency")
        elif chunk_by == "frequency":
            observation = ("frequency", "time")
        return self.dataset.stack(observation=observation)
