from collections import OrderedDict
import numpy as np

from aurora.interval import Interval
from aurora.sandbox.io_helpers.emtf_band_setup import EMTFBandSetupFile


class FrequencyBand(Interval):
    """
    Extends the interval class.

    has a lower bound, an upper bound and a central frequency

    These are intervals
    an method for Fourier coefficient indices

    Some thoughts 20210617:

    TLDR:
    For simplicity, I'm going with Half open, df/2 intervals when they are
    perscribed by FC indexes, and half_open gates_and_fenceposts when they
    are not.  The gates and fenceposts can be converted to the percribed form by
    mapping to emtf_band_setup_form and then mapping to FCIndex form.
    A 3dB point correction etc maybe done in a later version.

    <ON DEFAULT FREQUENCY BAND CONFIGURATIONS>
    Because these are Interval()s there is a little complication:
    If we use closed intervals we can have an issue with the same Fourier
    coefficient being in more than one band [a,b],[b,c] if b corresponds to a harmonic.
    Honestly this is not a really big deal, but it feels sloppy. The default
    behaviour should partition the frequency axis, not break it into sets with
    non-zero overlap, even though the overlapping sets are of measure zero.

    On the other hand, it is common enough (at low frequency) to have bands
    which are only 1 Harmonic wide, and if we dont use closed intervals we
    can wind up with intervals like [a,a), which is the empty set.

    The best solution I can think of for now incorporates the fact that the
    harmonic frequencies we are going to interact with in digital processing
    will be a discrete collection, basically fftfreqs, which are separated by df.

    If we are given the context of df (= 1/(N*dt)) wher N is number of
    samples in the original time series and dt is the sample interval,
    then we can use half-open intervals with width df centered at the
    frequencies under consideration.

    I.e since the actual picking of Fourier coefficients and indexes will
    always occur in the context of a sampling rate and a frequency axis and
    we will know df and therefore we can pad the frequency bounds by +/-
    df/2.

    In that case we can use open, half open, or closed intervals, it really
    doesn't matter, so we will choose half open
    [f_i-df/2, f_i+df/2) to get the satisfying property of covering the
    frequency axis completely but ensure no accidental double coverage.

    Notes that this is just a default convention.  There is no rule against
    using closed intervals, nor having overlapping bands.

    The df/2 trick also protects us from numeric roundoff errors resulting in
    edge frequencies landing in a bin other than that which is intended.

    There is one little oddity which accompanies this scheme.  Consider the
    case where you have a 1-harmonic wide band, say at 10Hz.  And df for
    arguments sake is 0.05 Hz.  The center frequency harmonically will not
    evaluate to 10Hz exactly, rather it will evaluate to sqrt((
    9.95*10.05))=9.9999874, and not 10.  This is a little bit unsatisfying
    but I take solace in two things:
    1. The user is welcome to use their own convention, [10.0, 10.0], or even
    [10.0-epsilon , 10.0+epsilon] if worried about numeric ghosts which
    asymptotes to 10.0
    2.  I'm not actually 100% sure that the geometric center frequency of the
    harmonic at 10Hz is truly 10.0.  Afterall a band has finite width even if
    the harmonic is a Dirac spike.

    At the end of the day we need to choose something, so its half-open,
    lower-closed intervals by default.

    Here's a good webpage with formulas if we want to get really fancy with
    3dB band edges.
    http://www.sengpielaudio.com/calculator-cutoffFrequencies.htm

    </ON DEFAULT FREQUENCY BAND CONFIGURATIONS>


    """

    def __init__(self, **kwargs):
        Interval.__init__(self, **kwargs)
        if kwargs.get("upper_closed") is None:
            self.upper_closed = False


    def fourier_coefficient_indices(self, frequencies):
        """

        Parameters
        ----------
        frequencies: numpy array
            Intended to represent the one-sided frequency axis of the data
            that has been FFT-ed

        Returns
        -------

        """
        if self.lower_closed:
            cond1 = frequencies >= self.lower_bound
        else:
            cond1 = frequencies > self.lower_bound
        if self.upper_closed:
            cond2 = frequencies <= self.upper_bound
        else:
            cond2 = frequencies < self.upper_bound

        indices = np.where(cond1 & cond2)[0]
        return indices

    def in_band_harmonics(self, frequencies):
        indices = self.fourier_coefficient_indices(frequencies)
        harmonics = frequencies[indices]
        return harmonics

    @property
    def center_frequency(self):
        return np.sqrt(self.lower_bound * self.upper_bound)

    @property
    def center_period(self):
        return 1./self.center_frequency





class FrequencyBands(object):
    """
    Use this as the core element for BandAveragingScheme
    This is just collection of frequency bands objects.

    If there was no decimation, this would basically be the BandAveragingScheme
    How does it differ from a bandaveraging scheme?
    It doesn't support Decimation levels.

    Context: A band is an Interval().
    FrequencyBands can be represented as an IntervalSet()

    20210617: Unforunately, using a single "band_edges" array of fenceposts
    is not a general solution.  There is no reason to force the user to have
    bands that abutt one another.  Therefore, changing to stop supporting
    band_edges 1-D array.  band_edges will need to be a 2d array.  n_bands, 2
    """

    def __init__(self, **kwargs):
        self.gates = None
        self.band_edges = kwargs.get("band_edges", None)
        #self.bands = OrderedDict()
        #frequencies ... can repeat (log spacing)

    @property
    def number_of_bands(self):
        return self.band_edges.shape[0]

    def bands(self):
        """
        make this a generator for iteration over bands
        Returns
        -------

        """
        raise NotImplementedError

    def band(self, i_band):
        """
        Decide to index bands from zero or one, i.e.  Choosing 0 for now.
        Parameters
        ----------
        i_band: integer key for band

        Returns
        -------

        """
        frequency_band = FrequencyBand(lower_bound=self.band_edges[i_band,0],
                                       upper_bound=self.band_edges[i_band,1]
                                       )

        return frequency_band

    def from_emtf_band_setup(self, filepath, decimation_level, sampling_rate,
                             num_samples_window):
        """
        This converts between EMTF band_setup files to a frequency_bands object.
        The band_setup file is represented as a dataframe with
        columns for decimation_level, first_fc_index,
        last_fc_index.

        Notes:
        1. In EMTF, the the DC terms were not carried in the FC Files so the
        integer-index 1 mapped to the first harmonic.  In aurora, the DC-term is
        kept (for now) and thus, because the fortran arrays index from 1, and
        python from 0, we don't need any modification to the Fourier coefficient
        indices here.If we wind up dropping the DC term from the STFT arrays we
        would need to add -1 to the upper and lower bound indices.
        2. EMTF band-setup files do not contain information about frequency.
        Frequecny was implicit in the processing scheme but not stated.  This
        leaves some ambuguity when reading in files with names like
        "bs_256.txt".  Does 256 refer to the number of taps in the STFT
        window, or to the number of positive frequencies (and the window was
        512-length).  Personal communication with Egbert 2021-06-22 indicates
        that normally the integer number in a band-setup file name associates with
        the time-domain window length.


        Parameters
        ----------
        filepath : str or pathlib.Path()
            The full path to the band_setup file
        decimation_level : integer
            Corresponds to the decimation level from the band setup file to
            create FrequecyBands from.
        sample_rate : float
            The sampling rate of the data at decimation_level

        Returns
        -------

        """
        emtf_band_setup = EMTFBandSetupFile(filepath=filepath)
        emtf_band_df = emtf_band_setup.get_decimation_level(decimation_level)
        df = sampling_rate / (num_samples_window)
        half_df = df / 2.0

        lower_edges = (emtf_band_df.lower_bound_index * df) - half_df
        upper_edges = (emtf_band_df.upper_bound_index * df) + half_df
        band_edges = np.vstack((lower_edges.values, upper_edges.values)).T
        self.band_edges = band_edges

        return

