import numpy as np
import pandas as pd

# from aurora.sandbox.io_helpers.emtf_band_setup import EMTFBandSetupFile


class FrequencyBand(pd.Interval):
    """
    Extends the interval class.

    has a lower_bound, upper_bound, central_frequency and method for Fourier
    coefficient indices

    Some thoughts 20210617:

    TLDR:
    For simplicity, I'm going with Half open, df/2 intervals when they are
    prscribed by FC indexes, and half_open gates_and_fenceposts when they
    are not.  The gates and fenceposts can be converted to the precribed form by
    mapping to emtf_band_setup_form and then mapping to FCIndex form.
    A 3dB point correction etc maybe done in a later version.

    ON DEFAULT FREQUENCY BAND CONFIGURATIONS
    Because these are Interval()s there is a little complication:
    If we use closed intervals we can have an issue with the same Fourier
    coefficient being in more than one band [a,b],[b,c] if b corresponds to a harmonic.
    Honestly this is not a really big deal, but it feels sloppy. The default
    behaviour should partition the frequency axis, not break it into sets with
    non-zero overlap, even though the overlapping sets are of measure zero
    analytically, in digital land this matters.

    On the other hand, it is common enough (at low frequency) to have bands
    which are only 1 Harmonic wide, and if we dont use closed intervals we
    can wind up with intervals like [a,a), which is the empty set.

    The best solution I can think of for now incorporates the fact that the
    harmonic frequencies we are going to interact with in digital processing
    will be a discrete collection separated by df.

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




    """

    def __init__(self, left, right, closed="left", **kwargs):
        pd.Interval.__init__(self, left, right, **kwargs)
        self.lower_bound = self.left
        self.upper_bound = self.right

    def lower_closed(self):
        return self.closed_left

    def upper_closed(self):
        return self.closed_right

    def fourier_coefficient_indices(self, frequencies):
        """

        Parameters
        ----------
        frequencies: numpy array
            Intended to represent the one-sided (positive) frequency axis of
            the data that has been FFT-ed

        Returns
        -------
        indices: numpy array of integers
            Integer indices of the fourier coefficients associated with the
            frequecies passed as input argument
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
        """

        Parameters
        ----------
        frequencies: array-like, floating poirt

        Returns: numpy array
            the actual harmonics or frequencies in band, rather than the indices.
        -------

        """
        indices = self.fourier_coefficient_indices(frequencies)
        harmonics = frequencies[indices]
        return harmonics

    @property
    def center_frequency(self):
        """
        ToDo: add an entry to the processing config metadata that allows user to
        specify if geometric or arithmetic mean is to be used for center frequency

        Returns
        -------
        center_frequency: float
            The frequency associated with the band center.
        """
        # return (self.lower_bound + self.upper_bound)/2
        return np.sqrt(self.lower_bound * self.upper_bound)

    @property
    def center_period(self):
        return 1.0 / self.center_frequency


class FrequencyBands(object):
    """
    This is just collection of frequency bands objects.
    If there was no decimation, this would basically be the BandAveragingScheme

    Context: A band is an Interval().
    FrequencyBands can be represented as an IntervalSet()

    The core underlying variable is "band_edges", which is a 2D array, with one row
    per frequency band and two columns, one for the left-hand (lower bound) of the
    frequency band and one for the right-hand (upper bound).

    Using a single "band_edges" array of fenceposts is not a general solution. There
    is no reason to force the bands to be adjacent.  Therefore, will stop supporting
    band_edges 1-D array.  band_edges will need to be a 2d array.  n_bands, 2

    20210720: In fact, it would be more general, and explicit to pass a
    vector of lower_bound_edges and upper_bound_edges.  Also, there should be
    other ways to populate this class, for example by passing it frequency_band
    objects, self.append(frequency_band).  However, that opens the door for
    overlapping frequency bands which in general are not well ordered without
    choosing an order operation (band center would be a good one - but caution here,
    band centers may not be unique!)  However, that should be extremely rare.  The
    only issue with an append method is that after we append a band, we will need to
    perform an ordering/sorting on individual band edges.

    To add complexity, another use case maybe to support bands with gaps at certain
    harmonics, the way this is currently set up, all harmonics between the lower_bound
    and the upper bounds are considered to be part of the band, but if we
    have a known "bad harmonic" or several noisy harmoincs within a band
    there is currently no way using somple FrequencyBand objects that we can
    excise these harmonics.  We could add a "ignore" list of either
    frequencies, bands or harmonics, but that is for later.
    """

    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        kwargs
        band_edges: 2d numpy array
        """
        self.gates = None
        self.band_edges = kwargs.get("band_edges", None)

    @property
    def number_of_bands(self):
        return self.band_edges.shape[0]

    def validate(self):
        """
        placeholder for sanity checks.
        Main reason this is here is in anticipation of supporting an append()
        method to this class that accepts FrequencyBand objects.  In that
        case we would like to re-order the band edges


        """
        band_centers = self.band_centers()

        # check band centers are monotonically increasing
        monotone_condition = np.all(band_centers[1:] > band_centers[:-1])
        if monotone_condition:
            pass
        else:
            print(
                "Band Centers are Not Monotonic.  This probably means that "
                "the bands are being defined in an adhoc / on the fly way"
            )
            print("This condition untested 20210720")
            print("Attempting to reorganize bands")
            # use np.argsort to rorganize the bands
            self.band_edges = self.band_edges[np.argsort(band_centers), :]

        # check other conditions?:

        return

    def bands(self, direction="increasing_frequency"):
        """
        make this a generator for iteration over bands
        Returns
        -------

        """
        band_indices = range(self.number_of_bands)
        if direction == "increasing_period":
            band_indices = np.flip(band_indices)
        return (self.band(i_band) for i_band in band_indices)
        # raise NotImplementedError

    def band(self, i_band):
        """
        ToDO: Decide whether to index bands from zero or one, i.e.  Choosing 0 for now.

        Parameters
        ----------
        i_band: integer key for band

        Returns
        -------
        frequency_band: FrequencyBand() object
        """
        frequency_band = FrequencyBand(
            self.band_edges[i_band, 0],
            self.band_edges[i_band, 1],
        )

        return frequency_band

    def band_centers(self, frequency_or_period="frequency"):
        """
        Parameters
        ----------
        frequency_or_period : str
        "frequency" or "period" determines if the vector of band centers is
        returned in "Hz" or "s"

        Returns
        -------
        band_centers : numpy array
            center frequencies of the bands in Hz or in s

        """
        band_centers = np.full(self.number_of_bands, np.nan)
        for i_band in range(self.number_of_bands):
            frequency_band = self.band(i_band)
            band_centers[i_band] = frequency_band.center_frequency
        if frequency_or_period == "period":
            band_centers = 1.0 / band_centers
        return band_centers

    def from_decimation_object(self, decimation_object):
        """
        Convert band edges from a :class:`aurora.config.Decimation` object,
        Which has all the information in it.

        :param decimation_object: DESCRIPTION
        :type decimation_object: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        # replace below with decimation_object.delta_frequency ?
        df = (
            decimation_object.decimation.sample_rate
            / decimation_object.window.num_samples
        )
        half_df = df / 2.0
        # half_df /=100
        lower_edges = (decimation_object.lower_bounds * df) - half_df
        upper_edges = (decimation_object.upper_bounds * df) + half_df
        band_edges = np.vstack((lower_edges, upper_edges)).T
        self.band_edges = band_edges
