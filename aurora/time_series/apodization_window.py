"""
@author: kkappler

Module to manage windowing prior to FFT.  Intended to support most
apodization  windows available via scipy.signal.get_window()


    Supported Window types = ['boxcar', 'triang', 'blackman', 'hamming', 'hann',
      'bartlett', 'flattop', 'parzen', 'bohman', 'blackmanharris',
      'nuttall', 'barthann', 'kaiser', 'gaussian', 'general_gaussian',
      'slepian', 'chebwin']

    have_additional_args = {
      'kaiser' : 'beta',
      'gaussian' : 'std',
      'general_gaussian' : ('power', 'width'),
      'slepian' : 'width',
      'chebwin' : 'attenuation',
    }

The Taper Config has 2 possible forms:
1. Standard form for accessing scipy.signal:
["taper_family", "num_samples_window", "additional_args"]
2. User-defined : for defining custom tapers

Example 1 : Standard form
"taper_family" = "hamming"
"num_samples_window" = 128
"additional_args" = {}

Example 2 : Standard form
"taper_family" = "kaiser"
"num_samples_window" = 64
"additional_args" = {"beta":8}

Examples 3 : User Defined
2. user-defined: ["array"]
In this case num_samples_window is defined by the array.
"array" = [1, 2, 3, 4, 5, 4, 3, 2, 1]
If "array" is non-empty then assume the user-defined case.

It is a little bit unsatisfying that the args need to be ordered for
scipy.signal.get_window().  Probably use OrderedDict()
for any windows that have more than one additional args.

For example
"taper_family" = 'general_gaussian'
"additional_args" = OrderedDict("power":1.5, "sigma":7)

"""


import numpy as np
import scipy.signal as ssig


class ApodizationWindow(object):
    """
    Instantiate an apodization window object.  Example usages:
    apod_window = ApodizationWindow()
    taper=ApodizationWindow(taper_family='hanning', num_samples_window=55 )

    Window factors S1, S2, CG, ENBW are modelled after Heinzel et al. p12-14
    [1] Spectrum and spectral density estimation by the Discrete Fourier transform
    (DFT), including a comprehensive list of window functions and some new
    flat-top windows.  G. Heinzel, A. Roudiger and R. Schilling, Max-Planck
    Institut fur Gravitationsphysik (Albert-Einstein-Institut)
    Teilinstitut Hannover February 15, 2002
    See Also
    [2] Harris FJ. On the use of windows for harmonic analysis with the discrete
    Fourier transform. Proceedings of the IEEE. 1978 Jan;66(1):51-83.

    Nomenclature from Heinzel et al.
    ENBW: Effective Noise BandWidth, see Equation (22)
    NENBW Normalized Equivalent Noise BandWidth, see Equation (21)

    Parameters
    ----------
    taper_family : string
        Specify the taper type - boxcar, kaiser, hanning, etc
    num_samples_window : int
        The number of samples in the taper
    taper : numpy array
        The actual window coefficients themselves.  This can be passed if a
        particular custom window is desired.
    additional_args: dictionary
        These are any additional requirements scipy needs in order to
        generate the window.
    """

    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        kwargs: dict
            See parameters list in class level documentation above
        """
        self.taper_family = kwargs.get("taper_family", "boxcar")
        self._num_samples_window = kwargs.get("num_samples_window", 0)
        self._taper = kwargs.get("taper", np.empty(0))
        self.additional_args = kwargs.get("taper_additional_args", {})

        self._coherent_gain = None
        self._nenbw = None
        self._S1 = None
        self._S2 = None
        self._apodization_factor = None

        if self.taper.size == 0:
            self.make()

    @property
    def summary(self):
        """

        Returns
        -------
        out_str: str
            String comprised of the taper_family, number_of_samples, and True/False
            if self.taper is not None
        """
        self.test_linear_spectral_density_factor()
        string1 = f"{self.taper_family} {self.num_samples_window}"
        string1 += f" taper_exists = {bool(self.taper.any())}"
        string2 = f"NENBW = {self.nenbw:.3f}, CG = {self.coherent_gain:.3f},  "
        string2 += f"window factor = {self.apodization_factor:.3f}"
        out_str = "\n".join([string1, string2])
        return out_str

    def __str__(self):
        """
        ? __repr__?
        """
        return f"{self.taper}"

    @property
    def num_samples_window(self):
        if self._num_samples_window == 0:
            self._num_samples_window = len(self.taper)
        return self._num_samples_window

    def make(self):
        """
        this is just a wrapper call to scipy.signal
        Note: see scipy.signal.get_window for a description of what is
        expected in args[1:]. http://docs.scipy.org/doc/scipy/reference/
        generated/scipy.signal.get_window.html

        note: this is just repackaging the args so that scipy.signal.get_window()
        accepts all cases.
        """
        window_args = [v for k, v in self.additional_args.items()]
        window_args.insert(0, self.taper_family)
        window_args = tuple(window_args)
        # print(f"\n\nWINDOW args {window_args}")
        self.taper = ssig.get_window(window_args, self.num_samples_window)
        self.apodization_factor  # calculate
        return

    @property
    def S1(self):
        """sum of the window coefficients"""
        if getattr(self, "_S1", None) is None:
            self._S1 = sum(self.taper)
        return self._S1

    @property
    def S2(self):
        """sum of squares of the window coefficients"""
        if getattr(self, "_S2", None) is None:
            self._S2 = sum(self.taper**2)
        return self._S2

    @property
    def coherent_gain(self):
        """DC gain of the window normalized by window length"""
        return self.S1 / self.num_samples_window

    @property
    def nenbw(self):
        """NENBW Normalized Equivalent Noise BandWidth, see Equation (21) in
        Heinzel et al 2002"""
        return self.num_samples_window * self.S2 / (self.S1**2)

    def enbw(self, fs):
        """
        Notes that unlike NENBW, CG, S1, S2, this is not a pure property of the
        window -- but instead this is a property of the window combined with
        the sample rate.
        Parameters
        ----------
        fs : sampling frequency (1/dt)

        Returns
        -------

        """
        """Effective Noise BandWidth = fs*NENBW/N = fs S2/(S1**2)"""

        return fs * self.S2 / (self.S1**2)

    def test_linear_spectral_density_factor(self):
        """This is just a test to verify some algebra
        Claim:
        The lsd_calibration factors
        A      (1./coherent\_gain)\*np.sqrt((2\*dt)/(nenbw\*N))
        and
        B      np.sqrt(2/(sample\_rate\*self.S2))
        are identical.

        Note sqrt(2\*dt)==sqrt(2\*sample_rate) so we can cancel these terms and
        A=B IFF
        (1./coherent\_gain) * np.sqrt(1/(nenbw\*N)) == 1/np.sqrt(S2)
        which I show in githib aurora issue #3 via .
        (CG\*\*2) \* NENBW \*N   =  S2

        
        """
        lsd_factor1 = (1.0 / self.coherent_gain) * np.sqrt(
            1.0 / (self.nenbw * self.num_samples_window)
        )
        lsd_factor2 = 1.0 / np.sqrt(self.S2)
        if not np.isclose(lsd_factor1, lsd_factor2):
            print(f"factor1 {lsd_factor1} vs factor2 {lsd_factor2}")
            print("Incompatible spectral density factors")
            raise Exception

    @property
    def taper(self):
        return self._taper

    @taper.setter
    def taper(self, x):
        self._taper = x
        self._S1 = None
        self._S2 = None

    @property
    def apodization_factor(self):
        if self._apodization_factor is None:
            self._apodization_factor = np.sqrt(self.nenbw) * self.coherent_gain
        return self._apodization_factor
