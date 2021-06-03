"""
Based loosely on TaperModule() concept developed by kkappler in 2012, this is
a leaner version intended to support most apodization windows available via
scipy.signal.get_window()


    Supported Window types = ['boxcar', 'triang', 'blackman', 'hamming', 'hann',
      'bartlett', 'flattop', 'parzen', 'bohman', 'blackmanharris',
      'nuttall', 'barthann', 'kaiser', 'gaussian', 'general_gaussian',
      'slepian', 'chebwin']

    have_additional_args = {
      'kaiser' : 'beta',
      'gaussian' : 'std',
      'general_gaussian' : ('power', 'width'),
      'slepian' : 'width',
      'chebwin' : 'attenuation'
    }

The Taper Config has 2 possible forms:
1. Standard form: ["taper_family", "num_samples_window", "additional_args"]

Example 1
"taper_family" = "hamming"
"num_samples_window" = 128
"additional_args" = {}

Example 2
"taper_family" = "kaiser"
"num_samples_window" = 64
"additional_args" = {"beta":8}

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
    usage: apod_window = ApodizationWindow()
    @type taper_family: string
    @ivar taper_family: Specify the taper type - boxcar, kaiser, hanning, etc
    @type num_samples_window: Integer
    @ivar num_samples_window: The number of samples in the taper
    @type taper: numpy array
    @ivar taper: The actual taper window itself
    @type coherentGain: float
    @ivar coherentGain:
    @type NENBW: float
    @ivar NENBW: normalized equivalent noise bandwidth
    @type S1: float
    @ivar S1: window sum
    @type S2: float
    @ivar S2: sum of squares of taper elements

    @author: kkappler
    @note: example usage:
        tpr=ApodizationWindow(taper_family='hanning', num_samples_window=55 )

    Window factors S1, S2, CG, ENBW are modelled after Heinzel et al. p12-14
    [1] Spectrum and spectral density estimation by the Discrete Fourier transform
    (DFT), including a comprehensive list of window functions and some new
    flat-top windows.  G. Heinzel, A. Roudiger and R. Schilling, Max-Planck
    Institut fur Gravitationsphysik (Albert-Einstein-Institut)
    Teilinstitut Hannover February 15, 2002
    See Also
    [2] Harris FJ. On the use of windows for harmonic analysis with the discrete
    Fourier transform. Proceedings of the IEEE. 1978 Jan;66(1):51-83.


    Instantiate an apodization window object.


        Parameters
        ----------
        kwargs:
        taper_family
        num_samples_window
        taper
        additional_args
    """
    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        kwargs:
        taper_family
        num_samples_window
        taper
        additional_args
        """
        self.taper_family = kwargs.get('taper_family', 'boxcar')
        self._num_samples_window = kwargs.get('num_samples_window', 0)
        self._taper = kwargs.get('taper', np.empty(0))
        self.additional_args = kwargs.get('additional_args', {})

        self._coherent_gain = None
        self._nenbw = None
        self._S1 = None
        self._S2 = None
        self._apodization_factor = None

        if self.taper.size==0:
            self.make()

    @property
    def summary(self):
        """
        Returns a string comprised of the taper_family, number_of_samples, 
        and True/False if self.taper is not None

        Note: ? cal this __str__()
        -------

        """
        string1 = f"{self.taper_family} {self.num_samples_window}"
        string1 += f" taper_exists={bool(self.taper.any())}"
        string2 = f"NENBW:{self.nenbw:.3f}, CG:{self.coherent_gain:.3f}"
        string2 += f"window factor={self.apodization_factor:.3f}"
        return "\n".join([string1, string2])

    def __str__(self):
        """
        ? __repr__?
        """
        return f"{self.taper}"

    @property
    def num_samples_window(self):
        if self._num_samples_window==0:
            self._num_samples_window = len(self.taper)
        return self._num_samples_window

    def make(self):
        """
        this is just a wrapper call to scipy.signal
        Note: see scipy.signal.get_window for a description of what is
        expected in args[1:]. http://docs.scipy.org/doc/scipy/reference/
        generated/scipy.signal.get_window.html

        note: this is just repackaging the args so that scipy.signal.get_window() accepts all cases.
        """
        window_args = [v for k,v in self.additional_args.items()]
        window_args.insert(0, self.taper_family)
        window_args = tuple(window_args)
        #print(f"\n\nWINDOW args {window_args}")
        self.taper = ssig.get_window(window_args, self.num_samples_window)
        self.apodization_factor#calculate
        return



    @property
    def S1(self):
        if getattr(self, "_S1", None) is None:
            self._S1 = sum(self.taper)
        return self._S1

    @property
    def S2(self):
        if getattr(self, "_S2", None) is None:
            self._S2 = sum(self.taper**2)
        return self._S2

    @property
    def coherent_gain(self):
        return self.S1 / self.num_samples_window

    @property
    def nenbw(self):
        return self.num_samples_window * self.S2 / (self.S1 ** 2)

    @property
    def taper(self):
        return self._taper

    @taper.setter
    def taper(self, x):
        self._taper=x
        self._S1 = None
        self._S2 = None


    @property
    def apodization_factor(self):
        if self._apodization_factor is None:
            self._apodization_factor = np.sqrt(self.nenbw) * self.coherent_gain
        return self._apodization_factor




def test_can_inititalize_apodization_window():
    """
    """
    apodization_window = ApodizationWindow(num_samples_window=4)
    print(apodization_window.summary)
    apodization_window = ApodizationWindow(taper_family='hamming', num_samples_window=128)
    print(apodization_window.summary)
    apodization_window = ApodizationWindow(taper_family='blackmanharris', num_samples_window=256)
    print(apodization_window.summary)
    apodization_window = ApodizationWindow(taper_family='kaiser', num_samples_window=128, additional_args={"beta":8})
    print(apodization_window.summary)
    apodization_window = ApodizationWindow(taper_family='slepian', num_samples_window=64, additional_args={"width":0.3})
    print(apodization_window.summary)
    apodization_window = ApodizationWindow(taper_family='custom', num_samples_window=64,
                                           taper=np.abs(np.random.randn(64)))
    print(apodization_window.summary)



def main():
    """
    """
    test_can_inititalize_apodization_window()
    print("fin")

if __name__ == "__main__":
    main()
