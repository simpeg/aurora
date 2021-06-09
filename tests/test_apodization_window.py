import numpy as np

from aurora.signal.apodization_window import ApodizationWindow

def test_default_boxcar():
    apodization_window = ApodizationWindow(num_samples_window=4)
    print(apodization_window.summary)

def test_hamming():
    apodization_window = ApodizationWindow(taper_family='hamming',
                                           num_samples_window=128)
    print(apodization_window.summary)

def test_blackmanharris():
    apodization_window = ApodizationWindow(taper_family='blackmanharris',
                                           num_samples_window=256)
    print(apodization_window.summary)

def test_kaiser():
    apodization_window = ApodizationWindow(taper_family='kaiser',
                                           num_samples_window=128,
                                           additional_args={"beta": 8})
    print(apodization_window.summary)

def test_tukey():
    apodization_window = ApodizationWindow(taper_family='tukey',
                                           num_samples_window=30000,
                                           additional_args={"alpha": 0.25})

    print(apodization_window.summary)
def test_dpss():
    #<?>
    # Passes locally but fails on miniconda ci test... not sure why,
    # maybe not yet standard part of scipy.signal?  Get a to-be-deprecated
    # warning when call slepian. dpss is supposed to replace it.
    apodization_window = ApodizationWindow(taper_family='dpss',
                                           num_samples_window=64,
                                           additional_args={"NW":3.0})
    print(apodization_window.summary)
    # </?>

    #<?>
    #sorting out which of slepian or dpss is causing failure.
    #just muted dpss and got a fail.


def test_slepian():
    apodization_window = ApodizationWindow(taper_family='slepian',
                                           num_samples_window=64,
                                           additional_args={"width": 0.3})
    print(apodization_window.summary)

def test_custom():
    apodization_window = ApodizationWindow(taper_family='custom',
                                           num_samples_window=64,
                                           taper=np.abs(np.random.randn(64)))
    print(apodization_window.summary)

def test_can_inititalize_apodization_window():
    """
    """
    pass

def main():
    test_default_boxcar()
    test_hamming()
    test_blackmanharris()
    test_kaiser()
    test_slepian()
    test_dpss()
    test_custom()
    test_tukey()
#    test_can_inititalize_apodization_window()

if __name__ == "__main__":
    main()
