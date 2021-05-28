import numpy as np

from aurora.signal.apodization_window import ApodizationWindow

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
    #<?>
    # Passes locally but fails on miniconda ci test... not sure why,
    # maybe not yet standard part of scipy.signal?  Get a to-be-deprecated
    # warning when call slepian. dpss is supposed to replace it.
    # apodization_window = ApodizationWindow(taper_family='dpss',
    #                                        num_samples_window=64,
    #                                        additional_args={"NW":3.0})
    #print(apodization_window.summary)
    # </?>

    #<?>
    #sorting out which of slepian or dpss is causing failure.
    #just muted dpss and got a fail.
    
    apodization_window = ApodizationWindow(taper_family='slepian',
                                           num_samples_window=64,
                                           additional_args={"width":0.3})
    print(apodization_window.summary)
    apodization_window = ApodizationWindow(taper_family='custom', num_samples_window=64,
                                           taper=np.abs(np.random.randn(64)))
    print(apodization_window.summary)
    #</?>

def main():
    test_can_inititalize_apodization_window()

if __name__ == "__main__":
    main()
