# -*- coding: utf-8 -*-
"""
"""
from loguru import logger
import numpy as np
import unittest

from aurora.time_series.apodization_window import ApodizationWindow


class TestApodizationWindow(unittest.TestCase):
    """
    Test ApodizationWindow
    """

    def setUp(self):
        pass

    #        self.band = Band()

    def test_default_boxcar(self):
        window = ApodizationWindow(num_samples_window=4)
        assert window.nenbw == 1.0
        assert window.coherent_gain == 1.0
        assert window.apodization_factor == 1.0
        logger.info(window.summary)

    def test_hamming(self):
        window = ApodizationWindow(taper_family="hamming", num_samples_window=128)
        assert np.isclose(window.nenbw, 1.362825788751716)
        assert np.isclose(window.coherent_gain, 0.54)
        assert np.isclose(window.apodization_factor, 0.6303967004989797)
        logger.info(window.summary)

    def test_blackmanharris(self):
        window = ApodizationWindow(
            taper_family="blackmanharris", num_samples_window=256
        )
        assert np.isclose(window.nenbw, 2.0043529382170493)
        assert np.isclose(window.coherent_gain, 0.35874999999999996)
        assert np.isclose(window.apodization_factor, 0.5079009302511663)
        logger.info(window.summary)

    def test_kaiser(self):
        apodization_window = ApodizationWindow(
            taper_family="kaiser",
            num_samples_window=128,
            taper_additional_args={"beta": 8},
        )
        logger.info(apodization_window.summary)

    def test_tukey(self):
        apodization_window = ApodizationWindow(
            taper_family="tukey",
            num_samples_window=30000,
            taper_additional_args={"alpha": 0.25},
        )

        logger.info(apodization_window.summary)

    def test_dpss(self):
        """ """
        apodization_window = ApodizationWindow(
            taper_family="dpss",
            num_samples_window=64,
            taper_additional_args={"NW": 3.0},
        )
        logger.info(apodization_window.summary)

    def test_custom(self):
        apodization_window = ApodizationWindow(
            taper_family="custom",
            num_samples_window=64,
            taper=np.abs(np.random.randn(64)),
        )
        logger.info(apodization_window.summary)


if __name__ == "__main__":
    # taw = TestApodizationWindow()
    # taw.test_blackmanharris()
    unittest.main()
