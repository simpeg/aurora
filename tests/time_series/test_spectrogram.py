# -*- coding: utf-8 -*-
"""
"""

import unittest

from aurora.time_series.spectrogram import Spectrogram


class TestSpectrogram(unittest.TestCase):
    """
    Test Spectrogram class
    """

    @classmethod
    def setUpClass(self):
        pass

    def setUp(self):
        pass

    def test_initialize(self):
        spectrogram = Spectrogram()
        assert isinstance(spectrogram, Spectrogram)


if __name__ == "__main__":
    # tmp = TestSpectrogram()
    # tmp.test_initialize()
    unittest.main()
