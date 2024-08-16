# -*- coding: utf-8 -*-
"""
This module contains unittests for the xarray_helpers module.
"""

import numpy as np
import unittest

import xarray as xr

from aurora.time_series.xarray_helpers import covariance_xr
from aurora.time_series.xarray_helpers import initialize_xrda_1d
from aurora.time_series.xarray_helpers import initialize_xrda_2d


class TestXarrayHelpers(unittest.TestCase):
    """
    Test methods in xarray helpers
    - may get broken into separate tests if this module grows
    """

    @classmethod
    def setUpClass(self):
        self.standard_channel_names = ["ex", "ey", "hx", "hy", "hz"]

    def setUp(self):
        pass

    def test_initialize_xrda_1d(self):
        dtype = float
        value = -1
        tmp = initialize_xrda_1d(self.standard_channel_names, dtype=dtype, value=value)
        self.assertTrue((tmp.data == value).all())

    def test_initialize_xrda_2d(self):
        dtype = float
        value = -1
        tmp = initialize_xrda_2d(self.standard_channel_names, dtype=dtype, value=value)
        self.assertTrue((tmp.data == value).all())

    def test_covariance_xr(self):
        np.random.seed(0)
        n_observations = 100
        xrds = xr.Dataset(
            {
                "hx": (
                    [
                        "time",
                    ],
                    np.abs(np.random.randn(n_observations)),
                ),
                "hy": (
                    [
                        "time",
                    ],
                    np.abs(np.random.randn(n_observations)),
                ),
            },
            coords={
                "time": np.arange(n_observations),
            },
        )

        X = xrds.to_array()
        cov = covariance_xr(X)
        self.assertTrue((cov.data == cov.data.transpose().conj()).all())

    def test_sometehing_else(self):
        """
        Place holder

        """
        pass


if __name__ == "__main__":
    unittest.main()
