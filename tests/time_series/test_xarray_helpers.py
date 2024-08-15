# -*- coding: utf-8 -*-
"""
"""

import unittest
from aurora.time_series.xarray_helpers import initialize_xrda_1d


class TestXarrayHelpers(unittest.TestCase):
    """
    Test methods in xarray helpers
    - may get broken into separate tests if this module grows
    """

    @classmethod
    def setUpClass(self):
        pass

    def setUp(self):
        pass

    def test_initialize_xrda_1d(self):
        channels = ["ex", "ey", "hx", "hy", "hz"]
        dtype = float
        value = -1
        tmp = initialize_xrda_1d(channels, dtype=dtype, value=value)
        self.assertTrue((tmp.data == value).all())

    def test_sometehing_else(self):
        """
        Place holder

        """
        pass


if __name__ == "__main__":
    unittest.main()
