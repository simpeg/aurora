# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 14:11:24 2022

@author: jpeacock
"""

import unittest
from mt_metadata.transfer_functions.processing.aurora import Regression


class TestRegression(unittest.TestCase):
    """
    Test Station metadata
    """

    def setUp(self):
        self.regression = Regression()

    def test_initialization(self):
        for key in self.regression.get_attribute_list():
            with self.subTest(key):
                self.assertEqual(
                    self.regression.get_attr_from_name(key),
                    self.regression._attr_dict[key]["default"],
                )


if __name__ == "__main__":
    unittest.main()
