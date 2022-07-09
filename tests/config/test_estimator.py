# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 14:11:24 2022

@author: jpeacock
"""

import unittest
from aurora.config import Estimator


class TestEstimator(unittest.TestCase):
    """
    Test Station metadata
    """

    def setUp(self):
        self.estimator = Estimator()

    def test_initialization(self):
        for key in self.estimator.get_attribute_list():
            with self.subTest(key):
                self.assertEqual(
                    self.estimator.get_attr_from_name(key),
                    self.estimator._attr_dict[key]["default"],
                )


if __name__ == "__main__":
    unittest.main()
