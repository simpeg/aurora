# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 14:11:24 2022

@author: jpeacock
"""

import unittest
from mt_metadata.transfer_functions.processing.aurora import Decimation


class TestDecimation(unittest.TestCase):
    """
    Test Station metadata
    """

    def setUp(self):
        self.decimation = Decimation()

    def test_initialization(self):
        for key in self.decimation.get_attribute_list():
            with self.subTest(key):
                self.assertEqual(
                    self.decimation.get_attr_from_name(key),
                    self.decimation._attr_dict[key]["default"],
                )


if __name__ == "__main__":
    unittest.main()
