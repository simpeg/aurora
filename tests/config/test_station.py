# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 14:11:24 2022

@author: jpeacock
"""

import unittest
from mt_metadata.transfer_functions.processing.aurora import Station, Channel


class TestStation(unittest.TestCase):
    """
    Test Station metadata
    """

    def setUp(self):
        self.station = Station()

    def test_initialization(self):
        with self.subTest("test id"):
            self.assertEqual(None, self.station.id)
        with self.subTest("test mth5_path"):
            self.assertEqual(None, self.station.mth5_path)
        with self.subTest("test id"):
            self.assertEqual(False, self.station.remote)
        # with self.subTest("test channel_scale_factors"):
        #     self.assertEqual({}, self.station.channel_scale_factors)

    # def test_add_channel(self):
    #     ch = Channel(id="ex", scale_factor=10)
    #     self.station.channel_scale_factors = ch
    #     with self.subTest("is dict"):
    #         self.assertIsInstance(self.station.channel_scale_factors, dict)

    # def test_add_dict(self):
    #     ch_dict = {"ex": Channel(id="ex")}
    #     self.station.channel_scale_factors = ch_dict
    #     with self.subTest("is dict"):
    #         self.assertIsInstance(self.station.channel_scale_factors, dict)


if __name__ == "__main__":
    unittest.main()
