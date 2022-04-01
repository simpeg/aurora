# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 14:11:24 2022

@author: jpeacock
"""

import unittest
from aurora.config import Station, Channel, Stations


class TestStation(unittest.TestCase):
    """
    Test Station metadata
    """

    def setUp(self):
        self.stations = Stations()

    def test_initialization(self):
        with self.subTest("test local"):
            self.assertIsInstance(self.stations.local, Station)
        # with self.subTest("test remote type"):
        #     self.assertIsInstance(self.stations.remote, dict)
        # with self.subTest("test remote dict"):
        #     self.assertEqual(self.stations.remote, {})

    # def test_add_remote_station(self):
    #     rr = Station(id="rr01")
    #     self.stations.remote = rr
    #     with self.subTest("is dict"):
    #         self.assertIsInstance(self.stations.remote, dict)
    #     with self.subTest("has keys"):
    #         self.assertListEqual(["rr01"], list(self.stations.remote.keys()))

    # def test_add_dict(self):
    #     rr_dict = {"rr01": Station(id="rr01")}
    #     self.stations.remote = rr_dict
    #     with self.subTest("is dict"):
    #         self.assertIsInstance(self.stations.remote, dict)
    #     with self.subTest("has keys"):
    #         self.assertListEqual(["rr01"], list(self.stations.remote.keys()))


if __name__ == "__main__":
    unittest.main()
