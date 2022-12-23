# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 14:31:37 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest
from pathlib import Path


from aurora import EstimateTransferFunction
from aurora.config import BANDS_DEFAULT_FILE

# =============================================================================


class TestEstimateTransferFunction(unittest.TestCase):
    def setUp(self):
        self.etf = EstimateTransferFunction()

    def test_set_local_path(self):
        self.etf.local_mth5_path = "/home/path"

        self.assertIsInstance(self.etf.local_mth5_path, Path)

    def test_set_local_path_none(self):
        self.etf.local_mth5_path = None
        self.assertIsNone(self.etf.local_mth5_path)

    def test_set_local_path_fail(self):
        def set_path(value):
            self.etf.local_mth5_path = value

        self.assertRaises(TypeError, set_path, 10)

    def test_set_remote_path(self):
        self.etf.remote_mth5_path = "/home/path"

        self.assertIsInstance(self.etf.remote_mth5_path, Path)

    def test_set_remote_path_none(self):
        self.etf.remote_mth5_path = None
        self.assertIsNone(self.etf.remote_mth5_path)

    def test_set_remote_path_fail(self):
        def set_path(value):
            self.etf.remote_mth5_path = value

        self.assertRaises(TypeError, set_path, 10)

    def test_set_bands_path_fail(self):
        def set_path(value):
            self.etf.bands_file_path = value

        self.assertRaises(IOError, set_path, "/home/path")

    def test_set_bands_path_none(self):
        self.etf.bands_file_path = None
        self.assertEqual(Path(BANDS_DEFAULT_FILE), self.etf.bands_file_path)

    def test_run_summary_fail(self):
        def get_run_summary():
            return self.etf.run_summary

        self.assertRaises(ValueError, get_run_summary)

    def test_kernel_dataset_fail(self):
        def get_kernel_dataset():
            return self.etf.kernel_dataset

        self.assertRaises(ValueError, get_kernel_dataset)

    def test_get_station_runs_dict(self):
        self.etf.local_station = "mt001"
        with self.subTest("list input"):
            rd = self.etf._get_station_runs_dict(self.etf.local_station, ["a"])
            self.assertDictEqual({"mt001": ["a"]}, rd)

        with self.subTest("string input"):
            rd = self.etf._get_station_runs_dict(self.etf.local_station, "a")
            self.assertDictEqual({"mt001": ["a"]}, rd)

        with self.subTest("fail"):
            self.assertRaises(
                TypeError,
                self.etf._get_station_runs_dict,
                self.etf.local_station,
                10,
            )


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
