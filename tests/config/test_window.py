# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 14:11:24 2022

@author: jpeacock
"""

import unittest
from aurora.config import Window


class TestWindow(unittest.TestCase):
    """
    Test Station metadata
    """

    def setUp(self):
        self.window = Window()

    def test_initialization(self):
        with self.subTest("test num_samples"):
            self.assertEqual(self.window.num_samples, 128)
        with self.subTest("test overlap"):
            self.assertEqual(self.window.overlap, 32)
        with self.subTest("test type"):
            self.assertEqual(self.window.type, "boxcar")
        with self.subTest("test additional_args"):
            self.assertEqual(self.window.additional_args, {})

    def test_add_args_fail(self):
        args = ["a", "b"]

        def set_args(args):
            self.window.additional_args = args

        self.assertRaises(TypeError, set_args, args)

    def test_add_args(self):
        args = {"beta": 10}
        self.window.additional_args = args
        self.assertDictEqual(args, self.window.additional_args)


if __name__ == "__main__":
    unittest.main()
