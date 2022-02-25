# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 14:11:24 2022

@author: jpeacock
"""

import unittest
from aurora.config import Decimation

class TestDecimation(unittest.TestCase):
    """
    Test Station metadata
    """
    
    def setUp(self):
        self.decimation = Decimation()
        
    def test_initialization(self):
        with self.subTest("test level_id"):
            self.assertEqual(self.decimation.level_id, 0)
        with self.subTest("test factor"):
            self.assertEqual(self.decimation.factor, 1)
        with self.subTest("test method"):
            self.assertEqual(self.decimation.method, "default")
        
        


if __name__ == "__main__":
    unittest.main()            
