# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 11:46:46 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest

from aurora.config import Station, Run
from mt_metadata.timeseries import TimePeriod
# =============================================================================

class TestStationDataset(unittest.TestCase):
    
    def setUp(self):
        starts = ["2020-01-01T00:00:00", "2020-02-02T00:00:00"]
        ends = ["2020-01-31T12:00:00", "2020-02-28T12:00:00"]
        self.station = Station()
        self.station.id = "mt01"
        self.station.mth5_path = r"/home/mth5_path.h5"
        self.station.remote = False
        
        self.maxDiff = None
        
        for ii in range(5):
            r = Run(id=f"{ii:03}", sample_rate=10)
            r.input_channels = ["hx", "hy"]
            r.output_channels = ["hz", "ex", "ey"]
            for start, end in zip(starts, ends):
                r.time_periods.append(TimePeriod(start=start, end=end))
                
            self.station.runs.append(r)
            
    def test_to_dataframe(self):
        df = self.station.to_dataset_dataframe()
        
        with self.subTest("columns"):
            self.assertListEqual(df.columns.to_list(),
                                 ['station_id',
                                  'run_id',
                                  'start',
                                  'end',
                                  'mth5_path',
                                  'sample_rate',
                                  'input_channels',
                                  'output_channels',
                                  "remote"])
            
        with self.subTest("single station"):
            self.assertTrue(len(df.station_id.unique())==1)
            
    def test_from_dataframe(self):
        df = self.station.to_dataset_dataframe()
        station_2 = Station()
        station_2.from_dataset_dataframe(df)
        df2 = station_2.to_dataset_dataframe()
        
        with self.subTest("df"):
            self.assertTrue((df2 == df).all().all())
        with self.subTest("dict"):
            self.assertDictEqual(station_2.to_dict(single=True), 
                                 self.station.to_dict(single=True))
        


if __name__ == "__main__":
    unittest.main()
    
    

