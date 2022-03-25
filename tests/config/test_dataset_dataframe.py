# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 11:46:46 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from aurora.config import Station, Run
from mt_metadata.timeseries import TimePeriod
# =============================================================================

starts = ["2020-01-01T00:00:00", "2020-02-02T00:00:00"]
ends = ["2020-01-31T12:00:00", "2020-02-28T12:00:00"]
s = Station()
s.id = "mt01"
s.mth5_path = "/home/mth5_path.h5"

for ii in range(5):
    r = Run(id=f"{ii:03}", sample_rate=10)
    r.input_channels = ["hx", "hy"]
    r.output_channels = ["hz", "ex", "ey"]
    for start, end in zip(starts, ends):
        r.time_periods.append(TimePeriod(start=start, end=end))
        
    s.runs.append(r)
    
    
    

