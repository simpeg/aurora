# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 16:03:21 2021

@author: jpeacock
"""
from pathlib import Path
import pandas as pd
from mth5.timeseries import ChannelTS, RunTS
from mth5.mth5 import MTH5

fn = Path(r"c:\Users\jpeacock\Documents\GitHub\aurora\data\emtf_synthetic\test1.asc")

df = pd.read_csv(fn, names=["hx", "hy", "hz", "ex", "ey"], sep="\s+")

ch_list = []
for col in df.columns:
    data = df[col].values
    if col in ["ex", "ey"]:
        chts = ChannelTS(channel_type="electric", data=data,
                         channel_metadata={"component": col})
        # add metadata to the channel here
        chts.channel_metadata.dipole_length = 50
        
    elif col in ["hx", "hy", "hz"]:
        chts = ChannelTS(channel_type="magnetic", data=data,
                         channel_metadata={"component": col})
    chts.sample_rate = 1
    
    ch_list.append(chts)
    
runts = RunTS(array_list=ch_list)

# add in metadata
runts.station_metadata.id = "mt001"
runts.run_metadata.id = "001"


m = MTH5()
m.open_mth5(fn.parent.joinpath("emtf_synthetic.h5"))
station_group = m.add_station("mt001")
run_group = station_group.add_run("001")
run_group.from_runts(runts)
m.close_mth5()
