# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 14:15:20 2022

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
import pandas as pd

from mt_metadata.base.helpers import write_lines
from mt_metadata.base import get_schema, Base
from mt_metadata.timeseries import TimePeriod
from .standards import SCHEMA_FN_PATHS
from .run import Run

# =============================================================================
attr_dict = get_schema("station", SCHEMA_FN_PATHS)
# =============================================================================
class Station(Base):
    __doc__ = write_lines(attr_dict)

    def __init__(self, **kwargs):
        super().__init__(attr_dict=attr_dict, **kwargs)
        self._runs = []
            
    @property
    def runs(self):
        return self._runs
    
    @runs.setter
    def runs(self, values):
        self._runs = []
        if not isinstance(values, list):
            values = [values]
            
        for item in values:
            if isinstance(item, str):
                run = Run(id=item)
            elif isinstance(item, Run):
                run = item
            elif isinstance(item, dict):
                run = Run()
                run.from_dict(item)
                    
            else:
                raise TypeError(f"not sure what to do with type {type(item)}")
            
            self._runs.append(run)
            
    @property
    def run_list(self):
        """ list of run names """
        
        return [r.id for r in self.runs]
    
    @property
    def run_dict(self):
        """
        need to have a dictionary, but it can't be an attribute cause that
        gets confusing when reading in a json file
        
        :return: DESCRIPTION
        :rtype: TYPE

        """
        return dict([(rr.id, rr) for rr in self.runs])
    
    def to_dataset_dataframe(self):
        """
        Create a dataset definition dataframe that can be used in the 
        processing
        
        [
            "station_id", 
            "run_id",
            "start",
            "end",
            "mth5_path",
            "sample_rate",
            "input_channels",
            "output_channels"
        ] 
        
        """
        
        data_list = []
        
        for run in self.runs:
            for tp in run.time_periods:
                entry = {
                    "station_id": self.id,
                    "run_id": run.id,
                    "start": tp.start,
                    "end": tp.end,
                    "mth5_path": self.mth5_path,
                    "sample_rate": run.sample_rate,
                    "input_channels": run.input_channel_names,
                    "output_channels": run.output_channel_names}
                data_list.append(entry)
                
        df = pd.DataFrame(data_list)
        df.start = pd.to_datetime(df.start)
        df.end = pd.to_datetime(df.end)
  
        return df
    
    def from_dataset_dataframe(self, df):
        """
        set a data frame
        
        [
            "station_id", 
            "run_id",
            "start",
            "end",
            "mth5_path",
            "sample_rate",
            "input_channels",
            "output_channels"
        ] 
        
        :param df: DESCRIPTION
        :type df: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        
        self.runs = []
        
        self.id = df.station_id.unique()[0]
        self.mth5_path = df.mth5_path.unique()[0]
        
        for entry in df.itertuples():
            try:
                r = self.run_dict[entry.run_id]
                r.time_periods.append(TimePeriod(start=entry.start.isoformat(),
                                                 end=entry.end.isoformat()))
                
            except KeyError:
                r = Run(
                    id=entry.run_id, 
                    sample_rate=entry.sample_rate,
                    input_channels=entry.input_channels,
                    output_channels=entry.output_channels
                    )
                
                r.time_periods.append(TimePeriod(start=entry.start.isoformat(),
                                                 end=entry.end.isoformat()))
                self.runs.append(r)
            
            
            
            
