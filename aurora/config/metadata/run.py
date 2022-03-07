# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 14:15:20 2022

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
from mt_metadata.base.helpers import write_lines
from mt_metadata.base import get_schema, Base
from .standards import SCHEMA_FN_PATHS
from mt_metadata.timeseries import TimePeriod
from .channel import Channel

# =============================================================================
attr_dict = get_schema("run", SCHEMA_FN_PATHS)
attr_dict.add_dict(TimePeriod()._attr_dict, "time_period")
# =============================================================================
class Run(Base):
    __doc__ = write_lines(attr_dict)

    def __init__(self, **kwargs):
        self._input = []
        self._output = []
        self.time_period = TimePeriod()
        
        super().__init__(attr_dict=attr_dict, **kwargs)
        
    @property
    def input_channels(self):
        return self._input
    
    @input_channels.setter
    def input_channels(self, values):
        self._input = []
        if not isinstance(values, list):
            values = [values]
            
        for item in values:
            if isinstance(item, str):
                ch = Channel(id=item)
            elif isinstance(item, Channel):
                ch = item
                
            else:
                raise TypeError(f"not sure what to do with type {type(item)}")
            
            self._input.append(ch)
                
            
    @property
    def output_channels(self):
        return self._output
    
    @output_channels.setter
    def output_channels(self, values):
        self._output = []
        if not isinstance(values, list):
            values = [values]
            
        for item in values:
            if isinstance(item, str):
                ch = Channel(id=item)
            elif isinstance(item, Channel):
                ch = item
                    
            else:
                raise TypeError(f"not sure what to do with type {type(item)}")
            
            self._output.append(ch)
                

        
    
    
