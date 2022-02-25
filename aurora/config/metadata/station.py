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
from .channel import Channel

# =============================================================================
attr_dict = get_schema("station", SCHEMA_FN_PATHS)
# =============================================================================
class Station(Base):
    __doc__ = write_lines(attr_dict)

    def __init__(self, **kwargs):
        super().__init__(attr_dict=attr_dict, **kwargs)
        self.channel_scale_factors = {}
        
    @property
    def channel_scale_factors(self):
        return self._channel_scale_factors
    
    @channel_scale_factors.setter
    def channel_scale_factors(self, values):
        if isinstance(values, list):
            ch_dict = {}
            for element in values:
                if not isinstance(element, Channel):
                    raise ValueError("Elements of list must be Channel objects")
                ch_dict[element.id] = element
            self._channel_scale_factors = ch_dict
        
        elif isinstance(values, dict):
            for key, value in values.items():
                if not isinstance(value, Channel):
                    raise ValueError("Elements of list must be Channel objects")
            
            self._channel_scale_factors = values
            
        elif isinstance(values, Channel):
            self._channel_scale_factors[values.id] =  values
            
    # def to_dict(self, nested=False, single=False, required=True):
        
            

        

        
    
    
