# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 13:58:07 2022

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
from mt_metadata.base import Base, get_schema
from .standards import SCHEMA_FN_PATHS
from .station import Station

# =============================================================================
attr_dict = get_schema("stations", SCHEMA_FN_PATHS)
attr_dict.add_dict(get_schema("station", SCHEMA_FN_PATHS), "local")

# =============================================================================
class Stations(Base):
    """
    class to hold station information
    
    station to process
    remote references to use
    
    """
    
    def __init__(self, **kwargs):
        self.local = Station()
        self._remote = {}
        
        super().__init__(attr_dict=attr_dict, **kwargs)
        
        
    @property
    def remote(self):
        return self._remote
    
    @remote.setter
    def remote(self, rr_station):
        if isinstance(rr_station, list):
            for item in rr_station:
                if not isinstance(item, Station):
                    raise TypeError(f"list item must be Station object not {type(item)}")
                self._remote[item.id] = item
                
        elif isinstance(rr_station, dict):
            self._remote = rr_station
        elif isinstance(rr_station, Station):
            self._remote[rr_station.id] = rr_station
        else:
            raise ValueError(f"not sure to do with {type(rr_station)}")
        
        
    
        