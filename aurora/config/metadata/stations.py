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
        self._remote = []
        
        super().__init__(attr_dict=attr_dict, **kwargs)
        
        
    @property
    def remote(self):
        return_list = []
        for rr in self._remote:
            if isinstance(rr, dict):
                b = Station()
                b.from_dict(rr)
                b.remote = True
            elif isinstance(rr, Station):
                b = rr
                b.remote = True
            return_list.append(rr)
        return return_list
    
    @remote.setter
    def remote(self, rr_station):
        self._remote = []
        if isinstance(rr_station, list):
            for item in rr_station:
                if not isinstance(item, Station):
                    raise TypeError(f"list item must be Station object not {type(item)}")
                self._remote.append(item)
                
        elif isinstance(rr_station, dict):
            remote = Station()
            remote.from_dict(rr_station)
            remote.remote = True
            self._remote.append(remote)
            
        elif isinstance(rr_station, Station):
            rr_station.remote = True
            self._remote.append(rr_station)
        else:
            raise ValueError(f"not sure to do with {type(rr_station)}")
            
    def add_remote(self, rr):
        """
        add a band
        """
        
        if not isinstance(rr, (Station, dict)):
            raise TypeError(
                f"List entry must be a Station object not {type(rr)}"
                )
        if isinstance(rr, dict):
            obj = Station()
            obj.from_dict(rr)
        
        else:
            obj = rr
            
        obj.remote = True

        self._remote.append(obj)
        
    @property
    def remote_dict(self):
        """
        need to have a dictionary, but it can't be an attribute cause that
        gets confusing when reading in a json file
        
        :return: DESCRIPTION
        :rtype: TYPE

        """
        return dict([(rr.id, rr) for rr in self.remote])
        
        
    
        