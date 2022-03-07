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
                    
            else:
                raise TypeError(f"not sure what to do with type {type(item)}")
            
            self._runs.append(run)
            
    @property
    def run_list(self):
        """ list of run names """
        
        return [r.id for r in self.runs]
        
            
    
            

        
            

        

        
    
    
