# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 14:15:20 2022

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

from mt_metadata.base.helpers import write_lines
from mt_metadata.base import get_schema, Base
from .standards import SCHEMA_FN_PATHS

from . import Window, Decimation, Band, Regression


# =============================================================================
attr_dict = get_schema("decimation_level", SCHEMA_FN_PATHS)
attr_dict.add_dict(get_schema("decimation", SCHEMA_FN_PATHS), "decimation")
attr_dict.add_dict(get_schema("window", SCHEMA_FN_PATHS), "window")
attr_dict.add_dict(get_schema("regression", SCHEMA_FN_PATHS), "regression")

# =============================================================================
class DecimationLevel(Base):
    __doc__ = write_lines(attr_dict)

    def __init__(self, **kwargs):
        
        self.window = Window()
        self.decimation = Decimation()
        self.regression = Regression()
        
        self._bands = []
        
        super().__init__(attr_dict=attr_dict, **kwargs)
        
    @property
    def bands(self):
        return self._bands
    
    @bands.setter
    def bands(self, value):
        """
        Set bands make sure they are a band object
        
        :param value: list of bands
        :type value: list, Band

        """
        
        if isinstance(value, Band):
            self._bands = [value]
        
        elif isinstance(value, list):
            self._bands = []
            for obj in value:
                if not isinstance(obj, Band):
                    raise TypeError(
                        f"List entry must be a Band object not {type(obj)}"
                        )
                else:
                    self._bands.append(obj)
        else:
            raise TypeError(f"Not sure what to do with {type(value)}")
            
        
            
                    
            

        

        
    
    
