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

from .window import Window
from .decimation import Decimation


# =============================================================================
attr_dict = get_schema("decimation_level", SCHEMA_FN_PATHS)
attr_dict.add_dict(get_schema("decimation"), "decimation")
attr_dict.add_dict(get_schema("window"), "window")
attr_dict.add_dict(get_schema("regression"), "regression")

# =============================================================================
class DecimationLevel(Base):
    __doc__ = write_lines(attr_dict)

    def __init__(self, **kwargs):
        
        self.window = Window()
        self.decimation = Decimation()
        
        # self.bands = Bands()
        
        super().__init__(attr_dict=attr_dict, **kwargs)

        

        
    
    
