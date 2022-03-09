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

from . import DecimationLevel, Stations, Band


# =============================================================================
attr_dict = get_schema("processing", SCHEMA_FN_PATHS)
attr_dict.add_dict(Stations()._attr_dict, "stations")

# =============================================================================
class Processing(Base):
    __doc__ = write_lines(attr_dict)

    def __init__(self, **kwargs):
        
        self.stations = Stations()
        self._decimations = {}
        
        super().__init__(attr_dict=attr_dict, **kwargs)
        
    @property
    def decimations(self):
        return_dict = {}
        for k, v in self._decimations.items():
            if isinstance(v, dict):
                level = DecimationLevel()
                level.from_dict(v)
            elif isinstance(v, DecimationLevel):
                level = v
            return_dict[k] = level
            
        return self._decimations
    
    @decimations.setter
    def decimations(self, value):
        """
        dictionary of decimations levels
        
        :param value: dict of decimation levels
        :type value: dict

        """
        
        if isinstance(value, DecimationLevel):
            self._decimations = {value.decimation.level: value}

        
        elif isinstance(value, dict):
            self._decimations = {}
            for key, obj in value.items():
                if not isinstance(obj, DecimationLevel):
                    raise TypeError(
                        f"List entry must be a DecimationLevel object not {type(obj)}"
                        )
                else:
                    self._decimations[key] = obj
                    
        elif isinstance(value, list):
            self._decimations = {}
            for obj in value:
                if not isinstance(obj, DecimationLevel):
                    raise TypeError(
                        f"List entry must be a DecimationLevel object not {type(obj)}"
                        )
                else:
                    self._decimations[obj.decimation.level] = obj
                    
        else:
            raise TypeError(f"Not sure what to do with {type(value)}")
            
    def get_decimation_level(self, level):
        """
        Get a decimation level for easy access
        
        :param level: DESCRIPTION
        :type level: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        
        try:
            decimation = self.decimations[level]
        
        except KeyError:
            raise KeyError(f"Could not find {level} in decimations.")
            
        if isinstance(decimation, dict):
            decimation_level = DecimationLevel()
            decimation_level.from_dict(decimation)
            return decimation_level
        
        return decimation
    
    def read_emtf_bands(self, emtf_fn):
        """
        Read an emtf style file for defining the bands
        
        :param emtf_fn: full path to emtf band file
        :type emtf_fn: string or Path

        """
        
        emtf_fn = Path(emtf_fn)
        
        if not emtf_fn.exists():
            raise IOError(f"Could not find {emtf_fn}")
            

        for line in emtf_fn.read_text().split("\n")[1:]:
            if len(line) >= 3:
                level, low, high = [int(ii.strip()) for ii in line.split()]
                band = Band(decimation_level=level, index_min=low, index_max=high)
                try:
                    self.decimations[level].add_band(band)
                except KeyError:
                    new_level = DecimationLevel()
                    new_level.decimation.level = level
                    new_level.add_band(band)
                    self.decimations.update({level: new_level})
                
