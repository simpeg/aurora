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
        self._decimations = []
        
        super().__init__(attr_dict=attr_dict, **kwargs)
        
    @property
    def decimations(self):
        return_list = []
        for item in self._decimations:
            if isinstance(item, dict):
                level = DecimationLevel()
                level.from_dict(item)
            elif isinstance(item, DecimationLevel):
                level = item
            return_list.append(level)
            
        return return_list
    
    @decimations.setter
    def decimations(self, value):
        """
        dictionary of decimations levels
        
        :param value: dict of decimation levels
        :type value: dict

        """
        
        if isinstance(value, DecimationLevel):
            self._decimations.append(value)

        
        elif isinstance(value, dict):
            self._decimations = []
            for key, obj in value.items():
                if not isinstance(obj, DecimationLevel):
                    raise TypeError(
                        f"List entry must be a DecimationLevel object not {type(obj)}"
                        )
                else:
                    self._decimations.append(obj)
                    
        elif isinstance(value, list):
            self._decimations = []
            for obj in value:
                if isinstance(value, DecimationLevel):
                    self._decimations.append(obj)

                elif isinstance(obj, dict):
                    level = DecimationLevel()
                    level.from_dict(obj)
                    self._decimations.append(level)
                else:
                    raise TypeError(
                        f"List entry must be a DecimationLevel or dict object not {type(obj)}"
                        )
                    
        else:
            raise TypeError(f"Not sure what to do with {type(value)}")
            
    @property
    def decimations_dict(self):
        """
        need to have a dictionary, but it can't be an attribute cause that
        gets confusing when reading in a json file
        
        :return: DESCRIPTION
        :rtype: TYPE

        """
        return dict([(d.decimation.level, d) for d in self.decimations])
    
            
    def get_decimation_level(self, level):
        """
        Get a decimation level for easy access
        
        :param level: DESCRIPTION
        :type level: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        
        try:
            decimation = self.decimations_dict[level]
        
        except KeyError:
            raise KeyError(f"Could not find {level} in decimations.")
            
        if isinstance(decimation, dict):
            decimation_level = DecimationLevel()
            decimation_level.from_dict(decimation)
            return decimation_level
        
        return decimation
    
    def add_decimation_level(self, decimation_level):
        """
        add a decimation level
        """
        
        if not isinstance(decimation_level, (DecimationLevel, dict)):
            raise TypeError(
                f"List entry must be a DecimationLevel object not {type(decimation_level)}"
                )
        if isinstance(decimation_level, dict):
            obj = DecimationLevel()
            obj.from_dict(decimation_level)
        
        else:
            obj = decimation_level

        self._decimations.append(obj)
    
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
                # python starts from 0
                level = int(level) - 1
                band = Band(decimation_level=level, index_min=low, index_max=high)
                try:
                    self.decimations_dict[level].add_band(band)
                except KeyError:
                    new_level = DecimationLevel()
                    new_level.decimation.level = int(level)
                    new_level.add_band(band)
                    self.add_decimation_level(new_level)

    def json_fn(self):
        json_fn = self.id + "_processing_config.json"
        return json_fn            
    
    def num_decimation_levels(self):
        return len(self.decimations)

    def drop_reference_channels(self):
        for decimation in self.decimations:
            decimation.reference_channels = []
        return

    def validate(self):
        """
        Placeholder.  Some of the checks and methods here maybe better placed in
        TFKernel, which would validate the dataset against the processing config.

        The reason the validator is being created is that the default estimation
        engine from the json file is "RME_RR", which is fine (we expect to in general
        do more RR processing than SS) but if there is only one station (no remote)
        then the RME_RR should be replaced by default with "RME".

        Returns
        -------

        """
        # Make sure a RR method is not being called for a SS config
        if not self.stations.remote:
            for decimation in self.decimations:
                if decimation.estimator.engine == "RME_RR":
                    decimation.estimator.engine = "RME"
