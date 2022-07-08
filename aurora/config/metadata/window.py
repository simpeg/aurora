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

# =============================================================================
attr_dict = get_schema("window", SCHEMA_FN_PATHS)
# =============================================================================
class Window(Base):
    __doc__ = write_lines(attr_dict)

    def __init__(self, **kwargs):
        super().__init__(attr_dict=attr_dict, **kwargs)
        self.additional_args = {}

    @property
    def additional_args(self):
        return self._additional_args

    @additional_args.setter
    def additional_args(self, args):
        if not isinstance(args, dict):
            raise TypeError("additional_args must be a dictionary")
        self._additional_args = args
