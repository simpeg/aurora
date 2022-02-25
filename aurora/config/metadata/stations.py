# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 13:58:07 2022

@author: jpeacock
"""

from . import Station

class Stations:
    """
    class to hold station information
    
    station to process
    remote references to use
    
    """
    
    def __init__(self, **kwargs):
        self.local = Station()
        self.remote = {}
        
    
        