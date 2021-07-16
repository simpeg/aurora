#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 18:45:37 2017

@author: kkappler
"""


import os
import numpy as np
from scipy.interpolate import interp1d


def interpolate_symmetric_function(x,y,**kwargs):
    """
    A function having even symmetery about x=0 in terms of amplitudes is interpolated
    TODO: but isn't the symmetery even for amplitude and odd for phase?
    @kwarg kind: see doc for interp1d
    @note: This function originally designed to use with fap-tables for 
    calibration calculations
    """
    log_scale = kwargs.get('logScale',True)
    kind = kwargs.get('kind','linear')
    if log_scale:
        temp_function = interp1d(np.log(x), np.log(y), kind=kind, 
                                 bounds_error=False, fill_value='extrapolate')
        interpolator = lambda f: np.exp(temp_function(np.log(np.abs(f))))
    else:
        temp_function = interp1d(x, y, kind=kind, 
                                 bounds_error=False, fill_value='extrapolate')
        interpolator = lambda f: temp_function(np.abs(f))
    return interpolator



