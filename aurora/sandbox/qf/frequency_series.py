# -*- coding: utf-8 -*-
"""
@date: 20170727
@author: kkappler

This is a general way to handle calibration functions and possibly to factor the STFT
which of course is a multivariate frequency series ...

Basically, a frequency series is just a series, like as in pd.Series()
but in this case the x-axis is based on frequecny not time.

The fundamental requirements for this class are to be able to load a fap-table
(frequecy, ampltude, phase) and return a lambda function (or similar) that
can interpolate/extrapolate as needed.

Specific Requirements:
    
    
@TODO: Add support for parametric frequency response (pz tables) 

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import datetime
import pdb
from aurora.sandbox.qf.supporting_interpolation import interpolate_symmetric_function
from scipy.interpolate import interp1d

class FrequencySeries():
    """
    @note: no assumption of uniform sampling
    @note: This class works with fap-tables
    """

    def __init__(self, **kwargs):
        """
        @ivar data: numpy array; (1d)
        @note: 20170726  In future we will support multivariate
        time series.  In keeping with thinking about "tabular, culumnwise data"
        lets say that the first dimension (num rows) is the time dimension, and
        num columns (or the second dimension) is the enumerator of channels.

        """
        self.frequencies = None
        self.data = kwargs.get('data', None)
        self.log_amplitude_interpolate = kwargs.get('log_amplitude_interpolate', True)
        self.log_phase_interpolate = kwargs.get('log_phase_interpolate', False)
        self.interp_kind = 'linear'
        self._amplitudes = None
        self._phases = None

    def read_from_fap_table(self, filename):
        """
        @type filename: string points to fap-table.
        @TODO: Built-in assumption that columns are Frequncy amplitude and phase
        We need to confirm this
        @note: a similar class at one time had support for units/ parsing
        column names and genrealized TF handling, however those classes cannot
        be used for our research because of IP concerns so this is a hack to 
        replace those classes.  
        """
        fap_table = pd.read_csv(filename)
        column_labels =fap_table.columns
        
        if not column_labels[0][0:9].lower()=='frequency':
            print("unknown column in fap/cal table")
            raise(Exception)
        self.frequencies = fap_table[column_labels[0]]

        if not column_labels[1][0:9].lower()=='amplitude':
            print("unknown column in fap/cal table")
            raise(Exception)
        self._amplitudes = fap_table[column_labels[1]]
        if not column_labels[2][0:5].lower()=='phase':
            print("unknown column in fap/cal table")
            raise(Exception)
        self._phases = fap_table[column_labels[2]]
        print('fap read complete;')
        self.generate_response_functions()
        #pdb.set_trace()
        
        
        
        #add some vaidation here; like check that the units are expected
        #assign self.frequencies the value of the frequency column
        #assign self.data to be amplitudes*exp(j*phases) where ampl, phase are columns
        return
    
    def generate_response_functions(self):
        """
        @warn: handling of negative frequencies not yet determined
        @TODO: Add support for parametric frequency response (pz tables) 
        """
        self.amplitude_response = interpolate_symmetric_function(self.frequencies, self.amplitudes,\
            logScale=self.log_amplitude_interpolate, kind=self.interp_kind, \
            bounds_error=False, fill_value='extrapolate')
        
        self.phase_response = interpolate_symmetric_function(self.frequencies, self.phases,\
            logScale=self.log_phase_interpolate, kind=self.interp_kind, \
            bounds_error=False, fill_value='extrapolate')        
        
        print("OK now test this against an older version of amplitude response, \
              one  that is based on deprecated.  DO this by generating random \
              numbers and running them through both functions and quantifying the difference")

        j = np.complex(0, 1)
        self.complex_response = lambda f: self.amplitude_response(f) * \
        np.exp( j * np.sign(f) * (np.pi/180) * self.phase_response(f))
        return
    

    @property
    def amplitudes(self):
        if self._amplitudes is not None:
            return self._amplitudes
        else:
            return np.abs(self.data)

    @property
    def phases(self):
        if self._phases is not None:
            return self._phases
        else:
            return np.angle(self.data)

#    def _as_function(self):
#        j = np.complex(0,1)
#        self.angle_radians = lambda f: np.sign(f) * \
#            self.phaseResponseFunction(np.abs(f))
#            self.complexResponseFunction = lambda f: self.amplitudeResponseFunction(f) * \
#            np.exp(j*1.*self.angleInRadians(f))
#        return self.complexResponseFunction
    def plot(self,**kwargs):
        """
        @TODO: support loglog, linlin, semilogx, semilogy

        @kwarg savefigname: filename for a png to save
        @TODO: add figsize and dpi to standardize these figures for svfig
        @TODO; standardize on samples, seconds, datetimes, etc for time axis
        """
        plt.figure()
        plt.clf()

        save_figure_filename = kwargs.get('save_figure_filename',None)
        ttl = kwargs.get('title',None)

        plt.loglog(self.frequencies,self.amplitudes)

        if ttl is not None:
            plt.title(ttl)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')
        if save_figure_filename is not None:
            plt.savefig(save_figure_filename+'.png')


def main():
    data = np.random.rand(10000)
    x = FrequencySeries()
    print("finito {}".format(datetime.datetime.now()))

if __name__ == '__main__':
    main()
