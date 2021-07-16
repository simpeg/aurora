# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 16:10:20 2016

@author: knk1504

super crude development place for instruments.
Probably want a special class Digitizer(),
nothing more than a shelvedInstument with a cpv value

And perhaps a filter_list as a sequence of TFs that we can use for a generic
deployed instrument.
"""
from pathlib import Path

import os
import pdb
import numpy as np
from aurora.sandbox.qf.frequency_series import FrequencySeries

#calibration_path = os.path.join(cache_path, 'resources', 'calibration')
resources_path = Path("/home/kkappler/software/quakefinder/analysis_super/analysis/resources")
calibration_path = resources_path.joinpath('calibration')


#digitizer gets a code for sps or floating point name.

class Instrument(object):
    """
    """
    def __init__(self, **kwargs):
        """
        """
        self.make = kwargs.get('make', None)
        self.model = kwargs.get('model', None)
        self.serial_number = kwargs.get('serial_number', None)
        self.channel = kwargs.get('channel', None)
        self.epoch = kwargs.get('epoch', None)


    def get_cal_file_name(self):
        """
        """
        serial_number = f"{self.serial_number}".zfill(4)
        csv_name = f"{serial_number}_{self.channel}_{self.epoch}.csv"
        full_file_name = os.path.join(calibration_path, self.make, self.model, csv_name)
        return full_file_name

    def __str__(self):
        """
        """
        return "{}-{}".format(self.make, self.model)


class Digitizer(Instrument):
    """
    Special case of Instrument, MMCSE not quite enough to specify the cal-file
    because a digitizer's calibration function in general depends on its sampling
    rate.
    """
    def __init__(self, **kwargs):
        """
        """
        #super(DerivedMeasurand, self).__init__(**kwargs)
        super(Digitizer, self).__init__(**kwargs)
        #Instrument.__init__(self, **kwargs)
        self.idealized_sampling_rate = kwargs.get('sps', None)
        self.actual_sampling_rate = kwargs.get('actual_sps', None)
        self.sampling_rate_label = None

    def get_sampling_rate_label(self):
        """
        The cal-file will be different for the same digitizer MMSCE at different
        sampling rate.  Sampling rates (even idealized ones) can potentially be
        some messy floating point number, which I would prefer
        to not make part of the filename.  This function is a place where i could
        do things like map idealized_sps = 62.259786249058238 -> '62', or 'B'
        or some code like that.  In the general case it is not sufficient to just
        round the sps to the nearest integer (although, now when I think on it,
        its pretty darn close).  For now default to:
        """
        sampling_rate_label = "{}".format(int(np.round(self.idealized_sampling_rate)))
        return sampling_rate_label

    def get_cal_file_name(self):
        """
        """
        serial_number = f"{self.serial_number}".zfill(4)
        csv_name = "{}_{}_{}_{}.csv".format(serial_number, \
        self.channel, self.epoch, self.get_sampling_rate_label())
        full_file_name = os.path.join(calibration_path, self.make, self.model, csv_name)
        return full_file_name


class DeployedInstrument(Instrument):
    """
    This data structure represents the concatenation of instrument, analog
    conditioning and digitization.
    Its pretty fundamental and deserves some design time, but starting with
    a crude prototype hack today (21Mar['span', 'min', 'max', 'sdev', 'label', 'npts', 'id', 'mean']2016)

    @ivar filter_list : a list of instruments each having a response fcn
    @ivar digitizer: Digitizer(Instrument)
    self.analog_board = kwargs.get('analog_board', None) #Instrument
    self.sensor = kwargs.get('sensor', None) #Instrument

    @note 20160324: may want to create a default Instrument with zerophase and unit gain
    as a placeholder ... or not
    """
    def __init__(self, **kwargs):
        """
        """
        super(DeployedInstrument, self).__init__(**kwargs)
        self.sensor = kwargs.get('sensor', Instrument())#None
        self.digitizer = kwargs.get('digitizer', Digitizer())
        self.board = kwargs.get('board', Instrument())
        self.filter_list  = kwargs.get('filter_list', [])
        self.scalar_amplifier = kwargs.get('scalar_amplifier', 1.0)
        self.response_function = kwargs.get('response_function', None)

        #<newly added - should be supported on a case by case basis, i.e.
        #each type of sensor should comput this its own way, i.e. mag vs dipole
        #done differemt
        self.azimuth = kwargs.get('azimuth', None)
        self.dip = kwargs.get('dip', None)

    def get_response_function(self, ignore_digitizer=True, ignore_board=True):
        """
        usage:
        crf_a, crf_b, crf_c = get_response_function()

        @change 20160308: modified to support cal_file_a as a kwarg

        @note 20160323: thinking the default stuff is dangerous and will cause errors
        later.  Change to force explicit instrument passing in next version

        """
        fap_table_a = FrequencySeries()
        fap_table_a.read_from_fap_table(filename=self.sensor.get_cal_file_name())
        crf_a = fap_table_a.complex_response
        if not ignore_board:
            fap_table_b = FrequencySeries()
            fap_table_b.read_from_fap_table(filename=self.board.get_cal_file_name())
            crf_b = fap_table_b.complex_response
        if not ignore_digitizer:
            fap_table_c = FrequencySeries()
            fap_table_c.read_from_fap_table(filename=self.digitizer.get_cal_file_name())
            crf_c = fap_table_c.complex_response

        if ignore_board and ignore_digitizer:
            ('print ignoring board and digitizer response')
            self.response_function = lambda f: self.scalar_amplifier * crf_a(f)
        elif ignore_board:
            ('print ignoring board response')
            self.response_function = lambda f: self.scalar_amplifier * crf_a(f) * crf_c(f)
        elif ignore_digitizer:
            ('print ignoring digitizer response')
            self.response_function = lambda f: self.scalar_amplifier * crf_a(f) * crf_b(f)
        else:
            self.response_function = lambda f: self.scalar_amplifier * crf_a(f) * crf_b(f) * crf_c(f)
        return self.response_function




    def get_sensor_response_function(self):
        """
        @TODO: 20170623: modify maybe so that default args are use a,b,c, but
        can ask for 'sensor_only' etc rather than several methods (no_digi, sensor, etc)
        @note 20160323: THIS IS SUPPOSED TO BE A METHOD OF A DEPLOYED INSTRUMENT!!!
        """
        fap_table_a = FrequencySeries()
        #pdb.set_trace()
        fap_table_a.read_from_fap_table(filename=self.sensor.get_cal_file_name())
        crf_a = fap_table_a.complex_response
        #cal_file_a = FrequencySeries(file_name = self.sensor.get_cal_file_name())
        #cal_file_a = CalibrationFile(file_name = self.sensor.get_cal_file_name())
        #crf_a = fap_table_a.complexResponseFunction

        self.response_function = lambda f: self.scalar_amplifier * crf_a(f)

        return self.response_function






def main():
    """
    """
    print("finito")

if __name__ == "__main__":
    main()

