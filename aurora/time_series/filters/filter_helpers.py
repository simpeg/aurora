from mt_metadata.timeseries.filters.coefficient_filter import CoefficientFilter
from mt_metadata.timeseries.filters.frequency_response_table_filter import (
    FrequencyResponseTableFilter,
)
from aurora.general_helper_functions import TEST_PATH


def make_coefficient_filter(gain=1.0, name="unit_conversion"):
    # in general, you need to add all required fields from the
    # standards.json
    # coeff_filter = CoefficientFilter()
    cf = CoefficientFilter()
    cf.units_in = "digital counts"
    cf.units_out = "millivolts"
    cf.gain = gain
    cf.name = name
    return cf


def make_frequency_response_table_filter(case="bf4"):
    fap_filter = FrequencyResponseTableFilter()
    if case == "bf4":
        import numpy as np
        import pandas as pd

        bf4_path = TEST_PATH.joinpath("parkfield", "bf4_9819.csv")
        df = pd.read_csv(bf4_path)  # , skiprows=1)
        # Hz, V/nT, degrees
        fap_filter.frequencies = df["Frequency [Hz]"].values
        fap_filter.amplitudes = df["Amplitude [V/nT]"].values
        fap_filter.phases = np.deg2rad(df["Phase [degrees]"].values)
        fap_filter.units_in = "volts"
        fap_filter.units_out = "nanotesla"
        fap_filter.gain = 1.0
        fap_filter.name = "bf4"
        return fap_filter
    else:
        print(f"case {case} not supported for FAP Table")
        raise Exception
