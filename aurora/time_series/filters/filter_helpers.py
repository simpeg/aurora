from mt_metadata.timeseries.filters.coefficient_filter import CoefficientFilter
from mt_metadata.timeseries.filters.frequency_response_table_filter import (
    FrequencyResponseTableFilter,
)
from aurora.general_helper_functions import TEST_PATH


def make_coefficient_filter(gain=1.0, name="unit_conversion", **kwargs):
    """

    Parameters
    ----------
    gain
    name
    units_in : string
        one of "digital counts", "millivolts", etc.
        TODO: Add a refernce here to the list of units supported in mt_metadata

    Returns
    -------

    """
    # in general, you need to add all required fields from the standards.json
    default_units_in = "units in"
    default_units_out = "units out"
    default_name = "generic coefficient filter"
    cf = CoefficientFilter()
    cf.name = kwargs.get("name", default_name)
    cf.units_in = kwargs.get("units_in", default_units_in)
    cf.units_out = kwargs.get("units_out", default_units_out)
    cf.gain = gain

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


MT2SI_ELECTRIC_FIELD_FILTER = make_coefficient_filter(
    gain=1e6,
    units_in="millivolts per kilometer",
    units_out="volts per meter",
    name="MT to SI electric field " "conversion",
)
