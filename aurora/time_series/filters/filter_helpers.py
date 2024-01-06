from mt_metadata.timeseries.filters.coefficient_filter import CoefficientFilter
from mt_metadata.timeseries.filters.frequency_response_table_filter import (
    FrequencyResponseTableFilter,
)
from loguru import logger


def make_coefficient_filter(gain=1.0, name="generic coefficient filter", **kwargs):
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
    default_units_in = "unknown"
    default_units_out = "unknown"

    cf = CoefficientFilter()
    cf.gain = gain
    cf.name = name

    cf.units_in = kwargs.get("units_in", default_units_in)
    cf.units_out = kwargs.get("units_out", default_units_out)

    return cf


def make_frequency_response_table_filter(file_path, case="bf4"):
    """
    Parameters
    ----------
    filepath: pathlib.Path or string
    case : string, placeholder for handlig different fap table formats.

    Returns
    -------
    fap_filter: FrequencyResponseTableFilter
    """
    fap_filter = FrequencyResponseTableFilter()

    if case == "bf4":
        import numpy as np
        import pandas as pd

        df = pd.read_csv(file_path)  # , skiprows=1)
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
        logger.error(f"case {case} not supported for FAP Table")
        raise Exception


def make_volt_per_meter_to_millivolt_per_km_converter():
    """
    This represents a filter that converts from mV/km to V/m.

    Returns
    -------

    """
    coeff_filter = make_coefficient_filter(
        gain=1e-6,
        units_in="millivolts per kilometer",
        units_out="volts per meter",
        name="MT to SI electric field conversion",
    )
    return coeff_filter


def make_tesla_to_nanotesla_converter():
    """
    This represents a filter that converts from nt to T.

    Returns
    -------

    """
    coeff_filter = make_coefficient_filter(
        gain=1e-9,
        units_in="nanotesla",
        units_out="tesla",
        name="MT to SI magnetic field conversion",
    )
    return coeff_filter


MT2SI_ELECTRIC_FIELD_FILTER = make_volt_per_meter_to_millivolt_per_km_converter()
MT2SI_MAGNETIC_FIELD_FILTER = make_tesla_to_nanotesla_converter()


def main():
    make_volt_per_meter_to_millivolt_per_km_converter()
