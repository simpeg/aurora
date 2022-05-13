"""
Definitions used in the creation of synthetic mth5 files.

If this model gets used a lot we may want to formalize these, either with a class like:
class StationConfig(object):
    def __init__(self, **kwargs):
        self.raw_data_path = kwargs.get("raw_data_path", None)
        self.columns = ["hx", "hy", "hz", "ex", "ey"]
        self.mth5_path = kwargs.get("mth5_path", None)

        #<depends on columns>
        self.noise_scalar = {}
        for col in self.columns:
            self.noise_scalar[col] = 0.0
        #</depends on columns>
Or we could the station info in a dictionary and formlize it with a standards.json
"""

import random

from aurora.test_utils.synthetic.paths import DATA_PATH
from aurora.time_series.filters.filter_helpers import make_coefficient_filter

random.seed(0)


def make_filters(as_list=False):
    """
    Because the data from EMTF is already in mV/km and nT these filters are just
    placeholders to show where they would get assigned.

    Returns
    -------
    filters_list: list
        filters that can be ussed to populate the filters lists of synthetic data
    """
    unity_coeff_filter = make_coefficient_filter(name="1", gain=1.0)
    multipy_by_10_filter = make_coefficient_filter(gain=10.0, name="10")
    divide_by_10_filter = make_coefficient_filter(gain=0.1, name="0.1")

    if as_list:
        return [unity_coeff_filter, multipy_by_10_filter, divide_by_10_filter]
    else:
        filters = {}
        filters["1x"] = unity_coeff_filter
        filters["10x"] = multipy_by_10_filter
        filters["0.1x"] = divide_by_10_filter
        return filters


def make_station_01_config_dict():
    station_dict = {}
    station_dict["raw_data_path"] = DATA_PATH.joinpath("test1.asc")
    station_dict["mth5_path"] = DATA_PATH.joinpath("test1.h5")
    station_dict["columns"] = ["hx", "hy", "hz", "ex", "ey"]
    station_dict["noise_scalar"] = {}
    for col in station_dict["columns"]:
        station_dict["noise_scalar"][col] = 0.0

    # create a tuple of index, n_samples that get set to nan, see issue #59
    station_dict["nan_indices"] = {}
    for col in station_dict["columns"]:
        station_dict["nan_indices"][col] = []
        if col == "hx":
            station_dict["nan_indices"][col].append([11, 100])
        if col == "hy":
            station_dict["nan_indices"][col].append([11, 100])
            station_dict["nan_indices"][col].append([20000, 444])
        # if col == "ex":
        #     station_dict["nan_indices"][col].append([10000, 100])

    filters = make_filters()
    station_dict["filters"] = {}
    for col in station_dict["columns"]:
        if col in ["ex", "ey"]:
            station_dict["filters"][col] = [filters["1x"].name,]
    for col in station_dict["columns"]:
        if col in ["hx", "hy", "hz"]:
            station_dict["filters"][col] = [filters["10x"].name, filters["0.1x"].name]

    station_dict["run_id"] = "001"
    station_dict["station_id"] = "test1"
    station_dict["sample_rate"] = 1.0
    station_dict["latitude"] = 17.996

    return station_dict

def make_station_02_config_dict():
    station_dict = make_station_01_config_dict()
    station_dict["raw_data_path"] = DATA_PATH.joinpath("test2.asc")
    station_dict["mth5_path"] = DATA_PATH.joinpath("test2.h5")
    station_dict["station_id"] = "test2"
    station_dict["nan_indices"] = {}
    for col in station_dict["columns"]:
        station_dict["nan_indices"][col] = []
    return station_dict
