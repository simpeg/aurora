"""
Definitions used in the creation of synthetic mth5 files.


Survey level: 'mth5_path', Path to output h5
Station level: 'station_id', name of the station
Station level:'latitude':17.996

Run level:'columns', :channel names as a list; ["hx", "hy", "hz", "ex", "ey"]
Run level: 'raw_data_path', Path to ascii data source
Run level: 'noise_scalar', [0.0, 0.0, 0.0, 0.0, 0.0]
Run level: 'nan_indices', iterable of integers, where to put nan [
Run level: 'filters', dict of filters keyed by columns
Run level: 'run_id', name of the run
Run level: 'sample_rate', 1.0

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

FILTERS = make_filters()

class SyntheticRun(object):
    def __init__(self, id,  **kwargs):
        self.id = id
        self.sample_rate = kwargs.get("sample_rate", 1.0)
        self.raw_data_path = kwargs.get("raw_data_path", None)
        self.channels = kwargs.get("channels", ["hx", "hy", "hz", "ex", "ey"])
        self.noise_scalar = kwargs.get("noise_scalar", None)
        self.nan_indices = kwargs.get("nan_indices", {})
        self.filters = kwargs.get("filters", {})

        if self.noise_scalar is None:
            self.noise_scalar = {}
            for channel in self.channels:
                self.noise_scalar[channel] = 0.0 #np.random.rand(1)


class SyntheticStation(object):
    def __init__(self, id,  **kwargs):
        self.id = id
        self.latitude = kwargs.get("latitude", 0.0)
        self.runs = []
        self.mth5_path = kwargs.get("mth5_path", None) #not always used


def make_station_01():
    test1 = SyntheticStation("test1")
    test1.mth5_path = DATA_PATH.joinpath("test1.h5")
    channels = ["hx", "hy", "hz", "ex", "ey"]

    run_001 = SyntheticRun("001",
                           raw_data_path=DATA_PATH.joinpath("test1.asc"),
                           )
    nan_indices = {}
    for col in run_001.channels:
        nan_indices[col] = []
        if col == "hx":
            nan_indices[col].append([11, 100])
        if col == "hy":
            nan_indices[col].append([11, 100])
            nan_indices[col].append([20000, 444])
    run_001.nan_indices = nan_indices

    filters = {}
    for col in run_001.channels:
        if col in ["ex", "ey"]:
            filters[col] = [FILTERS["1x"].name,]
        elif col in ["hx", "hy", "hz"]:
            filters[col] = [FILTERS["10x"].name, FILTERS["0.1x"].name]
    run_001.filters = filters

    test1.runs = [run_001,]

    return test1





def make_station_02():
    test2 = make_station_01()
    test2.mth5_path = DATA_PATH.joinpath("test2.h5")
    test2.id = "test2"
    test2.runs[0].raw_data_path = DATA_PATH.joinpath("test2.asc")
    nan_indices = {}
    for channel in test2.runs[0].channels:
        nan_indices[channel] = []
    test2.runs[0].nan_indices = nan_indices
    return test2


def make_station_03():
    station_dict = make_station_01_config_dict()
    station_dict["raw_data_path"] = DATA_PATH.joinpath("test3.asc")
    station_dict["mth5_path"] = DATA_PATH.joinpath("test3.h5")
    station_dict["station_id"] = "test3"
    station_dict["nan_indices"] = {}
    for col in station_dict["columns"]:
        station_dict["nan_indices"][col] = []
    station_dict["run_id"] = ["a", "b", "c", "d"]
    return station_dict


# def main():
#     make_station_01()
#
# if __name__ == "__main__":
#     main()