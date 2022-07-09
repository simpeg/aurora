"""
Definitions used in the creation of synthetic mth5 files.


Survey level: 'mth5_path', Path to output h5
Station level: 'station_id', name of the station
Station level:'latitude':17.996

Run level:'columns', :channel names as a list; ["hx", "hy", "hz", "ex", "ey"]
Run level: 'raw_data_path', Path to ascii data source
Run level: 'noise_scalars', dict keyed by channel, default is zero,
Run level: 'nan_indices', iterable of integers, where to put nan [
Run level: 'filters', dict of filters keyed by columns
Run level: 'run_id', name of the run
Run level: 'sample_rate', 1.0

"""
import numpy as np
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
    def __init__(self, id, **kwargs):
        self.id = id
        self.sample_rate = kwargs.get("sample_rate", 1.0)
        self.raw_data_path = kwargs.get("raw_data_path", None)
        self.channels = kwargs.get("channels", ["hx", "hy", "hz", "ex", "ey"])
        self.noise_scalars = kwargs.get("noise_scalars", None)
        self.nan_indices = kwargs.get("nan_indices", {})
        self.filters = kwargs.get("filters", {})

        if self.noise_scalars is None:
            self.noise_scalars = {}
            for channel in self.channels:
                self.noise_scalars[channel] = 0.0  # np.random.rand(1)


class SyntheticStation(object):
    def __init__(self, id, **kwargs):
        self.id = id
        self.latitude = kwargs.get("latitude", 0.0)
        self.runs = []
        self.mth5_path = kwargs.get("mth5_path", None)  # not always used


def make_station_01():
    station = SyntheticStation("test1")
    station.mth5_path = DATA_PATH.joinpath("test1.h5")

    run_001 = SyntheticRun(
        "001",
        raw_data_path=DATA_PATH.joinpath("test1.asc"),
    )
    nan_indices = {}
    for ch in run_001.channels:
        nan_indices[ch] = []
        if ch == "hx":
            nan_indices[ch].append([11, 100])
        if ch == "hy":
            nan_indices[ch].append([11, 100])
            nan_indices[ch].append([20000, 444])
    run_001.nan_indices = nan_indices

    filters = {}
    for ch in run_001.channels:
        if ch in ["ex", "ey"]:
            filters[ch] = [
                FILTERS["1x"].name,
            ]
        elif ch in ["hx", "hy", "hz"]:
            filters[ch] = [FILTERS["10x"].name, FILTERS["0.1x"].name]
    run_001.filters = filters

    station.runs = [
        run_001,
    ]

    return station


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
    station = SyntheticStation("test3")
    station.mth5_path = DATA_PATH.joinpath("test3.h5")
    channels = ["hx", "hy", "hz", "ex", "ey"]

    nan_indices = {}
    for ch in channels:
        nan_indices[ch] = []

    filters = {}
    for ch in channels:
        if ch in ["ex", "ey"]:
            filters[ch] = [
                FILTERS["1x"].name,
            ]
        elif ch in ["hx", "hy", "hz"]:
            filters[ch] = [FILTERS["10x"].name, FILTERS["0.1x"].name]

    run_001 = SyntheticRun(
        "001",
        raw_data_path=DATA_PATH.joinpath("test1.asc"),
        nan_indices=nan_indices,
        filters=filters,
    )

    noise_scalars = {}
    for ch in channels:
        noise_scalars[ch] = 2.0
    run_002 = SyntheticRun(
        "002",
        raw_data_path=DATA_PATH.joinpath("test1.asc"),
        noise_scalars=noise_scalars,
        nan_indices=nan_indices,
        filters=filters,
    )

    for ch in channels:
        noise_scalars[ch] = 5.0
    run_003 = SyntheticRun(
        "003",
        raw_data_path=DATA_PATH.joinpath("test1.asc"),
        noise_scalars=noise_scalars,
        nan_indices=nan_indices,
        filters=filters,
    )

    for ch in channels:
        noise_scalars[ch] = 10.0
    run_004 = SyntheticRun(
        "004",
        raw_data_path=DATA_PATH.joinpath("test1.asc"),
        noise_scalars=noise_scalars,
        nan_indices=nan_indices,
        filters=filters,
    )

    run_001.filters = filters
    run_002.filters = filters
    run_003.filters = filters
    run_004.filters = filters

    station.runs = [run_001, run_002, run_003, run_004]

    return station


# def main():
#     make_station_01()
#
# if __name__ == "__main__":
#     main()
