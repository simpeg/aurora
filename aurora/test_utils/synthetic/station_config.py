"""
Definitions used in the creation of synthetic mth5 files.


Survey level: 'mth5_path', Path to output h5
Station level: 'station_id', name of the station
Station level:'latitude':17.996

Run level: 'columns', :channel names as a list; ["hx", "hy", "hz", "ex", "ey"]
Run level: 'raw_data_path', Path to ascii data source
Run level: 'noise_scalars', dict keyed by channel, default is zero,
Run level: 'nan_indices', iterable of integers, where to put nan [
Run level: 'filters', dict of filters keyed by columns
Run level: 'run_id', name of the run
Run level: 'sample_rate', 1.0

"""
import numpy as np
import random

from mt_metadata.transfer_functions.processing.aurora import ChannelNomenclature
from aurora.test_utils.synthetic.paths import SyntheticTestPaths
from aurora.time_series.filters.filter_helpers import make_coefficient_filter

random.seed(0)

synthetic_test_paths = SyntheticTestPaths()
DATA_PATH = synthetic_test_paths.ascii_data_path


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
    """ """

    def __init__(self, id, **kwargs):
        self.id = id
        self.sample_rate = kwargs.get("sample_rate", 1.0)
        self.raw_data_path = kwargs.get("raw_data_path", None)

        # set channel names
        self._channel_map = None
        self.channel_nomenclature_keyword = kwargs.get(
            "channel_nomenclature", "default"
        )
        self.set_channel_map()
        self.channels = kwargs.get("channels", list(self.channel_map.values()))

        self.noise_scalars = kwargs.get("noise_scalars", None)
        self.nan_indices = kwargs.get("nan_indices", {})
        self.filters = kwargs.get("filters", {})
        self.start = kwargs.get("start", None)

        if self.noise_scalars is None:
            self.noise_scalars = {}
            for channel in self.channels:
                self.noise_scalars[channel] = 0.0  # np.random.rand(1)

    @property
    def channel_map(self):
        return self._channel_map

    def set_channel_map(self):
        channel_nomenclature = ChannelNomenclature()
        channel_nomenclature.keyword = self.channel_nomenclature_keyword
        channel_map = channel_nomenclature.get_channel_map(
            self.channel_nomenclature_keyword
        )
        self._channel_map = channel_map


class SyntheticStation(object):
    """
    TODO: could add channel_nomenclature to this obj (instead of run, say) and clean
    things up somewhat. ... i.e. inclde the channel_map() property etc.

    """

    def __init__(self, id, **kwargs):
        self.id = id
        self.latitude = kwargs.get("latitude", 0.0)
        self.runs = []
        self.mth5_name = kwargs.get("mth5_name", None)


def make_station_01(channel_nomenclature="default"):
    channel_nomenclature_obj = ChannelNomenclature()
    channel_nomenclature_obj.keyword = channel_nomenclature
    EX, EY, HX, HY, HZ = channel_nomenclature_obj.unpack()
    station = SyntheticStation("test1")
    station.mth5_name = "test1.h5"

    run_001 = SyntheticRun(
        "001",
        raw_data_path=DATA_PATH.joinpath("test1.asc"),
        channel_nomenclature=channel_nomenclature,
        start=None,
    )
    nan_indices = {}
    for ch in run_001.channels:
        nan_indices[ch] = []
        if ch == HX:
            nan_indices[ch].append([11, 100])
        if ch == HY:
            nan_indices[ch].append([11, 100])
            nan_indices[ch].append([20000, 444])
    run_001.nan_indices = nan_indices

    filters = {}
    for ch in run_001.channels:
        if ch in [EX, EY]:
            filters[ch] = [
                FILTERS["1x"].name,
            ]
        elif ch in [HX, HY, HZ]:
            filters[ch] = [FILTERS["10x"].name, FILTERS["0.1x"].name]
    run_001.filters = filters

    station.runs = [
        run_001,
    ]

    return station


def make_station_02(channel_nomenclature="default"):
    test2 = make_station_01(channel_nomenclature=channel_nomenclature)
    test2.id = "test2"
    test2.mth5_name = "test2.h5"
    test2.runs[0].raw_data_path = DATA_PATH.joinpath("test2.asc")
    nan_indices = {}
    for channel in test2.runs[0].channels:
        nan_indices[channel] = []
    test2.runs[0].nan_indices = nan_indices
    return test2


def make_station_03(channel_nomenclature="default"):
    """
    Here we create a synthetic station with multiple runs.  Rather than generate fresh
    synthetic data, we just reuse test1.asc for each run.
    Parameters
    ----------
    channel_nomenclature: str
        one of the keys from CHANNEL_MAPS dict in
        mt_metadata.transfer_functions.processing.aurora.channel_nomenclature
        Example values ["default", "lemi12", "lemi34", "phoenix123"]

    Returns
    -------
    station: SyntheticStation()
        All the info needed in order to create synthetic data.
    """
    channel_nomenclature_obj = ChannelNomenclature()
    channel_nomenclature_obj.keyword = channel_nomenclature
    EX, EY, HX, HY, HZ = channel_nomenclature_obj.unpack()
    station = SyntheticStation("test3")
    station.mth5_name = "test3.h5"
    channels = channel_nomenclature_obj.channels

    nan_indices = {}
    for ch in channels:
        nan_indices[ch] = []

    filters = {}
    for ch in channels:
        if ch in [EX, EY]:
            filters[ch] = [
                FILTERS["1x"].name,
            ]
        elif ch in [HX, HY, HZ]:
            filters[ch] = [FILTERS["10x"].name, FILTERS["0.1x"].name]

    run_001 = SyntheticRun(
        "001",
        raw_data_path=DATA_PATH.joinpath("test1.asc"),
        nan_indices=nan_indices,
        filters=filters,
        channel_nomenclature=channel_nomenclature,
        start="1980-01-01T00:00:00+00:00",
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
        channel_nomenclature=channel_nomenclature,
        start="1980-01-02T00:00:00+00:00",
    )

    for ch in channels:
        noise_scalars[ch] = 5.0
    run_003 = SyntheticRun(
        "003",
        raw_data_path=DATA_PATH.joinpath("test1.asc"),
        noise_scalars=noise_scalars,
        nan_indices=nan_indices,
        filters=filters,
        channel_nomenclature=channel_nomenclature,
        start="1980-01-03T00:00:00+00:00",
    )

    for ch in channels:
        noise_scalars[ch] = 10.0
    run_004 = SyntheticRun(
        "004",
        raw_data_path=DATA_PATH.joinpath("test1.asc"),
        noise_scalars=noise_scalars,
        nan_indices=nan_indices,
        filters=filters,
        channel_nomenclature=channel_nomenclature,
        start="1980-01-04T00:00:00+00:00",
    )

    run_001.filters = filters
    run_002.filters = filters
    run_003.filters = filters
    run_004.filters = filters

    station.runs = [run_001, run_002, run_003, run_004]

    return station


def make_station_04(channel_nomenclature="default"):
    """Just like station 01, but data are resampled to 8Hz"""
    channel_nomenclature_obj = ChannelNomenclature()
    channel_nomenclature_obj.keyword = channel_nomenclature
    EX, EY, HX, HY, HZ = channel_nomenclature_obj.unpack()
    station = SyntheticStation("test1")
    station.mth5_name = "test_04_8Hz.h5"

    run_001 = SyntheticRun(
        "001",
        raw_data_path=DATA_PATH.joinpath("test1.asc"),
        channel_nomenclature=channel_nomenclature,
        start=None,
        sample_rate=8.0,
    )
    run_001.nan_indices = {}

    filters = {}
    for ch in run_001.channels:
        if ch in [EX, EY]:
            filters[ch] = [
                FILTERS["1x"].name,
            ]
        elif ch in [HX, HY, HZ]:
            filters[ch] = [FILTERS["10x"].name, FILTERS["0.1x"].name]
    run_001.filters = filters

    station.runs = [
        run_001,
    ]

    return station


# def main():
#     make_station_04()
#
# if __name__ == "__main__":
#     main()
