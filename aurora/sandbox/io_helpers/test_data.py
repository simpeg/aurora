import datetime
import numpy as np
import pandas as pd


from mth5.timeseries.channel_ts import ChannelTS
from mth5.timeseries.run_ts import RunTS
from mth5_test_data.util import MTH5_TEST_DATA_DIR as DATA_DIR
#~/.cache/iris_mt/mth5_test_data/mth5_test_data

HEXY = ['hx','hy','ex','ey'] #default components list



class TestDataHelper(object):
    def __init__(self, **kwargs):
        self.dataset_id = kwargs.get("dataset_id")

    def load_df(self, dataset_id=None):
        if dataset_id is None:
            dataset_id = self.dataset_id

        if dataset_id == "pkd_test_00":
            source_data_path = DATA_DIR.joinpath("iris/BK/2004/ATS")
            merged_h5 = source_data_path.joinpath("pkd_sao_272_00.h5")
            df = pd.read_hdf(merged_h5, "pkd")
            return df
        if dataset_id == "sao_test_00":
            source_data_path = DATA_DIR.joinpath("iris/BK/2004/ATS")
            merged_h5 = source_data_path.joinpath("pkd_sao_272_00.h5")
            df = pd.read_hdf(merged_h5, "sao")
            return df

        if dataset_id == "PKD_SAO_2004_272_00-2004_272_02":
            source_data_path = DATA_DIR.joinpath("iris/BK/2004/ATS")
            merged_h5 = source_data_path.joinpath("pkd_sao_272_00.h5")
            pkd_df = pd.read_hdf(merged_h5, "pkd")
            sao_df = pd.read_hdf(merged_h5, "sao")
            return sao_df

    def load_channel(self, station, component):
        if self.dataset_id == "PKD_SAO_2004_272_00-2004_272_02":
            source_data_path = DATA_DIR.joinpath("iris/BK/2004/ATS")
            merged_h5 = source_data_path.joinpath("pkd_sao_272_00.h5")
            df = pd.read_hdf(merged_h5, key=f"{component}_{station.lower()}")
            return df.values




DEFAULT_SAMPLING_RATE = 40.0
DEFAULT_START_TIME = datetime.datetime(2004, 9, 28, 0, 0, 0)
def get_channel(component, station_id="", start=None, sampling_rate=None,
                load_actual=True, component_station_label=False):
    """
    One off - specifically for loading PKD and SAO data for spectral tests.
    Parameters
    ----------
    component
    station_id
    load_actual

    Returns
    -------

    """
    test_data_helper = TestDataHelper(dataset_id="PKD_SAO_2004_272_00-2004_272_02")

    if component[0]=='h':
        ch = ChannelTS('magnetic')
    elif component[0]=='e':
        ch = ChannelTS('electric')

    if sampling_rate is None:
        print(f"no sampling rate given, using default {DEFAULT_SAMPLING_RATE}")
        sampling_rate = DEFAULT_SAMPLING_RATE
    ch.sample_rate = sampling_rate

    if start is None:
        print(f"no start time given, using default {DEFAULT_START_TIME}")
        start = DEFAULT_START_TIME
    ch.start = start

    print("insert ROVER call here to access PKD, date, interval")
    print("USE this to load the data to MTH5")
    #https: // github.com / kujaku11 / mth5 / blob / master / examples / make_mth5_from_z3d.py
    if load_actual:
        time_series = test_data_helper.load_channel(station_id, component)
    else:
        N = 288000
        time_series = np.random.randn(N)
    ch.ts = time_series

    ch.station_metadata.id = station_id

    ch.run_metadata.id = "001"#'MT001a'
    if component_station_label:
        component_string = "_".join([component,station_id,])
        ch.component = component_string
    else:
        ch.component = component

    return ch



def get_example_array_list(components_list=None, load_actual=True, station_id=None,
                           component_station_label=False):
    """
    instantites a list of Channel objects with data embedded.
    This is used to pass data to mth5 objects when creating test datasets.

    Parameters
    ----------
    components_list
    load_actual
    station_id
    component_station_label

    Returns
    -------

    """
    array_list = []
    for component in components_list:
        channel = get_channel(component,
                              station_id=station_id,
                              load_actual=load_actual,
                              component_station_label=component_station_label)
        array_list.append(channel)
    return array_list




def get_example_data(components_list=HEXY,
                     load_actual=True,
                     station_id=None,
                     component_station_label=False):
    array_list = get_example_array_list(components_list=components_list,
                                        load_actual=load_actual,
                                        station_id=station_id,
                                        component_station_label=component_station_label)
    mvts = RunTS(array_list=array_list)
    return mvts


def main():
    print("hi")

if __name__=="__main__":
    main()