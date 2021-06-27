"""
TEST DATASET DEFINITIONS:
pkd_test_00:
  network = "BK"
    starttime = UTCDateTime("2004-09-28T00:00:00")
    endtime = UTCDateTime("2004-09-28T23:59:59")
    channel_codes = "LQ2,LQ3,LT1,LT2"
    channel_codes = "BQ2,BQ3,BT1,BT2"
"""


import datetime
import numpy as np
import pandas as pd

from obspy import UTCDateTime

from aurora.sandbox.xml_sandbox import get_response_inventory_from_iris
from aurora.sandbox.xml_sandbox import describe_inventory_stages
from mth5.timeseries.channel_ts import ChannelTS
from mth5.timeseries.run_ts import RunTS
from mth5_test_data.util import MTH5_TEST_DATA_DIR as DATA_DIR
#~/.cache/iris_mt/mth5_test_data/mth5_test_data

HEXY = ['hx','hy','ex','ey'] #default components list

def scan_network_for_nonconformity(inventory):
    """
    One off method for dealing with issues of historical data.
    Checks for the following:
    1. Channel Codes: Q2, Q3 --> Q1, Q2
    2. Field-type code: "T" instead of "F"
    3. Tesla to nT
    Parameters
    ----------
    inventory

    Returns
    -------

    """
    networks = inventory.networks
    for network in networks:
        for station in network:
            channel_codes = [x.code[1:3] for x in station.channels]
            print(channel_codes)
            #<ELECTRIC CHANNEL REMAP {Q2, Q3}-->{Q1, Q2}>
            if ("Q2" in channel_codes) & ("Q3" in channel_codes):
                print("Detected a likely non-FDSN conformant convnetion "
                      "unless there is a vertical electric dipole")
                print("Fixing Electric channel codes")
                #run the loop twice so we don't accidentally 
                #map Q3 to Q2 and Q2 to 
                for channel in station.channels:
                    if channel.code[1:3] == "Q2":
                        channel._code = f"{channel.code[0]}Q1"
                for channel in station.channels:
                    if channel.code[1:3] == "Q3":
                        channel._code = f"{channel.code[0]}Q2"
                print("HACK FIX ELECTRIC CHANNEL CODES COMPLETE")
            # </ELECTRIC CHANNEL REMAP {Q2, Q3}-->{Q1, Q2}>
            
            # <MAGNETIC CHANNEL REMAP {T1,T2,T3}-->{F1, F2, F3}>
            cond1 = "T1" in channel_codes
            cond2 = "T2" in channel_codes
            cond3 = "T3" in channel_codes
            if (cond1 or cond2 or cond3):
                print("Detected a likely non-FDSN conformant convnetion "
                      "unless there are Tidal data in this study")
                print("Fixing Magnetic channel codes")
                for channel in station.channels:
                    if channel.code[1] == "T":
                        channel._code = f"{channel.code[0]}F{channel.code[2]}"
                print("HACK FIX MAGNETIC CHANNEL CODES COMPLETE")
            # </MAGNETIC CHANNEL REMAP {T1,T2,T3}-->{F1, F2, F3}>

            #<Tesla to nanoTesla>
            for channel in station:
                response = channel.response
                for stage in response.response_stages:
                    print(f"{channel.code} {stage.stage_sequence_number} {stage.input_units}")
                    if stage.input_units=="T":
                        stage.input_units == "nT"
                        stage.stage_gain *= 1e-9
                #print(f"{channel}")
            # <Tesla to nanoTesla>
    return inventory

class TestDataSetConfig(object):
    """
    Note this is actually IRIS-specific.  We should create another type of test
    dataset for mth5
    Need:
    -iris_metadata_parameters
    -data_parameters (how to rover, or load from local)
    -a way to speecify station-channel, this config will only work for single stations.

    """
    def __init__(self):
        self.network = None
        self.station = None
        self.channels = None
        self.starttime = None
        self.endtime = None
        self.description = None
        self.dataset_id = None
        self.components_list = None #


    def get_inventory_from_iris(self, ensure_inventory_stages_are_named=True):

        inventory = get_response_inventory_from_iris(network=self.network,
                                                     station=self.station,
                                                     channel=self.channel_codes,
                                                     starttime=self.starttime,
                                                     endtime=self.endtime,
                                                     )
        inventory = scan_network_for_nonconformity(inventory)
        if ensure_inventory_stages_are_named:
            describe_inventory_stages(inventory, assign_names=True)
            # describe_inventory_stages(inventory, assign_names=False)

        return inventory

    def get_test_dataset(self):
        array_list = get_example_array_list(components_list=self.components_list,
                                            load_actual=True,
                                            station_id=self.station,
                                            component_station_label=False)
        mvts = RunTS(array_list=array_list)
        return mvts

    def get_data_via_rover(self):
        """
        Need
        1. Where does the rover-ed file end up?  that path needs to be accessible to load the data
        after it is generated
        Returns
        -------

        """
        pass

    def get_station_xml_filename(self, tag=""):
        """
        Placeholder in case we need to make many of these
        TODO: Modify so the path comes from the dataset_id, not the station_id...

        """
        filebase = f"{self.dataset_id}.xml"
        if tag:
            filebase = f"{tag}_{filebase}"
        target_folder = DATA_DIR.joinpath("iris",f"{self.network}")
        target_folder.mkdir(exist_ok=True)
        xml_filepath = target_folder.joinpath(filebase)
        return xml_filepath

    def save_xml(self, experiment, tag=""):
        """
        could probably use inventory or experiement
        Maybe even add a type-checker here
        if isinstance(xml_obj, inventory):
            tag="inventory"
        elif  isinstance(xml_obj, Experiment()):
            tag="experiment"

        Parameters
        ----------
        experiement

        Returns
        -------

        """
        output_xml_path = self.get_station_xml_filename(tag=tag)
        experiment.to_xml(output_xml_path)
        print(f"saved experiement to {output_xml_path}")
        return

#<CREATE TEST CONFIGS>
def make_test_configs():
    test_data_set_configs = {}

    #<pkd_test_00 Single station>
    test_data_set = TestDataSetConfig()
    test_data_set.dataset_id = "pkd_test_00"
    test_data_set.network = "BK"
    test_data_set.station = "PKD"
    test_data_set.starttime = UTCDateTime("2004-09-28T00:00:00")
    test_data_set.endtime = UTCDateTime("2004-09-28T23:59:59")
    #test_data_set.channel_codes = "LQ2,LQ3,LT1,LT2"
    test_data_set.channel_codes = "BQ2,BQ3,BT1,BT2"
    test_data_set.description = "2h of PKD data for 2004-09-28 midnight UTC until 0200"
    test_data_set.components_list = HEXY

    test_data_set_configs[test_data_set.dataset_id] = test_data_set
    #</pkd_test_00 Single station>

    # <sao_test_00 Single station>
    test_data_set = TestDataSetConfig()
    test_data_set.dataset_id = "sao_test_00"
    test_data_set.network = "BK"
    test_data_set.station = "SAO"
    test_data_set.starttime = UTCDateTime("2004-09-28T00:00:00")
    test_data_set.endtime = UTCDateTime("2004-09-28T23:59:59")
    #test_data_set.channel_codes = "LQ2,LQ3,LT1,LT2"
    test_data_set.channel_codes = "BQ2,BQ3,BT1,BT2"
    test_data_set.description = "2h of SAO data for 2004-09-28 midnight UTC until 0200"
    test_data_set.components_list = HEXY

    test_data_set_configs[test_data_set.dataset_id] = test_data_set
    # </sao_test_00 Single station>

    #<FAP>
    test_data_set = TestDataSetConfig()
    test_data_set.dataset_id = "fap_test"
    test_data_set.network = "EM"
    test_data_set.station = "FL001"
    test_data_set.starttime = None#UTCDateTime("2004-09-28T00:00:00")
    test_data_set.endtime = None#UTCDateTime("2004-09-28T23:59:59")
    test_data_set.channel_codes = "MFN"#BQ2,BQ3,BT1,BT2"
    test_data_set.description = "test of a fap xml"

    test_data_set_configs["fap_test"] = test_data_set
    # </FAP>

    # # <SYNTHETIC> ?Not needed?
    # dataset_id = "synthetic"
    # test_data_set = TestDataSetConfig()
    # test_data_set.dataset_id = dataset_id
    # test_data_set.network = "XX"
    # test_data_set.station = "sythetic_station_01"
    # test_data_set.starttime = UTCDateTime("1977-03-02T14:56:00")
    # test_data_set.endtime = None  # UTCDateTime("2004-09-28T23:59:59")
    # test_data_set.channel_codes = "LQ1,LQ2,LF1,LF2, LF3"
    # test_data_set.description = "emtf historical synthetic test dataset"
    #
    # test_data_set_configs[dataset_id] = test_data_set
    # # </SYNTHETIC>


    return test_data_set_configs


TEST_DATA_SET_CONFIGS = make_test_configs()

#</CREATE TEST CONFIGS>


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
def get_channel(component, station_id="", start=None, sampling_rate=None, load_actual=True,
                component_station_label=False):
    """
    One off - specifically for loading PKD and SAO data for May 24th spectral tests.
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
    instantites a list of Channel objects with data embedded.  This is used to create a
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
