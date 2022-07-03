"""
Here is a test aimed at building an h5 for CAS04 (see Issue#31)
This test is also related to several other issues and testing out several
functionalities, including:
- make_mth5_from_fdsnclient() method of MakeMTH5.

First, an mth5 is created that summarizes available data.

This is done  can be done via either
1. Starting with a station.xml file (provided by Anna or Tim).
This xml was being used in lieu of what is kept in the iris archive because what is
in the archive had errors.  An alternative approach would be to get the inventory
object from IRIS.
2. Querying the IRIS metadata.




ToDo: DEBUG: Seem to be encountering an issue with mth5 not being 0.1.0
ToDo: ISSUE: Consider the case where you have a station data locally, but you also
have a candidate remote reference .. we want a tool that can load the RR metadata and
identify time intervals that data are available simultaneously
ToDo: CAV07, NVR11, REV06

"""

import pandas as pd
import pathlib

from aurora.general_helper_functions import TEST_PATH
from aurora.sandbox.mth5_channel_summary_helpers import channel_summary_to_make_mth5
from aurora.test_utils.dataset_definitions import TEST_DATA_SET_CONFIGS
from mth5.utils.helpers import initialize_mth5
from mth5.utils.helpers import read_back_data
#from mth5.clients.make_mth5_rev_002 import MakeMTH5
from mth5.clients.make_mth5 import MakeMTH5
from mt_metadata.timeseries.stationxml import xml_network_mt_survey
from mt_metadata.timeseries.stationxml import XMLInventoryMTExperiment
from helper_functions import xml_to_mth5

#Define paths
CAS04_PATH = TEST_PATH.joinpath("cas04")
DATA_PATH = CAS04_PATH.joinpath("data")
DATA_PATH.mkdir(exist_ok=True)
XML_PATH = CAS04_PATH.joinpath("cas04_from_tim_20211203.xml")
NETWORK = "8P"
def get_dataset_request_lists():
    dataset_request_lists = {}
    station_id = "CAS04"
    LQE = [NETWORK, station_id, '', 'LQE', '2020-06-02T19:00:00', '2020-07-13T19:00:00']
    LQN = [NETWORK, station_id, '', 'LQN', '2020-06-02T19:00:00', '2020-07-13T19:00:00']
    BFE = [NETWORK, station_id, '', 'LFE', '2020-06-02T19:00:00', '2020-07-13T19:00:00']
    BFN = [NETWORK, station_id, '', 'LFN', '2020-06-02T19:00:00', '2020-07-13T19:00:00']
    BFZ = [NETWORK, station_id, '', 'LFZ', '2020-06-02T19:00:00', '2020-07-13T19:00:00']
    dataset_request_lists[station_id] = [LQE, LQN, BFE, BFN, BFZ,]
    station_id = "CAV07"
    LQE = [NETWORK, station_id, '', 'LQE', '2020-06-02T19:00:00', '2020-07-13T19:00:00']
    LQN = [NETWORK, station_id, '', 'LQN', '2020-06-02T19:00:00', '2020-07-13T19:00:00']
    BFE = [NETWORK, station_id, '', 'LFE', '2020-06-02T19:00:00', '2020-07-13T19:00:00']
    BFN = [NETWORK, station_id, '', 'LFN', '2020-06-02T19:00:00', '2020-07-13T19:00:00']
    BFZ = [NETWORK, station_id, '', 'LFZ', '2020-06-02T19:00:00', '2020-07-13T19:00:00']
    dataset_request_lists[station_id] = [LQE, LQN, BFE, BFN, BFZ,]
    station_id = "NVR11"
    LQE = [NETWORK, station_id, '', 'LQE', '2020-06-02T19:00:00', '2020-07-13T19:00:00']
    LQN = [NETWORK, station_id, '', 'LQN', '2020-06-02T19:00:00', '2020-07-13T19:00:00']
    BFE = [NETWORK, station_id, '', 'LFE', '2020-06-02T19:00:00', '2020-07-13T19:00:00']
    BFN = [NETWORK, station_id, '', 'LFN', '2020-06-02T19:00:00', '2020-07-13T19:00:00']
    BFZ = [NETWORK, station_id, '', 'LFZ', '2020-06-02T19:00:00', '2020-07-13T19:00:00']
    dataset_request_lists[station_id] = [LQE, LQN, BFE, BFN, BFZ,]
    station_id = "REV06"
    LQE = [NETWORK, station_id, '', 'LQE', '2020-06-02T19:00:00', '2020-07-13T19:00:00']
    LQN = [NETWORK, station_id, '', 'LQN', '2020-06-02T19:00:00', '2020-07-13T19:00:00']
    BFE = [NETWORK, station_id, '', 'LFE', '2020-06-02T19:00:00', '2020-07-13T19:00:00']
    BFN = [NETWORK, station_id, '', 'LFN', '2020-06-02T19:00:00', '2020-07-13T19:00:00']
    BFZ = [NETWORK, station_id, '', 'LFZ', '2020-06-02T19:00:00', '2020-07-13T19:00:00']
    dataset_request_lists[station_id] = [LQE, LQN, BFE, BFN, BFZ,]
    return dataset_request_lists


def make_all_stations(h5_path="all.h5", mth5_version='0.1.0', return_obj=False):
    maker = MakeMTH5(mth5_version=mth5_version)
    maker.client = "IRIS"

    request_lists_dict = get_dataset_request_lists()
    request_list = request_lists_dict["CAS04"]
    request_list += request_lists_dict["CAV07"]
    request_list += request_lists_dict["NVR11"]
    request_list += request_lists_dict["REV06"]

    print(f"Request List \n {request_list}")
    # Turn list into dataframe
    metadata_request_df =  pd.DataFrame(request_list, columns=maker.column_names)
    print(f"metadata_request_df \n {metadata_request_df}")

    # Request the inventory information from IRIS
    inventory, traces = maker.get_inventory_from_df(metadata_request_df, data=False)
    translator = XMLInventoryMTExperiment()
    experiment = translator.xml_to_mt(inventory_object=inventory)

    #Note m is a MakeMTH5 obj, not an MTH5
    mth5_obj = initialize_mth5(h5_path)# mode="a")
    mth5_obj.from_experiment(experiment)
    mth5_obj.channel_summary.summarize()

    summary_df = mth5_obj.channel_summary.to_dataframe()
    #summary_df = summary_df.loc[0:4] #restrict to a single run
    #tmp_mth5_obj.close_mth5()


    #<TRANSFORM CHANNEL SUMMARY INTO REQUEST DF>
    #if active_runs is not None:
    #    summary_df = summary_df[summary_df["run"].isin(active_runs)] #summary_df[0:5]
    request_df = channel_summary_to_make_mth5(summary_df, network=NETWORK)

    print(request_df)
    print("OK")
    maker = MakeMTH5(mth5_version=mth5_version)
    #print("FAILED FOR 0.2.0 with some other error")
    #inventory, streams = maker.get_inventory_from_df(request_df, data=False, client="IRIS")    # inventory==inventory0??
    mth5_obj = maker.make_mth5_from_fdsnclient(request_df, client="IRIS",
                                               path=DATA_PATH, interact=True)
    #SOLUTION 2:
    # port_metadata(source=mth5_obj, target=mth5_path)
    #print(f"success {mth5_path}")
    if return_obj:
        return mth5_obj
    else:
        mth5_path = mth5_obj.filename
        mth5_obj.close_mth5()
        return mth5_path

def make_cas04_data_for_processing(xml_path=None, h5_path="tmp.h5",
                                   active_runs=["a", ], mth5_version="0.1.0",
                                   return_obj=False):
    """
    This example is intended to be a template for working with XML files and
    associated metadata.  When an XML file is to be tested,

    Parameters
    ----------
    xml_path: Str or None
        Path to xml file to use for metdata definition.  If None the xml will be
        downloaded from IRIS
    h5_path: str or Path
        Where the data will be stored
    active_runs: list
        List of strings with run names, e.g. ["a", "b", "c", "d"]
    mth5_version: str
        One of "0.1.0", or "0.2.0"

    Returns
    -------


    """
    #CREATE MTH5 FROM XML AND SUMMARIZE DATA TO QUEUE
    if xml_path is not None:
        mth5_obj = xml_to_mth5(str(xml_path))
    else:
        print("get file from IRIS")
        maker = MakeMTH5(mth5_version='0.1.0')
        maker.client = "IRIS"

        # Generate data frame of FDSN Network, Station, Location, Channel, Startime, Endtime codes of interest
        CAS04LQE = ['8P', 'CAS04', '', 'LQE', '2020-06-02T19:00:00', '2020-07-13T19:00:00']
        CAS04LQN = ['8P', 'CAS04', '', 'LQN', '2020-06-02T19:00:00', '2020-07-13T19:00:00']
        CAS04BFE = ['8P', 'CAS04', '', 'LFE', '2020-06-02T19:00:00', '2020-07-13T19:00:00']
        CAS04BFN = ['8P', 'CAS04', '', 'LFN', '2020-06-02T19:00:00', '2020-07-13T19:00:00']
        CAS04BFZ = ['8P', 'CAS04', '', 'LFZ', '2020-06-02T19:00:00', '2020-07-13T19:00:00']

        request_list = [CAS04LQE, CAS04LQN, CAS04BFE, CAS04BFN, CAS04BFZ]
        print(f"Request List \n {request_list}")

        # Turn list into dataframe
        metadata_request_df =  pd.DataFrame(request_list, columns=maker.column_names)
        print(f"metadata_request_df \n {metadata_request_df}")

        # Request the inventory information from IRIS
        inventory, traces = maker.get_inventory_from_df(metadata_request_df, data=False)
        translator = XMLInventoryMTExperiment()
        experiment = translator.xml_to_mt(inventory_object=inventory)

        #Note m is a MakeMTH5 obj, not an MTH5
        mth5_obj = initialize_mth5(h5_path)# mode="a")
        mth5_obj.from_experiment(experiment)
        mth5_obj.channel_summary.summarize()

    summary_df = mth5_obj.channel_summary.to_dataframe()
    #summary_df = summary_df.loc[0:4] #restrict to a single run
    #tmp_mth5_obj.close_mth5()


    #<TRANSFORM CHANNEL SUMMARY INTO REQUEST DF>
    if active_runs is not None:
        summary_df = summary_df[summary_df["run"].isin(active_runs)] #summary_df[0:5]
    request_df = channel_summary_to_make_mth5(summary_df, network=NETWORK)
    print(request_df)


    #SOLUTION 1:
    # mth5_obj.populate_runs_from_request(request_df, client="IRIS")
    maker = MakeMTH5(mth5_version=mth5_version)
    #print("FAILED FOR 0.2.0 with some other error")
    #inventory, streams = maker.get_inventory_from_df(request_df, data=False, client="IRIS")    # inventory==inventory0??
    mth5_obj = maker.make_mth5_from_fdsnclient(request_df, client="IRIS",
                                                path=DATA_PATH, interact=True)
    #SOLUTION 2:
    # port_metadata(source=mth5_obj, target=mth5_path)
    #print(f"success {mth5_path}")
    if return_obj:
        return mth5_obj
    else:
        mth5_path = mth5_obj.filename
        mth5_obj.close_mth5()
        return mth5_path


def test_make_mth5_from_individual_runs():
    for run in ["a", "b", "c", "d"]:
        print(f"Testing RUN {run}")
        h5_run_path = DATA_PATH.joinpath(f"cas04_from_iris_20220615_{run}.h5")
        mth5_path = make_cas04_data_for_processing(xml_path=None, h5_path=h5_run_path,
                                                   active_runs=[run, ])
        read_back_data(mth5_path, "CAS04", run)
        print(f"success for run {run}!")
    return

def test_make_mth5_from_individual_multiple_runs():
    runs = ["a", "b", "c", "d"]
    print(f"Testing RUN {runs}")
    h5_run_path = DATA_PATH.joinpath(f"cas04_from_iris_20220615_{'_'.join(runs)}.h5")
    mth5_path = make_cas04_data_for_processing(xml_path=None, h5_path=h5_run_path,
                                               active_runs=runs)
    for run in runs:
        read_back_data(mth5_path, "CAS04", run)
        print(f"success for run {run}!")
    return
def test_make_mth5():
    """
    WARNING: The returned variable is ci
    Returns
    -------

    """
    all_h5_path = make_all_stations()
    print(f"ALL data in {all_h5_path}")
    import pdb
    print("pdb")
    pdb.set_trace()
    #test_make_mth5_from_individual_runs()
    test_make_mth5_from_individual_multiple_runs()
    h5_path = DATA_PATH.joinpath("cas04_from_iris_20220615.h5")

    import pdb
    print("pdb")
    pdb.set_trace()
    # h5_path = DATA_PATH.joinpath("cas04.h5")
    # mth5_path = make_cas04_data_for_processing(xml_path=XML_PATH, h5_path=h5_path,
    #                                            active_runs=None)#["a", ])
    #mth5_path = DATA_PATH.joinpath("../backup/data/ZU_CAS04.h5")
    read_back_data(mth5_path, "CAS04", "a")
    read_back_data(mth5_path, "CAS04", "b")
    read_back_data(mth5_path, "CAS04", "c")
    read_back_data(mth5_path, "CAS04", "d")
    print("WARNING: The path being returned is not the path to the XML-based mth5")
    print("The metadata are coming from IRIS server - not the XML")
    return mth5_path


def run_tests():
    make_mth5_from_scratch = True
    if make_mth5_from_scratch:
        mth5_path = test_make_mth5()
    else:
        mth5_path = DATA_PATH.joinpath("8P_CAS04.h5")#../backup/data/
    return mth5_path

def main():
    mth5_path = run_tests()
    return mth5_path


if __name__ == "__main__":
    main()
    print("OK")
