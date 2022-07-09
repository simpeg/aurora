"""
Here is a test aimed at building an h5 for CAS04 (see Issue#31)

The station CAS04 was processed with several different RR stations.  An archived
image of the EMTF results can be found here:
https://user-images.githubusercontent.com/8312937/142910549-52de8007-62bf-407c-816d-a81bed08f298.png

The reference stations were: {CAV07, NVR11, REV06}.


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

from aurora.general_helper_functions import TEST_PATH
from aurora.sandbox.mth5_channel_summary_helpers import channel_summary_to_make_mth5
from mth5.utils.helpers import initialize_mth5
from mth5.utils.helpers import read_back_data

# from mth5.clients.make_mth5_rev_002 import MakeMTH5
from mth5.clients.make_mth5 import MakeMTH5
from mt_metadata.timeseries.stationxml import XMLInventoryMTExperiment
from helper_functions import xml_to_mth5

# Define paths
CAS04_PATH = TEST_PATH.joinpath("cas04")
DATA_PATH = CAS04_PATH.joinpath("data")
DATA_PATH.mkdir(exist_ok=True)
XML_PATH = CAS04_PATH.joinpath("cas04_from_tim_20211203.xml")

# Define args for data getter
NETWORK = "8P"
CAS04_START = "2020-06-02T19:00:00"
CAS04_END = "2020-07-13T19:00:00"
STATION_LIST = ["CAS04", "CAV07", "NVR11", "REV06"]
CHANNEL_LIST = [
    "LQE",
    "LQN",
    "LFE",
    "LFN",
    "LFZ",
]


def get_dataset_request_lists():
    """

    Returns
    -------
    dataset_request_lists: dict

    """
    station_list = ["CAS04", "CAV07", "NVR11", "REV06"]
    # dataset_request_lists = {}
    request_list = []
    for station_id in station_list:
        for channel_id in CHANNEL_LIST:
            request = [NETWORK, station_id, "", channel_id, CAS04_START, CAS04_END]
            request_list.append(request)
    return request_list


def make_all_stations(h5_path="all.h5", mth5_version="0.1.0", return_obj=False):
    maker = MakeMTH5(mth5_version=mth5_version)
    maker.client = "IRIS"

    request_list = get_dataset_request_lists()
    print(f"Request List \n {request_list}")
    # Turn list into dataframe
    metadata_request_df = pd.DataFrame(request_list, columns=maker.column_names)
    print(f"metadata_request_df \n {metadata_request_df}")

    # Request the inventory information from IRIS
    inventory, traces = maker.get_inventory_from_df(metadata_request_df, data=False)
    translator = XMLInventoryMTExperiment()
    experiment = translator.xml_to_mt(inventory_object=inventory)

    mth5_obj = initialize_mth5(h5_path)  # mode="a")
    mth5_obj.from_experiment(experiment)
    mth5_obj.channel_summary.summarize()

    summary_df = mth5_obj.channel_summary.to_dataframe()

    # TRANSFORM CHANNEL SUMMARY INTO REQUEST DF
    request_df = channel_summary_to_make_mth5(summary_df, network=NETWORK)

    print(request_df)
    print("OK")
    maker = MakeMTH5(mth5_version=mth5_version)
    # print("FAILED FOR 0.2.0 with some other error")
    # inventory, streams = maker.get_inventory_from_df(request_df,
    # data=False, client="IRIS")    # inventory==inventory0??
    mth5_obj = maker.make_mth5_from_fdsnclient(
        request_df, client="IRIS", path=DATA_PATH, interact=True
    )

    if return_obj:
        return mth5_obj
    else:
        mth5_path = mth5_obj.filename
        mth5_obj.close_mth5()
        return mth5_path


def test_make_mth5():
    """
    WARNING: The returned variable is ci
    Returns
    -------

    """
    mth5_path = make_all_stations()
    print(f"ALL data in {mth5_path}")

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
        mth5_path = DATA_PATH.joinpath("8P_CAS04.h5")  # ../backup/data/
    return mth5_path


def main():
    mth5_path = run_tests()
    return mth5_path


if __name__ == "__main__":
    main()
    print("OK")
