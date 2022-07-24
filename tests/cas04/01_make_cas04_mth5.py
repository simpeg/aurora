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

This is done via querying the IRIS metadata, but could be done starting with a
stationxml file in theory, and an example showing that workflow should be added in
future.

ToDo: Test make_all_stations() with mth5_version 0.1.0 and 0.2.0

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
START = "2020-06-02T19:00:00"
END = "2020-07-13T19:00:00"

# Test cast wide net (passes)
# START = "2000-06-02T19:00:00"
# END = "2023-07-13T19:00:00"

STATIONS = ["CAS04", "CAV07", "NVR11", "REV06"]
CHANNELS = [
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
    request_list: list

    """
    request_list = []
    for station_id in STATIONS:
        for channel_id in CHANNELS:
            request = [NETWORK, station_id, "", channel_id, START, END]
            request_list.append(request)
    return request_list


def make_all_stations(h5_path="all.h5", mth5_version="0.1.0", return_obj=False):
    """

    Parameters
    ----------
    h5_path
    mth5_version
    return_obj

    Returns
    -------

    """
    # Initialize mth5_maker
    maker = MakeMTH5(mth5_version=mth5_version)
    maker.client = "IRIS"

    # Make request list
    request_list = get_dataset_request_lists()
    print(f"Request List \n {request_list}")

    # Turn list into dataframe
    metadata_request_df = pd.DataFrame(request_list, columns=maker.column_names)
    print(f"metadata_request_df \n {metadata_request_df}")

    # Request the inventory information from IRIS
    inventory, streams = maker.get_inventory_from_df(metadata_request_df, data=False)

    # convert the inventory information to an mth5
    translator = XMLInventoryMTExperiment()
    experiment = translator.xml_to_mt(inventory_object=inventory)
    mth5_obj = initialize_mth5(h5_path)  # mode="a")
    mth5_obj.from_experiment(experiment)

    # get channel summary info
    mth5_obj.channel_summary.summarize()
    summary_df = mth5_obj.channel_summary.to_dataframe()

    # Transform channel_summary into request_df
    # TODO: Make this function run in PKD testing...
    request_df = channel_summary_to_make_mth5(summary_df, network=NETWORK)
    print(request_df)

    # Build the big mth5 with data
    maker = MakeMTH5(mth5_version=mth5_version)
    mth5_obj = maker.make_mth5_from_fdsnclient(
        request_df, client="IRIS", path=DATA_PATH, interact=True
    )

    if return_obj:
        return mth5_obj
    else:
        mth5_path = mth5_obj.filename
        mth5_obj.close_mth5()
        return mth5_path


def test_make_mth5(mth5_version="0.1.0"):
    """

    Returns
    -------
    mth5_path: string
        Where the built mth5 lives
    """
    mth5_path = make_all_stations(mth5_version=mth5_version)

    read_back_data(mth5_path, "CAS04", "a")
    read_back_data(mth5_path, "CAS04", "b")
    read_back_data(mth5_path, "CAS04", "c")
    read_back_data(mth5_path, "CAS04", "d")

    return mth5_path


def main():
    mth5_path = test_make_mth5(mth5_version="0.1.0")  # passes
    # mth5_path = test_make_mth5(mth5_version="0.2.0") #fails 10 Jul 2022
    return mth5_path


if __name__ == "__main__":
    main()
    print("OK")
