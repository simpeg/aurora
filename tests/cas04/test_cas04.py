"""
Here is an integrated test that is aimed at Issue #31.  This test is also related to
several other issues and testing out several functionalities.

First, an mth5 is created for all available data.  This is accomplished by starting
with a station_xml file that was provided by Anna (and then modified by Tim).

Seem to be encountering an issue with mth5 not being 0.1.0

This tests the make_mth5_from_fdsnclient() method of MakeMTH5.

T

"""

from obspy import read_inventory
import pandas as pd
import pathlib

from aurora.general_helper_functions import TEST_PATH
from aurora.test_utils.dataset_definitions import TEST_DATA_SET_CONFIGS
from mth5.utils.helpers import initialize_mth5
from mth5.utils.helpers import read_back_data
from mth5.clients.helper_functions import channel_summary_to_make_mth5
from mth5.clients.make_mth5_rev_002 import MakeMTH5
#from mth5.clients.make_mth5 import MakeMTH5
from mt_metadata.timeseries.stationxml import xml_network_mt_survey
from mt_metadata.timeseries.stationxml import XMLInventoryMTExperiment


CAS04_PATH = TEST_PATH.joinpath("cas04")
DATA_PATH = CAS04_PATH.joinpath("data")
DATA_PATH.mkdir(exist_ok=True)

def xml_to_mth5(xml_path, h5_path="tmp.h5"):
    """
    Parameters
    ----------
    xml_path

    Returns
    -------

    """
    inventory0 = read_inventory(xml_path) #8P
    translator = XMLInventoryMTExperiment()
    experiment = translator.xml_to_mt(inventory_object=inventory0)
    mth5_obj = initialize_mth5(h5_path)
    mth5_obj.from_experiment(experiment)
    return mth5_obj

def make_cas04_data_for_processing(xml_path, h5_path="tmp.h5",
                                   generate_channel_summary=False,
                                   summary_csv="channel_summary.csv",
                                   active_runs=["a", ]):
    """
    This example is intended to be a template for working with XML files and
    associated metadata.  When an XML file is to be tested,
    Returns
    -------

    """
    #<CREATE MTH5 FROM XML AND SUMMARIZE DATA TO QUEUE>
    inventory0 = read_inventory(xml_path) #8P
    translator = XMLInventoryMTExperiment()
    experiment = translator.xml_to_mt(inventory_object=inventory0)
    mth5_obj = initialize_mth5(h5_path)
    mth5_obj.from_experiment(experiment)
    if generate_channel_summary:
        summary_df = mth5_obj.channel_summary
        summary_df.to_csv(summary_csv)
    else:
        summary_df = pd.read_csv(summary_csv)
    #</CREATE MTH5 FROM XML AND SUMMARIZE DATA TO QUEUE>

    #<TRANSFORM CHANNEL SUMMARY INTO REQUEST DF>
    if active_runs is not None:
        summary_df = summary_df[summary_df["run"].isin(active_runs)] #summary_df[0:5]
    request_df = channel_summary_to_make_mth5(summary_df, network="ZU")
    print(request_df)
    #</TRANSFORM CHANNEL SUMMARY INTO REQUEST DF>


    maker = MakeMTH5(mth5_version="0.1.0")
    print("FAILED FOR 0.2.0 with some other error")
    #inventory, streams = maker.get_inventory_from_df(request_df, data=False, client="IRIS")
    # inventory==inventory0??
    mth5_path = maker.make_mth5_from_fdsnclient(request_df, client="IRIS", path=DATA_PATH)

    print(f"success {mth5_path}")
    return mth5_path


def test_make_mth5():
    xml_path = "cas04_from_tim_20211203.xml"
    h5_path = DATA_PATH.joinpath("cas04.h5")
    mth5_path = make_cas04_data_for_processing(xml_path, h5_path=h5_path,
                                               generate_channel_summary=True,
                                               summary_csv="channel_summary.csv",
                                               active_runs=None)#["a", ])
    #mth5_path = DATA_PATH.joinpath("../backup/data/ZU_CAS04.h5")
    read_back_data(mth5_path, "CAS04", "a")
    read_back_data(mth5_path, "CAS04", "b")
    read_back_data(mth5_path, "CAS04", "c")
    read_back_data(mth5_path, "CAS04", "d")

#     """
#
#     consider the case where you have a station data locally, but you also have a
#     candidate remote reference .. we want a tool that can load the RR metadata and
#     identify time intervals that data are available simultaneously
#
#     """




def main():
    test_make_mth5()

    print("OK")

if __name__ == "__main__":
    main()