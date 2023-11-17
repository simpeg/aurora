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
ToDo: July 2023: Identify if CAV07 is the reason for the problem ... add an option to make individual stations
CAV07 is the reason the build failed, CAV07 fails metadata pull with IndexError: index 0 is out of bounds for axis 0 with size 0


"""

import pandas as pd

from aurora.general_helper_functions import get_test_path
from aurora.general_helper_functions import execute_subprocess
from aurora.sandbox.mth5_helpers import build_request_df

from mth5.clients import FDSN
from mth5.utils.helpers import read_back_data
from loguru import logger


# Define paths
TEST_PATH = get_test_path()
CAS04_PATH = TEST_PATH.joinpath("cas04")
DATA_PATH = CAS04_PATH.joinpath("data")
DATA_PATH.mkdir(exist_ok=True)
XML_PATH = CAS04_PATH.joinpath("cas04_from_tim_20211203.xml")

# Define args for data getter
NETWORK_ID = "8P"
STATION_IDS = [
    "CAS04",
    "CAV07",
    "NVR11",
    "REV06",
]
CHANNELS = [
    "LQE",
    "LQN",
    "LFE",
    "LFN",
    "LFZ",
]
START = "2020-06-02T19:00:00"
END = "2020-07-13T19:00:00"
# Test use very large time interval to build all (passes)
# START = "2000-06-02T19:00:00"
# END = "2023-07-13T19:00:00"


def make_merged_request_dataframe(
    station_ids=STATION_IDS,
    network_id=NETWORK_ID,
    channels=CHANNELS,
    start=START,
    end=END,
):
    """
    Consider moving this to sandbox
    Args:
        station_ids:
        network_id:
        channels:
        start:
        end:

    Returns:

    """
    df_list = []
    for station_id in station_ids:
        df = build_request_df(
            network_id, station_id, channels=channels, start=start, end=end
        )
        df_list.append(df)
    output_df = pd.concat(df_list)
    output_df.reset_index(inplace=True, drop=True)
    return output_df


def make_all_stations_individually(
    mth5_version="0.1.0",
):
    """
    Makes 1 h5 for each station in STATION_IDS
    Args:
        mth5_version:
    """
    for station_id in STATION_IDS:
        # request_df = build_request_df(NETWORK_ID, station_id, channels=["*F*", "*Q*", ], start=None, end=None)
        request_df = build_request_df(
            NETWORK_ID, station_id, channels=CHANNELS, start=None, end=None
        )
        fdsn_object = FDSN(mth5_version=mth5_version)
        fdsn_object.client = "IRIS"
        fdsn_object.make_mth5_from_fdsn_client(
            request_df, interact=False, path=DATA_PATH
        )
    return


def make_all_stations_together(
    mth5_version="0.1.0", return_obj=False, force_download=False
):
    """

    Parameters
    ----------
    mth5_version
    return_obj

    Returns
    -------

    """
    request_df = make_merged_request_dataframe()
    fdsn_object = FDSN(mth5_version=mth5_version)
    fdsn_object.client = "IRIS"

    expected_file_name = DATA_PATH.joinpath(fdsn_object.make_filename(request_df))
    download = force_download
    if expected_file_name.exists():
        logger.info(f"Already have data for {expected_file_name.name}")
        download = False
        mth5_filename = expected_file_name
        if force_download:
            download = True
    if download:
        logger.info("getting...", request_df)
        mth5_filename = fdsn_object.make_mth5_from_fdsn_client(
            request_df, interact=False, path=DATA_PATH
        )

    return mth5_filename


def test_make_mth5(mth5_version="0.1.0"):
    """

    Returns
    -------
    mth5_path: string
        Where the built mth5 lives
    """
    # make_all_stations_individually()
    mth5_path = make_all_stations_together(
        mth5_version=mth5_version, force_download=True
    )
    if mth5_version == "0.1.0":
        new_filepath = str(mth5_path).replace(".h5", "_v1.h5")
    elif mth5_version == "0.2.0":
        new_filepath = str(mth5_path).replace(".h5", "_v2.h5")
    cmd = f"mv {mth5_path} {new_filepath}"
    execute_subprocess(cmd)

    if mth5_version == "0.1.0":
        survey = None
    else:
        survey = "CONUS South"
    read_back_data(new_filepath, "CAS04", "a", survey=survey)
    read_back_data(new_filepath, "CAS04", "b", survey=survey)
    read_back_data(new_filepath, "CAS04", "c", survey=survey)
    read_back_data(new_filepath, "CAS04", "d", survey=survey)

    return new_filepath


def main():
    mth5_path = test_make_mth5(mth5_version="0.1.0")  # passes
    mth5_path = test_make_mth5(mth5_version="0.2.0")  # passes 29 Jul 2022
    return mth5_path


if __name__ == "__main__":
    main()
    logger.info("OK")
