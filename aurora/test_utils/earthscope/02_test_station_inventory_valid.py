"""

Flow
Use stage 1 output csv

We will iterate over rows of the CSV, selecting only rows where the name is of the form:
18057859_EM_MH010.xml
uid_NETWORK_STATION.xml

For each such row, we make a list of stations that were identified
as self or RR

For every station in list:
    get metadata
    show number of channels
    any other pertinent information
"""



import numpy as np
import pandas as pd
import pathlib
import time

from matplotlib import pyplot as plt
from pathlib import Path

from aurora.sandbox.mth5_helpers import get_experiment_from_obspy_inventory
from aurora.sandbox.mth5_helpers import mth5_from_experiment

from aurora.test_utils.earthscope.helpers import build_request_df
from aurora.test_utils.earthscope.helpers import EXPERIMENT_PATH
from aurora.test_utils.earthscope.helpers import get_most_recent_review
from aurora.test_utils.earthscope.helpers import get_summary_table_filename
from aurora.test_utils.earthscope.helpers import load_data_availability_dfs
from mth5.mth5 import MTH5
from mth5.clients import FDSN, MakeMTH5
from mt_metadata.transfer_functions.core import TF
from mt_metadata import TF_XML

STAGE_ID = 2

AVAILABILITY_TABLE = load_data_availability_dfs()
coverage_csv = get_summary_table_filename(STAGE_ID)
GET_REMOTES_FROM = "spud_xml_review" # tf_xml


def initialize_metadata_df():
    """
    We want columns
    station_id
    network_id
    num_channels
    h5_path


    :return:
    """
    coverage_df = pd.DataFrame(columns=["station_id", "network_id", "filename", 
                                        "filesize",
                                        "num_channels_inventory",
                                        "num_channels_h5",
                                        "num_channels", "exception", "error_message"])
    return coverage_df



def batch_download_metadata(source_csv=None, results_csv=None):
    """

    :param xml_source_column:"data_xml_path" or "emtf_xml_path"
    specifies which of the two possible collections of xml files to use as source
    :return:
    """
    t0 = time.time()
    try:
        coverage_df = pd.read_csv(coverage_csv)
    except FileNotFoundError:
        coverage_df = initialize_metadata_df()

    if not source_csv:
        source_csv = get_most_recent_review(1)
    spud_df = pd.read_csv(source_csv)
    spud_df["data_xml_path_remotes"] = spud_df.data_xml_path_remotes.astype(str)

    xml_source = "data_xml_path"

    for i_row, row in spud_df.iterrows():
        if row[f"{xml_source}_error"] is True:
            print(f"Skipping {row.emtf_id} for now, tf not reading in")
            continue

        xml_path = pathlib.Path(row[xml_source])
        if "__" in xml_path.name:
            print(f"Skipping {row[xml_source]} for now, Station/network unknown")
            continue


        [xml_uid, network_id, station_id] = xml_path.stem.split("_")
        remotes = row.data_xml_path_remotes.split(",")
        if len(remotes)==1:
            if remotes[0] == "nan":
                remotes = []
        if remotes:
            print(f"remotes: {remotes} ")
        all_stations = remotes + [station_id,]

        for station in all_stations:
            availability_df = AVAILABILITY_TABLE[network_id]
            sub_availability_df = availability_df[availability_df["Station"] == station_id]
            availabile_channels = sub_availability_df['Channel'].unique()
            request_df = build_request_df(station, network_id,
                                          channels=availabile_channels, start=None, end=None)
            print(request_df)
            fdsn_object = FDSN(mth5_version='0.2.0')
            fdsn_object.client = "IRIS"

            expected_file_name = EXPERIMENT_PATH.joinpath(fdsn_object.make_filename(request_df))
            sub_coverage_df = coverage_df[coverage_df["filename"] == str(expected_file_name)]
            if len(sub_coverage_df):
                print(f"Already have data for {station}-{network_id}")
                print(f"{sub_coverage_df}")
                continue
            try:
                time.sleep(0.1)
                inventory, data = fdsn_object.get_inventory_from_df(request_df, data=False)
                n_ch_inventory = len(inventory.networks[0].stations[0].channels)
                experiment = get_experiment_from_obspy_inventory(inventory)
                mth5 = mth5_from_experiment(experiment, expected_file_name)
                mth5.channel_summary.summarize()
                channel_summary_df = mth5.channel_summary.to_dataframe()
                n_ch_h5 = len(channel_summary_df)

                new_row = {"station_id": station,
                           "network_id": network_id,
                           "filename": expected_file_name,
                           "filesize": expected_file_name.stat().st_size,
                           "num_channels_inventory":n_ch_inventory,
                           "num_channels_h5": n_ch_h5,
                           "exception":"",
                           "error_message":""}
                coverage_df = coverage_df.append(new_row, ignore_index=True)
                coverage_df.to_csv(coverage_csv, index=False)
            except Exception as e:
                print("")
                new_row = {"station_id":station,
                           "network_id":network_id,
                           "filename":"", #expected_file_name
                           "filesize": "", #expected_file_name.stat().st_size,
                           "num_channels_inventory":0,
                           "num_channels_h5": 0,
                           "exception":e.__class__.__name__,
                           "error_message":e.args[0]}
                coverage_df = coverage_df.append(new_row, ignore_index=True)
                coverage_df.to_csv(coverage_csv, index=False)

def review_results():
    coverage_csv = get_summary_table_filename(STAGE_ID)
    coverage_df = pd.read_csv(coverage_csv)
    print("OK")
    pass

def main():
    batch_download_metadata()
    review_results()

if __name__ == "__main__":
    main()