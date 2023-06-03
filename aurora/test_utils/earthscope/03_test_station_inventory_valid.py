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
from mth5.mth5 import MTH5
from mth5.clients import FDSN, MakeMTH5
from mt_metadata.transfer_functions.core import TF
from mt_metadata import TF_XML


    



def metadata_check(request_df):
    fdsn_object = FDSN(mth5_version='0.2.0')
    fdsn_object.client = "IRIS"
    inv = fdsn_object.get_inventory_from_df(request_df, data=False)
    experiment = get_experiment_from_obspy_inventory(inv[0])
    mth5 = mth5_from_experiment(experiment, 'qq.h5')
    mth5.channel_summary.summarize()
    channel_summary_df = mth5.channel_summary.to_dataframe()
    return channel_summary_df

#


data_coverage_csv_name = "local_data_coverage.csv"
data_coverage_csv_path = DATA_PATH.joinpath(data_coverage_csv_name)
GET_REMOTES_FROM = "spud_xml_review" # tf_xml


from aurora.test_utils.earthscope.helpers import get_most_recent_review

STAGE_ID = 2
def initialize_metadata_df():
    local_metadata_coverage_df = pd.DataFrame(columns=["station_id", "network_id", "filename", "filesize"])
    pass



def review_spud_tfs(xml_sources=["emtf_xml_path", "data_xml_path"],
                    source_csv=None,
                    results_csv=None):
    """

    :param xml_source_column:"data_xml_path" or "emtf_xml_path"
    specifies which of the two possible collections of xml files to use as source
    :return:
    """
    t0 = time.time()
    if not source_csv:
        source_csv = get_most_recent_review(1)
    source_df = pd.read_csv(source_csv)

    local_data_coverage_df = pd.read_csv(data_coverage_csv_path)

    xml_source = "data_xml_path"
    spud_csv_name = "spud_xml_review_2023-05-29_15:08:25.csv"
    spud_csv_path = SPUD_XML_PATH.joinpath(spud_csv_name)
    spud_df = pd.read_csv(spud_csv_path)

    for i_row, row in spud_df.iterrows():
        if row[f"{xml_source}_error"] is True:
            print(f"Skipping {row} for now, tf not reading in")
            continue

        xml_path = pathlib.Path(row[xml_source])
        if "__" in xml_path.name:
            print(f"Skipping {row[xml_source]} for now, Station/network unknown")
            continue


        [xml_uid, network_id, station_id] = xml_path.stem.split("_")
        extract remotes
        if GET_REMOTES_FROM == "tf_xml":
            tf = load_xml_tf(xml_path)
            rr_type = get_rr_type(tf)
            remotes = get_remotes_from_tf(tf)
        elif GET_REMOTES_FROM == "spud_xml_review":
            remotes = row.data_xml_path_remotes.split(",")
        if remotes:
            print(f"remotes: {remotes} ")
        all_stations = remotes + [station_id,]

        for station in all_stations:
            request_df = build_request_df([station,], network_id, start=None, end=None)
            print(request_df)
            fdsn_object = FDSN(mth5_version='0.2.0')
            fdsn_object.client = "IRIS"

            expected_file_name = EXPERIMENT_PATH.joinpath(fdsn_object.make_filename(request_df))
            sub_coverage_df = local_data_coverage_df[local_data_coverage_df["filename"] == str(expected_file_name)]
            if len(sub_coverage_df):
                print(f"Already have data for {station}-{network_id}")
                print(f"{sub_coverage_df}")
                continue
            try:
                mth5_filename = fdsn_object.make_mth5_from_fdsn_client(request_df,
                                                                       interact=False,
                                                                       path=DATA_PATH)
                new_row = {"station_id": station,
                           "network_id": network_id,
                           "filename": mth5_filename,
                           "filesize": mth5_filename.stat().st_size,
                           "exception":"",
                           "error_message":""}
                local_data_coverage_df = local_data_coverage_df.append(new_row, ignore_index=True)
                local_data_coverage_df.to_csv(local_data_coverage_csv, index=False)
            except Exception as e:
                print("")
                new_row = {"station_id":station,
                           "network_id":network_id,
                           "filename":"", #expected_file_name
                           "filesize": "", #expected_file_name.stat().st_size,
                           "exception":e.__class__.__name__,
                           "error_message":e.args[0]}
                local_data_coverage_df = local_data_coverage_df.append(new_row, ignore_index=True)
                local_data_coverage_df.to_csv(local_data_coverage_csv, index=False)
            # ADD A ROW TO A DF AS WE GO
        #
        #
        #     print("get metadata")
        #     print("check what the filename should be")
        #     print("Check if the file exists")
        #     print("check if the file contains all the data")
        #     print("if not, pull new data, and add to the file if it exists")

    #
    #     print("NOW get the RRS")
    #
    # if controls["review_spud_xmls_01"]:
    #     results_df = review_spud_tfs(xml_source=SPUD_XML_COLUMN)
    # else:
    #     results_df = pd.read_csv(SPUD_XML_REVIEW_CSV_01)
    # results_df = results_df[results_df.error==False]
    # results_df.reset_index(drop=True, inplace=True)
    # results_df2 = get_mth5_data(results_df)
    # grouper = results_df2.groupby("survey")
    # # get data
    #
    # print("OK")
    # for i_row, row in SPUD_DF.iterrows():
    #     xml_path = pathlib.Path(row[SPUD_XML_COLUMN])
    #     spud_tf = load_xml_tf(xml_path)
    #     print(row[SPUD_XML_COLUMN])
    # pass

def main():
    batch_download()

if __name__ == "__main__":
    main()