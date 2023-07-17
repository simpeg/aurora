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

from aurora.sandbox.mth5_helpers import build_request_df
from aurora.test_utils.earthscope.helpers import DataAvailability
from aurora.test_utils.earthscope.helpers import DATA_PATH
from aurora.test_utils.earthscope.helpers import get_most_recent_summary_filepath
from aurora.test_utils.earthscope.helpers import get_summary_table_filename
from aurora.test_utils.earthscope.helpers import restrict_to_mda
from aurora.test_utils.earthscope.helpers import USE_CHANNEL_WILDCARDS
from mth5.mth5 import MTH5
from mth5.clients import FDSN, MakeMTH5
from mt_metadata.transfer_functions.core import TF
from mt_metadata import TF_XML

STAGE_ID = 3


coverage_csv = get_summary_table_filename(STAGE_ID)
print(coverage_csv)


def initialize_mth5_df():
    coverage_df = pd.DataFrame(columns=["station_id", "network_id", "filename", "filesize"])
    return coverage_df



def batch_download_mth5(source_csv=None, results_csv=None):
    """

    :param xml_source_column:"data_xml_path" or "emtf_xml_path"
    specifies which of the two possible collections of xml files to use as source
    :return:
    """
    DATA_AVAILABILITY = DataAvailability()
    t0 = time.time()
    try:
        coverage_df = pd.read_csv(coverage_csv)
    except FileNotFoundError:
        coverage_df = initialize_mth5_df()

    if not source_csv:
        source_csv = get_most_recent_summary_filepath(2)
    source_df = pd.read_csv(source_csv)


    for i_row, row in source_df.iterrows():
        if isinstance(row.exception, str):
            print(f"Skipping row {i_row} for now, Exception {row.exception} was encounterd in metadata")
            continue
        if USE_CHANNEL_WILDCARDS:
            availabile_channels = ["*Q*", "*F*",]
        else:
            availabile_channels = DATA_AVAILABILITY.get_available_channels(
                network_id, station_id)

        request_df = build_request_df(row.network_id, row.station_id,
                                      channels=availabile_channels, start=None, end=None)
        #print(request_df)
        fdsn_object = FDSN(mth5_version='0.2.0')
        fdsn_object.client = "IRIS"
        if row.station_id == "ORF08":
            print("debug")
        # else:
        #     continue
        expected_file_name = DATA_PATH.joinpath(fdsn_object.make_filename(request_df))
        if expected_file_name.exists():
            print(f"Already have data for {row.station_id}-{row.network_id}")
            continue
        try:
            print(request_df)
            mth5_filename = fdsn_object.make_mth5_from_fdsn_client(request_df,
                                                                   interact=False,
                                                                   path=DATA_PATH)
            new_row = {"station_id": row.station_id,
                       "network_id": row.network_id,
                       "filename": mth5_filename,
                       "filesize": mth5_filename.stat().st_size,
                       "exception":"",
                       "error_message":""}
            coverage_df = coverage_df.append(new_row, ignore_index=True)
            coverage_df.to_csv(coverage_csv, index=False)
        except Exception as e:
            print("")
            new_row = {"station_id":row.station_id,
                       "network_id":row.network_id,
                       "filename":"", #expected_file_name
                       "filesize": "", #expected_file_name.stat().st_size,
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
    batch_download_mth5()
    #review_results()
    print("all done!")

if __name__ == "__main__":
    main()
