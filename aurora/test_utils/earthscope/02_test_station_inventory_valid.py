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
from aurora.test_utils.earthscope.helpers import EXPERIMENT_PATH
from aurora.test_utils.earthscope.helpers import get_most_recent_summary_filepath
from aurora.test_utils.earthscope.helpers import get_summary_table_filename
from aurora.test_utils.earthscope.helpers import get_summary_table_schema
from aurora.test_utils.earthscope.helpers import restrict_to_mda
from aurora.test_utils.earthscope.helpers import timestamp_now
from aurora.test_utils.earthscope.helpers import USE_CHANNEL_WILDCARDS
from mth5.mth5 import MTH5
from mth5.clients import FDSN, MakeMTH5
from mt_metadata.transfer_functions.core import TF
from mt_metadata import TF_XML

STAGE_ID = 2

KNOWN_NON_EARTHCSCOPE_STATIONS = ["FRD", ]


COVERAGE_DF_SCHEMA = get_summary_table_schema(2)

def initialize_metadata_df():
    """ """
    column_names = list(COVERAGE_DF_SCHEMA.keys())
    coverage_df = pd.DataFrame(columns=column_names)
    return coverage_df

def already_in_df(df, network_id, station_id):
    cond1 = df.network_id.isin([network_id, ])
    cond2 = df.station_id.isin([station_id, ])
    sub_df = df[cond1 & cond2]
    return len(sub_df)


def analyse_station_id(station_id):
    """
    Helper function that is not very robust, but specific to handling station_ids in the 2023 earthscope tests,
    in particular, parsing non-schema defined reference stations from SPUD TF XML.

    Parameters
    ----------
    station_id

    Returns
    -------

    """
    if station_id in KNOWN_NON_EARTHCSCOPE_STATIONS:
        print(f"skipping {station_id} -- it's not an earthscope station")
        return None
    if len(station_id) == 0:
        print("NO Station ID")
        return None
    if len(station_id) == 1:
        print(f"This ust be a typo or something station_id={station_id}")

        return None
    elif len(station_id) == 3:
        print(f"Probably forgot to archive the TWO-CHAR STATE CODE onto station_id {station_id}")
    elif len(station_id) == 4:
        print(f"?? Typo in station_id {station_id}??")
    elif len(station_id) > 5:
        print(f"run labels probably tacked onto station_id {station_id}")
        print("Can we confirm that FDSN has a max 5CHAR station ID code???")
        station_id = station_id[0:5]
        print(f"Setting to first 5 chars: {station_id}")
    elif len(station_id)==5:
        pass # looks normal
    else:
        print(f"Havent encountered case len(station_id)={len(station_id)}")
        raise NotImplementedError
    return station_id

def batch_download_metadata(source_csv=None,
                            results_csv=None,
                            append_rows_for_existing=False,
                            verbosity=1,
                            mth5_version="0.2.0"):
    """

    Parameters
    ----------
    source_csv
    results_csv
    append_rows_for_existing
    verbosity

    Returns
    -------

    """
    DATA_AVAILABILITY = DataAvailability()
    t0 = time.time()
    try:
        coverage_csv = get_summary_table_filename(STAGE_ID)
        coverage_df = pd.read_csv(coverage_csv)
    except FileNotFoundError:
        coverage_df = initialize_metadata_df()

    if not source_csv:
        source_csv = get_most_recent_summary_filepath(1)
    spud_df = pd.read_csv(source_csv)
    spud_df = restrict_to_mda(spud_df)
    print(f"Restricting spud_df to mda (Earthscope) entries: {len(spud_df)} rows")



    xml_source = "data"

    # Careful in this loop, it is associated with one station (that has a TF, but may have many remotes)
    for i_row, row in spud_df.iterrows():

        # Ignore XML that cannot be read
        if row[f"{xml_source}_error"] is True:
            print(f"Skipping {row.emtf_id} for now, tf not reading in")
            continue

        # Sort out remotes
        remotes = row.data_remotes.split(",")
        if len(remotes)==1:
            if remotes[0] == "nan":
                remotes = []
        if remotes:
            print(f"remotes: {remotes} ")

        all_stations = [row.station_id,] + remotes
        network_id = row.network_id
        for original_station_id in all_stations:
            station_id = analyse_station_id(original_station_id)
            if not station_id:
                continue

            new_row = {"station_id": station_id,
                       "network_id": network_id,
                       "emtf_id": row.emtf_id,
                       "data_id": row.data_id,
                       "data_xml_filebase": row.data_xml_filebase}
            if USE_CHANNEL_WILDCARDS:
                availabile_channels = ["*Q*", "*F*",]
            else:
                availabile_channels = DATA_AVAILABILITY.get_available_channels(
                    row.network_id, station_id)
            request_df = build_request_df(network_id, station_id,
                                          channels=availabile_channels, start=None, end=None)
            if verbosity > 1:
                print(f"request_df: \n {request_df}")
            fdsn_object = FDSN(mth5_version=mth5_version)
            fdsn_object.client = "IRIS"

            expected_file_name = EXPERIMENT_PATH.joinpath(fdsn_object.make_filename(request_df))

            if expected_file_name.exists():
                print(f"Already have data for {network_id}-{station_id}")
                if already_in_df(coverage_df, network_id, station_id):
                    continue
                if append_rows_for_existing:
                    m = MTH5()
                    m.open_mth5(expected_file_name)
                    channel_summary_df = m.channel_summary.to_dataframe()
                    n_ch_h5 = len(channel_summary_df)
                    m.close_mth5()
                    new_row["filename"] = expected_file_name.name
                    new_row["filesize"] = expected_file_name.stat().st_size
                    #new_row["num_channels_inventory"] = n_ch_inventory
                    new_row["num_channels_h5"] = n_ch_h5
                    #new_row["exception"] = ""
                    #new_row["error_message"] = ""
                    coverage_df = coverage_df.append(new_row, ignore_index=True)
                continue

            # Avoid duplication of already tried cases:
            if already_in_df(coverage_df, network_id, station_id):
                print(f"Already tried getting data for {network_id}-{station_id}")
                print(f"Because this is BELOW checking filename exists, to encounter this case, it must be true that the last attempt failed ...")
                print("SKIPPING IT FOR NOW")
                continue

            try:
                # time.sleep(0.1)
                inventory, data = fdsn_object.get_inventory_from_df(request_df, data=False)
                n_ch_inventory = len(inventory.networks[0].stations[0].channels)
                experiment = get_experiment_from_obspy_inventory(inventory)
                mth5 = mth5_from_experiment(experiment, expected_file_name)
                mth5.channel_summary.summarize()
                channel_summary_df = mth5.channel_summary.to_dataframe()
                n_ch_h5 = len(channel_summary_df)
                # ? do we need to close this object afterwards ?
                new_row["filename"] = expected_file_name.name
                new_row["filesize"] = expected_file_name.stat().st_size
                new_row["num_channels_inventory"] = n_ch_inventory
                new_row["num_channels_h5"] = n_ch_h5
                new_row["exception"] = ""
                new_row["error_message"] = ""
            except Exception as e:
                print(f"{e}")
                new_row["filename"] = ""
                new_row["filesize"] =  ""
                new_row["num_channels_inventory"] = 0
                new_row["num_channels_h5"] = 0
                new_row["exception"] = e.__class__.__name__
                new_row["error_message"] = e.args[0]
            coverage_df = coverage_df.append(new_row, ignore_index=True)
            coverage_df.to_csv(coverage_csv, index=False)


def add_filters_info():
    pass

def review_results():
    now_str = timestamp_now()
    fname = f"02_exceptions_summary_{now_str}.txt"

    coverage_csv = get_summary_table_filename(STAGE_ID)
    df = pd.read_csv(coverage_csv)
    df = df[COVERAGE_DF_SCHEMA] #sort columns in desired order

    to_str_cols = ["network_id", "station_id", "exception"]
    for str_col in to_str_cols:
        df[str_col] = df[str_col].astype(str)

    with open(fname, 'w') as f:
        msg = "*** EXCEPTIONS SUMMARY *** \n\n"
        print(msg)
        f.write(msg)

        exception_types = df.exception.unique()
        exception_types = [x for x in exception_types if x!="nan"]
        msg = f"Identified {len(exception_types)} exception types\n {exception_types}\n\n"
        print(msg)
        f.write(msg)


        exception_counts = {}
        for exception_type in exception_types:
            exception_df = df[df.exception == exception_type]
            n_exceptions = len(exception_df)
            unique_errors = exception_df.error_message.unique()
            n_unique_errors = len(unique_errors)
            msg = f"{n_exceptions} instances of {exception_type}, with {n_unique_errors} unique error(s)\n"
            print(msg)
            f.write(msg)
            print(unique_errors)
            msg = [f"{x}\n" for x in unique_errors]
            f.write("".join(msg) + "\n\n")
            exception_counts[exception_type] = len(exception_df)
            if exception_type=="IndexError":
                exception_df.to_csv("02_do_these_exist.csv", index=False)

        grouper = df.groupby(["network_id", "station_id"])
        msg = f"\n\nThere were {len(grouper)} unique network-station pairs in {len(df)} rows\n\n"
        print(msg)
        f.write(msg)
        print(exception_counts)
        f.write(str(exception_counts))
        msg = f"TOTAL #Exceptions {np.array(list(exception_counts.values())).sum()} of {len(df)} Cases"
        print(msg)
        f.write(msg)
    return

def main():
    t0 = time.time()
    # Normal usage: complete run, will not in-fill with info from existing
    batch_download_metadata()

    # Use when part of data already here on disk
    # This will be nearly complete, but does not fill out the n_ch_
    # batch_download_metadata(append_rows_for_existing=True)

    print(f"Total scraping time {time.time() - t0}")
    review_results()
    total_time_elapsed = time.time() - t0
    print(f"Total scraping & review time {total_time_elapsed:.2f}s")

if __name__ == "__main__":
    main()