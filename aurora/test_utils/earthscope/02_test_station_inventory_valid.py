"""
Iterate over rows of stage 1 output csv, selecting only rows where the name is of the form:
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
import requests
import time

from aurora.sandbox.mth5_helpers import get_experiment_from_obspy_inventory
from aurora.sandbox.mth5_helpers import mth5_from_experiment
from aurora.sandbox.mth5_helpers import enrich_channel_summary

from aurora.test_utils.earthscope.data_availability import DataAvailability
from aurora.test_utils.earthscope.data_availability import DataAvailabilityException
from aurora.test_utils.earthscope.data_availability import row_to_request_df
from aurora.test_utils.earthscope.data_availability import url_maker
from aurora.test_utils.earthscope.helpers import EXPERIMENT_PATH
from aurora.test_utils.earthscope.helpers import get_most_recent_summary_filepath
from aurora.test_utils.earthscope.helpers import get_summary_table_filename
from aurora.test_utils.earthscope.helpers import get_summary_table_schema
from aurora.test_utils.earthscope.helpers import restrict_to_mda
from aurora.test_utils.earthscope.helpers import SUMMARY_TABLES_PATH
from aurora.test_utils.earthscope.helpers import timestamp_now
from aurora.test_utils.earthscope.helpers import USE_CHANNEL_WILDCARDS
from mth5.mth5 import MTH5
from mth5.clients import FDSN

STAGE_ID = 2
KNOWN_NON_EARTHCSCOPE_STATIONS = ["FRD", ]
COVERAGE_DF_SCHEMA = get_summary_table_schema(STAGE_ID)


# CONFIG
MTH5_VERSION = "0.2.0"
VERBOSITY = 1
AUGMENT_WITH_EXISTING = True
USE_SKELETON = True # speeds up preparing the dataframe
N_PARTITIONS = 1
RAISE_EXCEPTION_IF_DATA_AVAILABILITY_EMPTY = True
MAX_TRIES = 3

if not USE_CHANNEL_WILDCARDS:
    DATA_AVAILABILITY = DataAvailability()


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


def prepare_dataframe_for_processing(source_csv=None, use_skeleton=USE_SKELETON):
    """
    Towards parallelization, I want to make the skeleton of the dataframe first, and then fill it in.
    Previously, we had added rows to the dataframe on the fly.

    Returns
    -------

    """
    skeleton_file = "02_skeleton.csv"
    if use_skeleton:
        df = pd.read_csv(skeleton_file)
        return df

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
            new_row["filename"] = ""
            new_row["filesize"] = ""
            new_row["num_channels_inventory"] = -1
            new_row["num_filterless_channels"] = -1
            new_row["filter_units_in_details"] = ""
            new_row["filter_units_out_details"]  = ""
            new_row["num_channels_h5"] = -1
            new_row["exception"] = ""
            new_row["error_message"] = ""
            # of course we should collect all the dictionaries first and then build the df,
            # this is inefficient, but tis a work in progress.
            coverage_df = coverage_df.append(new_row, ignore_index=True)

    # Now you have coverage df, but you need to uniquify it
    print(len(coverage_df))
    subset = ['network_id', 'station_id']
    ucdf = coverage_df.drop_duplicates(subset=subset, keep='first')
    print(len(ucdf))
    ucdf.to_csv("02_skeleton.csv", index=False)
    return ucdf

def get_augmented_channel_summary(m):
    channel_summary_df = m.channel_summary.to_dataframe()
    channel_summary_df = enrich_channel_summary(m, channel_summary_df, "num_filters")
    channel_summary_df = enrich_channel_summary(m, channel_summary_df, "filter_units_in")
    channel_summary_df = enrich_channel_summary(m, channel_summary_df, "filter_units_out")
    return channel_summary_df


def add_row_properties(expected_file_name, channel_summary_df, row):
    num_filterless_channels = len(channel_summary_df[channel_summary_df.num_filters == 0])
    n_ch_h5 = len(channel_summary_df)
    row["filename"] = expected_file_name.name
    row["filesize"] = expected_file_name.stat().st_size
    row["num_filterless_channels"] = num_filterless_channels
    aa = channel_summary_df.component.to_list()
    bb = channel_summary_df.num_filters.to_list()
    row["num_filter_details"] = str(dict(zip(aa, bb)))

    cc = channel_summary_df.filter_units_in.to_list()
    row["filter_units_in_details"] = str(dict(zip(aa, cc)))
    dd = channel_summary_df.filter_units_out.to_list()
    row["filter_units_out_details"] = str(dict(zip(aa, dd)))

    # new_row["num_channels_inventory"] = n_ch_inventory
    row["num_channels_h5"] = n_ch_h5
    row["exception"] = ""
    row["error_message"] = ""
    #return row

def get_from_iris():
    """Tool for multitry"""
    pass

def enrich_row(row):
    try:
        request_df = row_to_request_df(row, DATA_AVAILABILITY, verbosity=1, use_channel_wildcards=USE_CHANNEL_WILDCARDS,
                          raise_exception_if_data_availability_empty=RAISE_EXCEPTION_IF_DATA_AVAILABILITY_EMPTY)

    except Exception as e:
        print(f"{e}")
        row["num_channels_inventory"] = 0
        row["num_channels_h5"] = 0
        row["exception"] = e.__class__.__name__
        row["error_message"] = e.args[0]
        return row

    fdsn_object = FDSN(mth5_version=MTH5_VERSION)
    fdsn_object.client = "IRIS"
    expected_file_name = EXPERIMENT_PATH.joinpath(fdsn_object.make_filename(request_df))

    if expected_file_name.exists():
        print(f"Already have data for {row.network_id}-{row.station_id}")

        if AUGMENT_WITH_EXISTING:
            m = MTH5()
            m.open_mth5(expected_file_name)
            channel_summary_df = get_augmented_channel_summary(m)
            m.close_mth5()
            add_row_properties(expected_file_name, channel_summary_df, row)
    else:
        n_tries = 0
        while n_tries < MAX_TRIES:
            try:
                inventory, data = fdsn_object.get_inventory_from_df(request_df, data=False)
                n_ch_inventory = len(inventory.networks[0].stations[0].channels)
                row["num_channels_inventory"] = n_ch_inventory
                experiment = get_experiment_from_obspy_inventory(inventory)
                m = mth5_from_experiment(experiment, expected_file_name)
                m.channel_summary.summarize()
                channel_summary_df = get_augmented_channel_summary(m)
                m.close_mth5()
                add_row_properties(expected_file_name, channel_summary_df, row)
                n_tries = MAX_TRIES
            except Exception as e:
                print(f"{e}")
                row["num_channels_inventory"] = 0
                row["num_channels_h5"] = 0
                row["exception"] = e.__class__.__name__
                row["error_message"] = e.args[0]
                n_tries += 1
                if e.__class__.__name__ == "DataAvailabilityException":
                    n_tries = MAX_TRIES
    return row


def batch_download_metadata_v2(row_start=0, row_end=None):
    if USE_CHANNEL_WILDCARDS:
        availabile_channels = ["*Q*", "*F*", ]
    else:
        DATA_AVAILABILITY = DataAvailability()

    df = prepare_dataframe_for_processing()

    if row_end is None:
        row_end = len(df)
    df = df[row_start:row_end]

    if not N_PARTITIONS:
        enriched_df = df.apply(enrich_row, axis=1)
    else:
        import dask.dataframe as dd
        ddf = dd.from_pandas(df, npartitions=N_PARTITIONS)
        n_rows = len(df)
        df_schema = get_summary_table_schema(2)
        enriched_df = ddf.apply(enrich_row, axis=1, meta=df_schema).compute()

    coverage_csv = get_summary_table_filename(STAGE_ID)
    enriched_df.to_csv(coverage_csv, index=False)

def scan_data_availability_exceptions():
    """


    -------

    """
    coverage_csv = get_summary_table_filename(STAGE_ID)
    df = pd.read_csv(coverage_csv)
    sub_df = df[df["exception"]=="DataAvailabilityException"]
    #sub_df = df
    print(len(sub_df))
    for i, row in sub_df.iterrows():
        print(i)
        url = url_maker(row.network_id, row.station_id)
        response = requests.get(url)
        if response.status_code == 200:
            print('Web site exists')
            raise NotImplementedError
        else:
            print(f'Web site does not exist {response.status_code}')
    return

def review_results():
    now_str = timestamp_now()
    exceptions_summary_filebase = f"02_exceptions_summary_{now_str}.txt"
    exceptions_summary_filepath = SUMMARY_TABLES_PATH.joinpath(exceptions_summary_filebase)


    coverage_csv = get_summary_table_filename(STAGE_ID)
    df = pd.read_csv(coverage_csv)
    df = df[COVERAGE_DF_SCHEMA] #sort columns in desired order

    to_str_cols = ["network_id", "station_id", "exception"]
    for str_col in to_str_cols:
        df[str_col] = df[str_col].astype(str)

    with open(exceptions_summary_filepath, 'w') as f:
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
            print(unique_errors, "\n\n")
            msg = [f"{x}\n" for x in unique_errors]
            f.write("".join(msg) + "\n\n")
            exception_counts[exception_type] = len(exception_df)
            if exception_type == "IndexError":
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

def exception_analyser():
    """like batch_download, but will only try to pull selected row ids"""
    # batch_download_metadata_v2(row_start=853, row_end=854) #EM AB718 FDSNNoDataException
    #batch_download_metadata_v2(row_start=1399, row_end=1400) # ZU COR22 NotImplementedError
    # batch_download_metadata_v2(row_start=1337, row_end=1338) #
    #batch_download_metadata_v2(row_start=1784, row_end=1785) # ZU Y30 TypeError
    # batch_download_metadata_v2(row_start=613, row_end=614)  # EM OHM52 FDSNTimeoutException
    #batch_download_metadata_v2(row_start=1443, row_end=1444)  # 8P REU09 TypeError
    batch_download_metadata_v2(row_start=1487, row_end=1488)  # 8P REX11 IndexError


def main():
    # exception_analyser()
    # scan_data_availability_exceptions()
    t0 = time.time()
    batch_download_metadata_v2() # row_end=2)
    print(f"Total scraping time {time.time() - t0} using {N_PARTITIONS} partitions")
    review_results()
    total_time_elapsed = time.time() - t0
    print(f"Total scraping & review time {total_time_elapsed:.2f}s using {N_PARTITIONS} partitions")

if __name__ == "__main__":
    main()
