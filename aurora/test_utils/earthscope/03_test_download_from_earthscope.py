"""

Flow:
Use stage 2 output csv
For each such row,
    extract the network/station_id (if the metadata exist)
    download data

Handy steps for debugging:
# metadata_local = EXPERIMENT_PATH.joinpath(data_file_base)
# metadata_local.exists()
# m = MTH5()
# m.open_mth5(metadata_local)
# m.channel_summary
# m.close_mth5()
# m.open_mth5(data_file)
# m.channel_summary
"""


import argparse
import numpy as np
import pandas as pd
import pathlib
import time

from pathlib import Path

from aurora.sandbox.mth5_helpers import get_experiment_from_obspy_inventory
from aurora.sandbox.mth5_helpers import mth5_from_experiment

from aurora.sandbox.mth5_helpers import build_request_df
from aurora.test_utils.earthscope.helpers import DataAvailability
from aurora.test_utils.earthscope.helpers import DATA_PATH
from aurora.test_utils.earthscope.helpers import get_most_recent_summary_filepath
from aurora.test_utils.earthscope.helpers import get_summary_table_filename
from aurora.test_utils.earthscope.helpers import get_summary_table_schema
from aurora.test_utils.earthscope.helpers import restrict_to_mda
from aurora.test_utils.earthscope.helpers import USE_CHANNEL_WILDCARDS
from mth5.mth5 import MTH5
from mth5.clients import FDSN, MakeMTH5
from mt_metadata.transfer_functions.core import TF
from mt_metadata import TF_XML

STAGE_ID = 3
if not USE_CHANNEL_WILDCARDS:
    DATA_AVAILABILITY = DataAvailability()


def enrich_row(row):
    if isinstance(row.exception, str):
        print(f"Skipping row {row} for now, Exception {row.exception} was encounterd in metadata")
        return row

    if USE_CHANNEL_WILDCARDS:
        availabile_channels = ["*Q*", "*F*", ]
    else:
        availabile_channels = DATA_AVAILABILITY.get_available_channels(
            row.network_id, row.station_id)

    request_df = build_request_df(row.station_id, row.network_id,
                                  channels=availabile_channels, start=None, end=None)

    fdsn_object = FDSN(mth5_version='0.2.0')
    fdsn_object.client = "IRIS"

    expected_file_name = DATA_PATH.joinpath(fdsn_object.make_filename(request_df))
    if expected_file_name.exists():
        print(f"Already have data for {row.station_id}-{row.network_id}")
        row.at["data_mth5_size"] = expected_file_name.stat().st_size
        row.at["data_mth5_name"] = expected_file_name
        return row
    try:
        print(request_df)
        mth5_filename = fdsn_object.make_mth5_from_fdsn_client(request_df,
                                                               interact=False,
                                                               path=DATA_PATH)
        row.at["data_mth5_size"] = expected_file_name.stat().st_size
        row.at["data_mth5_name"] = expected_file_name
        row.at["data_mth5_exception"] = ""
        row.at["data_mth5_error_message"] = ""
    except Exception as e:
        row.at["data_mth5_size"] = 0
        row.at["data_mth5_name"] = ""
        row.at["data_mth5_exception"] = e.__class__.__name__
        row.at["data_mth5_error_message"] = e.args[0]
    return row

def prepare_dataframe_for_scraping(restrict_to_first_n_rows=False):
    """
    Define columns and default values
    Args:
        source_csv:

    Returns:

    """
    source_csv = get_most_recent_summary_filepath(2)
    source_df = pd.read_csv(source_csv)
    df = source_df.copy(deep=True)
    renamer_dict = {"filesize":"metadata_filesize",
                    "filename":"metadata_filename"}
    df = df.rename(columns=renamer_dict)
    df["data_mth5_size"] = 0
    df["data_mth5_name"] = ""
    df["data_mth5_exception"] = ""
    df["data_mth5_error_message"] = ""
    n_rows = len(df)
    info_str = f"There are {n_rows} network-station pairs"
    print(info_str)
    if restrict_to_first_n_rows:
        df = df.iloc[:restrict_to_first_n_rows]
        info_str += f"\n restricting to first {restrict_to_first_n_rows} rows for testing"
        n_rows = len(df)
    print(info_str)
    return df


def batch_download_mth5(output_csv=None, restrict_to_first_n_rows=False, npartitions=0):
    """

    :param xml_source_column:"data_xml_path" or "emtf_xml_path"
    specifies which of the two possible collections of xml files to use as source
    :return:
    """

    t0 = time.time()
    df = prepare_dataframe_for_scraping(restrict_to_first_n_rows=restrict_to_first_n_rows)
    if npartitions:
        import dask.dataframe as dd
        ddf = dd.from_pandas(df, npartitions=npartitions)
        n_rows = len(df)
        df_schema = get_summary_table_schema(STAGE_ID)
        ddf = ddf[list(df_schema.keys())] #force column order to agree with schema dict (probably should use ordered dict in schema definition)
        enriched_df = ddf.apply(enrich_row, axis=1, meta=df_schema).compute()
    else:
        enriched_df = df.apply(enrich_row, axis=1)
    if output_csv:
        enriched_df.to_csv(output_csv, index=False)

def review_results():
    coverage_csv = get_summary_table_filename(STAGE_ID)
    coverage_df = pd.read_csv(coverage_csv)
    print("OK")
    pass

def main():
    output_csv = get_summary_table_filename(STAGE_ID)
    batch_download_mth5(output_csv=output_csv, restrict_to_first_n_rows=10, npartitions=20)
    #review_results()
    print("all done!")

if __name__ == "__main__":
    main()
