"""
This script iterates over all of the scraped XML from SPUD and registers information about success or failure of
ingest into a mt_metadata TF object.

There are two possible places to access an xml in each row, called emtf_xml_path and data_xml_path.

It has been asserted that
(df.data_remotes.astype(str)==df.data_remotes_2.astype(str)).all()
(df.emtf_remotes.astype(str)==df.emtf_remotes_2.astype(str)).all()
(df.emtf_remotes.astype(str) == df.data_remotes_2.astype(str)).all()
which basically means we can deprecate one of get_remotes_from_tf, get_remotes_from_tf_2

Dask Notes:
- 0 partitions 720s
- 1 partitions 682s
- 2 partitions 723s
- 4 partitions 882s
- 12 partitions 866s
- 32 partitions 857s

Not much difference.

Link where I originally got into dask apply with partitions: ____find this again__

But I am starting to suspect multiprocessing is the right solution..
https://stackoverflow.com/questions/67457956/how-to-parallelize-the-row-wise-pandas-dataframes-apply-method
but read this first:
https://examples.dask.org/applications/embarrassingly-parallel.html
"""



import datetime
import pandas as pd
import pathlib
import time

from aurora.test_utils.earthscope.helpers import SPUD_XML_PATHS
from aurora.test_utils.earthscope.helpers import SUMMARY_TABLES_PATH
from aurora.test_utils.earthscope.helpers import load_xml_tf
from aurora.test_utils.earthscope.helpers import get_most_recent_summary_filepath
from aurora.test_utils.earthscope.helpers import get_remotes_from_tf
from aurora.test_utils.earthscope.helpers import get_remotes_from_tf_2
from aurora.test_utils.earthscope.helpers import get_rr_type
from aurora.test_utils.earthscope.helpers import get_summary_table_filename
from aurora.test_utils.earthscope.helpers import get_summary_table_schema
from aurora.test_utils.earthscope.helpers import load_most_recent_summary


STAGE_ID = 1


# Config Params
XML_SOURCES = ["emtf", "data"]
N_PARTITIONS = 1

def prepare_dataframe_for_scraping(restrict_to_first_n_rows=False,):
    """
    Define the data structure that is output from this stage of processing
    It is basically the df from the prvious stage (0) with some new rows added.

    :param restrict_to_first_n_rows:
    :return:
    """
    spud_xml_csv = get_summary_table_filename(0)
    spud_df = pd.read_csv(spud_xml_csv)

    # Set Up Schema with default values
    for xml_source in XML_SOURCES:
        spud_df[f"{xml_source}_error"] = False
        spud_df[f"{xml_source}_exception"] = ""
        spud_df[f"{xml_source}_error_message"] = ""
        spud_df[f"{xml_source}_remote_ref_type"] = ""
        spud_df[f"{xml_source}_remotes"] = ""


    return spud_df

def enrich_row(row):
    """
    This will eventually get used by dask, but as a step we need to make this a method
    that works with df.apply()
    Returns:

    """
    for xml_source in XML_SOURCES:
        xml_path = SPUD_XML_PATHS[xml_source].joinpath(row[f"{xml_source}_xml_filebase"])
        try:
            tf = load_xml_tf(xml_path)
            # OLD
            rr_type = get_rr_type(tf)
            row[f"{xml_source}_remote_ref_type"] = rr_type
            remotes = get_remotes_from_tf(tf)
            row[f"{xml_source}_remotes"] = ",".join(remotes)

            # NEW
            # remotes = tf.station_metadata.transfer_function.remote_references
            # remotes = [x for x in remotes if x != tf.station]
            # Do we want the "self" station being returned in remotes
            # rr_type = tf.station_metadata.transfer_function.processing_type
            # row[f"{xml_source}_remote_ref_type"] = rr_type
            # row[f"{xml_source}_remotes"] = ",".join(remotes)

        except Exception as e:
            row[f"{xml_source}_error"] = True
            row[f"{xml_source}_exception"] = e.__class__.__name__
            row[f"{xml_source}_error_message"] = e.args[0]
    return row


def batch_process(row_start=0, row_end=None):
    t0 = time.time()
    df = prepare_dataframe_for_scraping()
    if row_end is None:
        row_end = len(df)
    df = df[row_start:row_end]
    if not N_PARTITIONS:
        enriched_df = df.apply(enrich_row, axis=1)
    else:
        import dask.dataframe as dd
        ddf = dd.from_pandas(df, npartitions=N_PARTITIONS)
        n_rows = len(df)
        print(f"nrows ---> {n_rows}")
        df_schema = get_summary_table_schema(STAGE_ID)
        enriched_df = ddf.apply(enrich_row, axis=1, meta=df_schema).compute()

    results_csv = get_summary_table_filename(STAGE_ID)
    enriched_df.to_csv(results_csv, index=False)
    print(f"Took {time.time()-t0}s to review spud tfs, running with {N_PARTITIONS} partitions")
    return enriched_df


def summarize_errors():
    xml_sources = ["data", "emtf"]
    df = load_most_recent_summary(1)
    for xml_source in xml_sources:
        print(f"{xml_source} error \n {df[f'{xml_source}_error'].value_counts()}\n\n")

    print("OK")

def main():
    # normal
    # results_df = batch_process(row_end=1)
    results_df = batch_process()#row_end=100)

    # run only data
    #results_df = review_spud_tfs(xml_sources = ["data_xml_path", ])

    summarize_errors()


    # # DEBUGGING
    df = load_most_recent_summary(1)
    # old_df = pd.read_csv(SUMMARY_TABLES_PATH.joinpath("01_spud_xml_review_2023-06-03_114350.csv"))
    # n_xml = len(df)
    # is_not_mda = df.data_xml_path.str.contains("__")
    # n_non_mda = is_not_mda.sum()
    # n_mda = len(df) - n_non_mda
    # print(f"There are {n_mda} / {n_xml} files with mda string ")
    # print(f"There are {n_non_mda} / {n_xml} files without mda string ")
    # non_mda_df = df[is_not_mda]
    print("summarize")

    

if __name__ == "__main__":
    main()
