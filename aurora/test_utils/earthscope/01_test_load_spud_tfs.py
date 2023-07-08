"""
This script iterates over all of the scraped XML from SPUD and
registers information about success or failure of ingest into a mt_metadata TF object

There are two possible places to access an xml in each row, called
emtf_xml_path and data_xml_path.
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

DROP_COLS = ["emtf_xml_path", "data_xml_path"]
DF_SCHEMA = get_summary_table_schema(1)
USE_SECOND_WAY_OF_PARSING_REMOTES = False # Deprecated
# Have already asserted that
# (df.data_remotes.astype(str)==df.data_remotes_2.astype(str)).all()
# (df.emtf_remotes.astype(str)==df.emtf_remotes_2.astype(str)).all()
# (df.emtf_remotes.astype(str) == df.data_remotes_2.astype(str)).all()
def prepare_dataframe_for_scraping(restrict_to_first_n_rows=False):
    """
    Define the data structure that is output from this stage of processing
    It is basically the df from the prvious stage (0) with some new rows added.

    :param restrict_to_first_n_rows:
    :return:
    """
    spud_xml_csv = get_summary_table_filename(0)
    spud_df = pd.read_csv(spud_xml_csv)

def enrich_row():
    """
    This will eventually get used by dask, but as a step we need to make this a method
    that works with df.apply()
    Returns:

    """
    pass

def review_spud_tfs(xml_sources=["emtf", "data"], results_csv=""):
    """

    :param xml_sources:"data_xml_path" or "emtf_xml_path"
        20230702
    specifies which of the two possible collections of xml files to use as source
    :return:
    """
    if not results_csv:
        results_csv = get_summary_table_filename(STAGE_ID)

    t0 = time.time()
    spud_xml_csv = get_summary_table_filename(0)
    spud_df = pd.read_csv(spud_xml_csv)

    # Set Up Schema with default values
    for xml_source in xml_sources:
        spud_df[f"{xml_source}_error"] = False
        spud_df[f"{xml_source}_exception"] = ""
        spud_df[f"{xml_source}_error_message"] = ""
        spud_df[f"{xml_source}_remote_ref_type"] = ""
        spud_df[f"{xml_source}_remotes"] = ""
        spud_df[f"{xml_source}_remotes_2"] = ""

    for i_row, row in spud_df.iterrows():
        # if i_row<750:
        #     continue
        for xml_source in xml_sources:
            xml_path = SPUD_XML_PATHS[xml_source].joinpath(row[f"{xml_source}_xml_filebase"])
            try:
                spud_tf = load_xml_tf(xml_path)
                rr_type = get_rr_type(spud_tf)
                spud_df[f"{xml_source}_remote_ref_type"].iat[i_row] = rr_type
                remotes = get_remotes_from_tf(spud_tf)
                spud_df[f"{xml_source}_remotes"].iat[i_row] = ",".join(remotes)
                if USE_SECOND_WAY_OF_PARSING_REMOTES:
                    remotes2 = get_remotes_from_tf_2(spud_tf)
                    spud_df[f"{xml_source}_remotes_2"].iat[i_row] = ",".join(remotes)

            except Exception as e:
                spud_df[f"{xml_source}_error"].at[i_row] = True
                spud_df[f"{xml_source}_exception"].at[i_row] = e.__class__.__name__
                spud_df[f"{xml_source}_error_message"].at[i_row] = e.args[0]
        print(i_row, xml_source)
    spud_df.to_csv(results_csv, index=False)
    print(f"Took {time.time()-t0}s to review spud tfs")
    return spud_df



def summarize_errors():
    xml_sources = ["data", "emtf"]
    df = load_most_recent_summary(1)
    for xml_source in xml_sources:
        print(f"{xml_source} error \n {df[f'{xml_source}_error'].value_counts()}\n\n")

    print("OK")

def main():
    # normal
    results_df = review_spud_tfs()

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
