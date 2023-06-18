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

from aurora.test_utils.earthscope.helpers import SPUD_XML_CSV
from aurora.test_utils.earthscope.helpers import SPUD_XML_PATH
from aurora.test_utils.earthscope.helpers import SUMMARY_TABLES_PATH
from aurora.test_utils.earthscope.helpers import DATA_PATH
from aurora.test_utils.earthscope.helpers import load_xml_tf
from aurora.test_utils.earthscope.helpers import get_most_recent_summary_filepath
from aurora.test_utils.earthscope.helpers import get_remotes_from_tf
from aurora.test_utils.earthscope.helpers import get_remotes_from_tf_2
from aurora.test_utils.earthscope.helpers import get_rr_type
from aurora.test_utils.earthscope.helpers import get_summary_table_filename
from aurora.test_utils.earthscope.helpers import load_most_recent_summary

STAGE_ID = 1


def review_spud_tfs(xml_sources=["emtf_xml_path", "data_xml_path"],
                    results_csv=""):
    """

    :param xml_source_column:"data_xml_path" or "emtf_xml_path"
    specifies which of the two possible collections of xml files to use as source
    :return:
    """
    if not results_csv:
        results_csv = get_summary_table_filename(STAGE_ID)

    t0 = time.time()
    spud_df = pd.read_csv(SPUD_XML_CSV)

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
            xml_path = pathlib.Path(row[xml_source])
            try:
                spud_tf = load_xml_tf(xml_path)
                rr_type = get_rr_type(spud_tf)
                spud_df[f"{xml_source}_remote_ref_type"].iat[i_row] = rr_type
                remotes = get_remotes_from_tf(spud_tf)
                spud_df[f"{xml_source}_remotes"].iat[i_row] = ",".join(remotes)
                remotes2 = get_remotes_from_tf_2(spud_tf)
                spud_df[f"{xml_source}_remotes_2"].iat[i_row] = ",".join(remotes)

            except Exception as e:
                spud_df[f"{xml_source}_error"].at[i_row] = True
                spud_df[f"{xml_source}_exception"].at[i_row] = e.__class__.__name__
                spud_df[f"{xml_source}_error_message"].at[i_row] = e.args[0]
        print(i_row, row[xml_source])
    spud_df.to_csv(results_csv, index=False)
    print(f"Took {time.time()-t0}s to review spud tfs")
    return spud_df




def main():
    # normal
    results_df = review_spud_tfs()

    # run only data
    #results_df = review_spud_tfs(xml_sources = ["data_xml_path", ])


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