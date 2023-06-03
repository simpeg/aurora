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
from aurora.test_utils.earthscope.helpers import DATA_PATH
from aurora.test_utils.earthscope.helpers import load_xml_tf
from aurora.test_utils.earthscope.helpers import get_remotes_from_tf
from aurora.test_utils.earthscope.helpers import get_rr_type

SPUD_DF = pd.read_csv(SPUD_XML_CSV)
now = datetime.datetime.now().__str__().split(".")[0].replace(" ","_")
now_str = now.replace(":","")
SPUD_XML_REVIEW_CSV_NAME = f"spud_xml_review_{now_str}.csv"
SPUD_XML_REVIEW_CSV_PATH = SPUD_XML_PATH.joinpath(SPUD_XML_REVIEW_CSV_NAME)



def review_spud_tfs(xml_sources=["emtf_xml_path", "data_xml_path"],
                    results_csv=SPUD_XML_REVIEW_CSV_PATH):
    """

    :param xml_source_column:"data_xml_path" or "emtf_xml_path"
    specifies which of the two possible collections of xml files to use as source
    :return:
    """
    t0 = time.time()
    spud_df = pd.read_csv(SPUD_XML_CSV)

    for xml_source in xml_sources:
        spud_df[f"{xml_source}_error"] = False
        spud_df[f"{xml_source}_exception"] = ""
        spud_df[f"{xml_source}_error_message"] = ""
        spud_df[f"{xml_source}_remote_ref_type"] = ""
        spud_df[f"{xml_source}_remotes"] = ""

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

            except Exception as e:
                spud_df[f"{xml_source}_error"].at[i_row] = True
                spud_df[f"{xml_source}_exception"].at[i_row] = e.__class__.__name__
                spud_df[f"{xml_source}_error_message"].at[i_row] = e.args[0]
        print(i_row, row[xml_source])
    spud_df.to_csv(results_csv)
    print(f"Took {time.time()-t0}s to review spud tfs")
    return spud_df



def get_station_info():
    pass

def main():
    results_df = review_spud_tfs()

    # DEBUGGING
    # review_csv_name = "spud_xml_review_2023-05-28_13:21:18.csv"
    # review_csv_path = SPUD_XML_PATH.joinpath(review_csv_name)
    # df = pd.read_csv(review_csv)

    results_df = pd.read_csv(SPUD_XML_REVIEW_CSV_PATH)
    print("summarize")

    

if __name__ == "__main__":
    main()