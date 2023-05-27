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
from aurora.test_utils.earthscope.helpers import DATA_PATH
from mt_metadata.transfer_functions.core import TF

SPUD_EMTF_XML_COLUMN = "emtf_xml_path"
SPUD_DATA_XML_COLUMN = "data_xml_path"
SPUD_DF = pd.read_csv(SPUD_XML_CSV)
now = datetime.datetime.now().__str__().split(".")[0].replace(" ","_")
SPUD_XML_REVIEW_CSV = f"spud_xml_review_{now}.csv"


def load_xml_tf(file_path):
    """
    using emtf_xml path will fail with KeyError: 'field_notes'
    :param file_path:
    :return:
    """
    # if "15029445_EM_PAM57" in str(file_path):
    #     print("debug")
    print(f"reading {file_path}")
    spud_tf = TF(file_path)
    spud_tf.read()
    return spud_tf



def review_spud_tfs(xml_sources=["emtf_xml_path", "data_xml_path"],
                    results_csv=SPUD_XML_REVIEW_CSV):
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

    for i_row, row in spud_df.iterrows():
        if i_row<750:
            continue
        for xml_source in xml_sources:
            xml_path = pathlib.Path(row[xml_source])
            try:
                spud_tf = load_xml_tf(xml_path)
            except Exception as e:
                spud_df[f"{xml_source}_error"].at[i_row] = True
                spud_df[f"{xml_source}_exception"].at[i_row] = e.__class__.__name__
                spud_df[f"{xml_source}_error_message"].at[i_row] = e.args[0]
        print(i_row, row[xml_source])
    spud_df.to_csv(results_csv)
    print(f"Took {time.time()-t0}s to review spud tfs")
    return spud_df



def main():
    results_df = review_spud_tfs()#
    results_df = pd.read_csv(SPUD_XML_REVIEW_CSV)

if __name__ == "__main__":
    main()