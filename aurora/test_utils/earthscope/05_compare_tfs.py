"""
20230619
First attempt:
For every mda-containing data-xml in SPUD, we have tried to process and aurora TF.

"""



import numpy as np
import pandas as pd
import pathlib
import time

from matplotlib import pyplot as plt
from pathlib import Path

from aurora.test_utils.earthscope.helpers import build_request_df
from aurora.test_utils.earthscope.helpers import DATA_PATH
from aurora.test_utils.earthscope.helpers import EXPERIMENT_PATH
from aurora.test_utils.earthscope.helpers import SPUD_DATA_PATH
from aurora.test_utils.earthscope.helpers import SPUD_EMTF_PATH
from aurora.test_utils.earthscope.helpers import SPUD_XML_CSV
from aurora.test_utils.earthscope.helpers import SPUD_XML_PATH
from aurora.test_utils.earthscope.helpers import SUMMARY_TABLES_PATH
from aurora.test_utils.earthscope.helpers import AURORA_TF_PATH
from aurora.test_utils.earthscope.helpers import load_xml_tf
from aurora.test_utils.earthscope.helpers import load_most_recent_summary
from aurora.test_utils.earthscope.helpers import get_summary_table_filename
from aurora.test_utils.earthscope.helpers import restrict_to_mda

from aurora.config.config_creator import ConfigCreator
from aurora.pipelines.process_mth5 import process_mth5
from aurora.pipelines.run_summary import RunSummary
from aurora.transfer_function.kernel_dataset import KernelDataset

from mth5.mth5 import MTH5
from mth5.clients import FDSN, MakeMTH5
from mt_metadata.transfer_functions.core import TF
from mt_metadata import TF_XML


spud_df = load_most_recent_summary(1)
spud_df = restrict_to_mda(spud_df, RR="Robust Remote Reference")

STAGE_ID = 5


processing_summary_df = load_most_recent_summary(4)
tf_summary_csv = get_summary_table_filename(STAGE_ID)
print(tf_summary_csv)


tf_report_schema = ["data_id", "station_id", "network_id", "remote_id",
                                  "filename", "exception", "error_message", "data_xml_path"]
def initialize_tf_df():
    df = pd.DataFrame(columns=tf_report_schema)
    return df


def batch_compare(xml_source="data_xml_path"):
    """


    """
    tf_df = initialize_tf_df()

    for i_row, row in processing_summary_df.iterrows():
        print(row)
        if bool(np.isnan(row.exception)):
            pass
        else:
            print("SKIPPING EXCEPTION")
        spud_tf = load_xml_tf(row.data_xml_path)
        aurora_tf = load_xml_tf(row.filename)


        # Find Overlap of Periods where both TFs are defined
        print("ADD some accounting here for how much is dropped from each")
        # Selecting glb and lub
        #lowest_freq = max(aurora_tf.frequency.min(), spud_tf.frequency.min())
        #highest_freq = min(aurora_tf.frequency.max(), spud_tf.frequency.max())
        shortest_period = max(aurora_tf.transfer_function.period.data.min(),
                              spud_tf.transfer_function.period.data.min())
        longest_period = min(aurora_tf.transfer_function.period.data.max(),
                              spud_tf.transfer_function.period.data.max())
        cond1 = spud_tf.transfer_function.period >= shortest_period
        cond2 = spud_tf.transfer_function.period <= longest_period
        reduced_spud_tf = spud_tf.transfer_function.where(cond1 & cond2, drop=True)
        cond1 = aurora_tf.transfer_function.period >= shortest_period
        cond2 = aurora_tf.transfer_function.period <= longest_period
        reduced_aurora_tf = aurora_tf.transfer_function.where(cond1 & cond2, drop=True)

        # Now we have arrays that share a range (mostly)
        # try interpolating, in this case we interp aurora onto spud freqs
        qq = reduced_aurora_tf.interp(period=reduced_spud_tf.period)

        # reduce to 1D at a time
        input = "hx"
        output = "ex"
        one_dim = qq.sel(input=input, output=output)

        delta_along_dim = reduced_spud_tf.sel(input=input, output=output).data - one_dim.data
        delta_along_dim = delta_along_dim[np.isnan(delta_along_dim)==False]
        delta_along_dim = np.abs(delta_along_dim)

        # THIS IS THE ANSWER
        np.linalg.norm(delta_along_dim)

        try:
            new_row = {"data_id": row.data_id,
                       "station_id": row.station_id,
                       "network_id": row.network_id,
                       "remote_id": remote_id,
                       "filename": xml_file_path,
                       "exception": "",
                       "error_message": "",
                       "data_xml_path":row.data_xml_path}
        except Exception as e:
            new_row = {"data_id": row.data_id,
                       "station_id": row.station_id,
                       "network_id": row.network_id,
                       "remote_id": remote_id,
                       "filename": xml_file_path,
                       "exception": e.__class__.__name__,
                       "error_message": e.args[0],
                       "data_xml_path":row.data_xml_path}
       # tf_df = tf_df.append(new_row, ignore_index=True)
       # tf_df.to_csv(tf_csv, index=False)


def main():
    batch_compare()

if __name__ == "__main__":
    main()