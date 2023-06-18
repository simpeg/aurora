"""
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

STAGE_ID = 4


GET_REMOTES_FROM = "spud_xml_review"

processing_summary_csv = get_summary_table_filename(STAGE_ID)
print(SUMMARY_TABLES_PATH)
print(processing_summary_csv)
print(processing_summary_csv)






# def initialize_coverage_df():
#     #local_data_coverage_df = pd.DataFrame(columns=["station_id", "network_id", "filename", "filesize"])
#     pass
#
#
#
#
# local_data_coverage_csv = get_most_recent_summary_filepath(3)
# local_data_coverage_df = pd.read_csv(data_coverage_csv)


def batch_process(xml_source="data_xml_path"):
    """


    """

    for i_row, row in spud_df.iterrows():
        print(row)
        if row[f"{xml_source}_error"] is True:
            print(f"Skipping {row} for now, tf not reading in")
            continue
        if row.station_id == "CAM01":
            print("DEBUG")
        data_file_base = f"{row.network_id}_{row.station_id}.h5"
        data_file = DATA_PATH.joinpath(data_file_base)
        if not data_file.exists():
            print(f"skipping proc {data_file} DNE")
            continue
        # if GET_REMOTES_FROM == "tf_xml":
        #     tf = load_xml_tf(xml_path)
        #     rr_type = get_rr_type(tf)
        #     remotes = get_remotes_from_tf(tf)
        elif GET_REMOTES_FROM == "spud_xml_review":
            remotes = row.data_xml_path_remotes.split(",")

        if remotes:
            print(f"remotes: {remotes} ")
            remote_file_base = f"{row.network_id}_{remotes[0]}.h5"
            rr_file = DATA_PATH.joinpath(remote_file_base)
            if rr_file.exists():
                remote_id = remotes[0]
                mth5_files = [data_file, rr_file]
            else:
                rr_file = None
                remote_id = None
                mth5_files = [data_file,]
        else:
            remotes=None
            remote_id = None
            mth5_files = [data_file, ]

        xml_file_base = f"{row.network_id}_{row.station_id}_RR{remote_id}.xml"
        xml_file_path = AURORA_TF_PATH.joinpath(xml_file_base)
        if xml_file_path.exists():
            print(f"SKIPPING {xml_file_path} exists")
            continue
        mth5_run_summary = RunSummary()
        mth5s = [data_file, rr_file]
        mth5s = [x for x in mth5s if x is not None]
        mth5_run_summary.from_mth5s(mth5s)
        run_summary = mth5_run_summary.clone()
        run_summary.mini_summary

        # DEBUG:
        # metadata_local = EXPERIMENT_PATH.joinpath(data_file_base)
        # metadata_local.exists()
        # m = MTH5()
        # m.open_mth5(metadata_local)
        # m.channel_summary
        # m.close_mth5()
        # m.open_mth5(data_file)
        # m.channel_summary

        kernel_dataset = KernelDataset()
        kernel_dataset.from_run_summary(run_summary, row.station_id, remote_id)
        #kernel_dataset.from_run_summary(run_summary, "AZT14")
        # kernel_dataset.mini_summary
        if len(kernel_dataset.df) == 0:
            print("No RR Coverage")
            kernel_dataset.from_run_summary(run_summary, row.station_id)
            continue

        cc = ConfigCreator()
        config = cc.create_from_kernel_dataset(kernel_dataset)

        show_plot = False
        tf_cls = process_mth5(config,
                            kernel_dataset,
                            units="MT",
                            show_plot=show_plot,
                            z_file_path=None,
                        )
        # xml_file_base = f"{row.network_id}_{row.station_id}_RR{remote_id}.xml"
        # xml_file_path = AURORA_TF_PATH.joinpath(xml_file_base)
        tf_cls.write(fn=xml_file_path, file_type="emtfxml")


def main():
    batch_process()

if __name__ == "__main__":
    main()