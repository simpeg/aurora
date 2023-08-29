"""
"""



import numpy as np
import pandas as pd
import pathlib
import time

from matplotlib import pyplot as plt
from pathlib import Path

from aurora.test_utils.earthscope.helpers import DATA_PATH
from aurora.test_utils.earthscope.helpers import EXPERIMENT_PATH
from aurora.test_utils.earthscope.helpers import SUMMARY_TABLES_PATH
from aurora.test_utils.earthscope.helpers import AURORA_TF_PATH
from aurora.test_utils.earthscope.helpers import AURORA_Z_PATH
from aurora.test_utils.earthscope.helpers import load_most_recent_summary
from aurora.test_utils.earthscope.helpers import get_summary_table_filename
from aurora.test_utils.earthscope.helpers import get_summary_table_schema
from aurora.test_utils.earthscope.helpers import restrict_to_mda

from aurora.config.config_creator import ConfigCreator
from aurora.pipelines.process_mth5 import process_mth5
from aurora.pipelines.run_summary import RunSummary
from aurora.transfer_function.kernel_dataset import KernelDataset

from mth5.clients import FDSN, MakeMTH5
from mth5.helpers import close_open_files
from mth5.mth5 import MTH5

spud_df = load_most_recent_summary(1)
spud_df = restrict_to_mda(spud_df, RR="Robust Remote Reference")

USE_SKELETON = True
N_PARTITIONS = 0
STAGE_ID = 4


processing_summary_csv = get_summary_table_filename(STAGE_ID)
print(SUMMARY_TABLES_PATH)
print(processing_summary_csv)





processing_report_schema = get_summary_table_schema(4)

def initialize_processing_df():
    df = pd.DataFrame(columns=processing_report_schema.keys())
    return df

def prepare_dataframe_for_processing(use_skeleton=USE_SKELETON):
    """
    Towards parallelization, I want to make the skeleton of the dataframe first, and then fill it in.
    Ppeviously, we had added rows to the dataframe on the fly.

    Returns
    -------

    """
    skeleton_file = "04_skeleton.csv"
    if use_skeleton:
        df = pd.read_csv(skeleton_file)
        return df

    spud_df = load_most_recent_summary(1)
    spud_df = restrict_to_mda(spud_df, RR="Robust Remote Reference")

    # try:
    #     processing_df = pd.read_csv(processing_summary_csv)
    # except FileNotFoundError:
    processing_df = initialize_processing_df()

    for i_row, row in spud_df.iterrows():
        print(row) #station_id = row.station_id; network_id = row.network_id

        # Confirm XML is readable
        xml_source = "data"
        if row[f"{xml_source}_error"] is True:
            print(f"Skipping {row} for now, tf not reading in")
            continue

        # confirm we have h5 data
        # this could be maybe better checked by using table from 03, since h5 may exist but have no data
        data_file_base = f"{row.network_id}_{row.station_id}.h5"
        data_file = DATA_PATH.joinpath(data_file_base)
        if not data_file.exists():
            print(f"skipping proc {data_file} DNE")
            continue

        # check for a remote reference
        # probably don't need mth5_files variable in this preparation of dataframe
        remotes = row.data_remotes.split(",")
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
                mth5_files = [data_file, ]
        else:
            remotes = None
            remote_id = ""
            mth5_files = [data_file, ]

        xml_file_base = f"{row.network_id}_{row.station_id}_RR{remote_id}.xml"
        #xml_file_path = AURORA_TF_PATH.joinpath(xml_file_base)

        new_row = {
            "data_id": row.data_id,
            "network_id": row.network_id,
            "station_id": row.station_id,
            "remote_id": remote_id,
            "filename": xml_file_base,
            "exception": "",
            "error_message" : "",
            "data_xml_filebase": row.data_xml_filebase}
        processing_df = processing_df.append(new_row, ignore_index=True)

    # Now you have processing_df, not sure this needs to be uniquified ?...
    print(len(processing_df))
    subset = ['network_id', 'station_id', 'remote_id' ]
    ucpf = processing_df.drop_duplicates(subset=subset, keep='first')
    print(len(ucpf))
    ucpf.to_csv(skeleton_file, index=False)
    return ucpf


def enrich_row(row):
    mth5_file_base = f"{row.network_id}_{row.station_id}.h5"
    data_file = DATA_PATH.joinpath(mth5_file_base)
    if row.remote_id:
        remote_file_base = f"{row.network_id}_{row.remote_id}.h5"
        rr_file = DATA_PATH.joinpath(remote_file_base)
        mth5_files = [data_file, rr_file]
    else:
        mth5_files = [data_file, ]

    mth5_run_summary = RunSummary()
    mth5s = [data_file, rr_file]
    mth5s = [x for x in mth5s if x is not None]
    mth5_run_summary.from_mth5s(mth5s)
    run_summary = mth5_run_summary.clone()
    run_summary.mini_summary

    kernel_dataset = KernelDataset()
    kernel_dataset.from_run_summary(run_summary, row.station_id, row.remote_id)

    if len(kernel_dataset.df) == 0:
        print("No RR Coverage")
        kernel_dataset.from_run_summary(run_summary, row.station_id)


    try:
        cc = ConfigCreator()
        config = cc.create_from_kernel_dataset(kernel_dataset)
        show_plot = False
        z_file = AURORA_Z_PATH.joinpath(row.filename.replace("xml", "zrr"))
        tf_cls = process_mth5(config,
                              kernel_dataset,
                              units="MT",
                              show_plot=show_plot,
                              z_file_path=z_file,
                              )
        xml_file_path = AURORA_TF_PATH.joinpath(row.filename)
        tf_cls.write(fn=xml_file_path, file_type="emtfxml")
    except Exception as e:
        row.exception = e.__class__.__name__
        row.error_message = e.args[0]
    print("check if the mth5 objects get closed when we leave this function ... ")
    #close_open_files() wont work under dask, because we can't just blanket close all h5 files,
    # since other, parallel processes will be interacting with these files.
    return row



def batch_process():

    df = prepare_dataframe_for_processing()
    #df = df.iloc[0:1]
    if not N_PARTITIONS:
        enriched_df = df.apply(enrich_row, axis=1)

    else:
        import dask.dataframe as dd
        ddf = dd.from_pandas(df, npartitions=N_PARTITIONS)
        n_rows = len(df)
        print(f"nrows ---> {n_rows}")
        df_schema = get_summary_table_schema(2)
        enriched_df = ddf.apply(enrich_row, axis=1, meta=df_schema).compute()

    processing_summary_csv = get_summary_table_filename(STAGE_ID)
    enriched_df.to_csv(processing_summary_csv, index=False)



def compare_graphics():
    from aurora.transfer_function.plot.comparison_plots import compare_two_z_files
    compare_two_z_files(
        "/home/kkappler/Downloads/ORF08bc_G8.zrr",
        "EM_ORF08_RRORG08.zrr",
        angle2=0.0,
        label1="emtf",
        label2="aurora",
        scale_factor1=1,
        out_file="",
        markersize=3,
        rho_ylims=[1e0, 1e3],
        # rho_ylims=[1e-8, 1e6],
        xlims=[1, 5000],
    )

def main():
    batch_process()
    #compare_graphics()
    #ucpf = prepare_dataframe_for_processing()

if __name__ == "__main__":
    main()
