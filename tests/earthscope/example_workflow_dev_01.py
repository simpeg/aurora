"""
Based on example_workflow.py commmited by Laura Keyson 05 May 2023
Updated by Karl Kappler 20 May 2023
Flow is to use SPUD_XML_CSV as the primary iterator.


We will iterate over rows of the CSV, selecting only rows where the name is of the form:
18057859_EM_MH010.xml
uid_NETWORK_STATION.xml

For each of these, we will extract the data.

- data extraction will start with identifying relevant RR stations from the XML
- We will assume the RR stations are in the same Network as the local station
- Then, for each of [station, RR1, RR2, ... RRN]:
    The ffull dataset will be downloaded

a metadata (dataless) pull from Earthscope archive
- The available time intervals will be established



There are two possible places to access an xml in each row, called
emtf_xml_path and data_xml_path.  we will use data_xml_path as our source.

For each xml, we instantiate a tf object

columsn in this
For each of th


"""
# Required imports for the program.



import datetime as dt
import glob
import numpy as np
import pandas as pd
import pathlib
import time

from matplotlib import pyplot as plt
from pathlib import Path

from aurora.config import BANDS_DEFAULT_FILE
from aurora.config.config_creator import ConfigCreator
from aurora.pipelines.process_mth5 import process_mth5
from aurora.pipelines.run_summary import RunSummary
from aurora.test_utils.earthscope.helpers import build_request_df
from aurora.test_utils.earthscope.helpers import DATA_PATH
from aurora.test_utils.earthscope.helpers import SPUD_DATA_PATH
from aurora.test_utils.earthscope.helpers import SPUD_EMTF_PATH
from aurora.test_utils.earthscope.helpers import SPUD_XML_CSV
from aurora.test_utils.earthscope.helpers import get_remotes_from_tf
from aurora.test_utils.earthscope.helpers import get_rr_type
from aurora.test_utils.earthscope.helpers import load_xml_tf
from aurora.transfer_function.kernel_dataset import KernelDataset
from mth5.mth5 import MTH5
from mth5.clients import FDSN, MakeMTH5
from mt_metadata.transfer_functions.core import TF
from mt_metadata import TF_XML


## Set controls
controls = {}
controls["review_spud_xmls_01"] = False
controls["doInitializeMTH5"] = True
controls["doReadSPUD"] = True
controls["doGetData"] = False
controls["doRunAurora"] = True
controls["doAddSPUD"] = True
controls["doComparison"] = True

    



# def get_data(station, remotes, network, startdate=STARTDATE, enddate=ENDDATE):
#     fdsn_object = FDSN(mth5_version='0.2.0')
#     fdsn_object.client = "IRIS"
#
#     ## Make the data inquiry as a DataFrame
#     print(station)
#     request_list = [[network, station, '', '*', startdate, enddate]]
#
#     try:
#         for remote_station in remotes:
#             request_list.append([network, remote_station, '', '*', startdate, enddate])
#
#     except:
#         pass
#
#     print(request_list)
#
#     request_df = pd.DataFrame(request_list, columns=fdsn_object.request_columns)
#
#     print(f'Request List:\n{request_df}')
#
#     # make_mth5_object = MakeMTH5(mth5_version='0.1.0', interact=False)
#     # mth5_filename = make_mth5_object.from_fdsn_client(request_df, client="IRIS")
#
#     print("Making mth5 from fdsn client")
#
#     mth5_filename = fdsn_object.make_mth5_from_fdsn_client(request_df,
#                                                            interact=False,
#                                                            path=DATA_PATH)
#
#     return mth5_filename


def run_aurora(mth5_filename):
    print("Running AURORA")
    mth5_run_summary = RunSummary()
    mth5_run_summary.from_mth5s([mth5_filename, ])
    run_summary = mth5_run_summary.clone()
    print(run_summary.df)

    coverage_short_list_columns = ['station_id', 'run_id', 'start', 'end', ]
    kernel_dataset = KernelDataset()

    print(type(remotes[0]), remotes[0])
    kernel_dataset.from_run_summary(run_summary, station, remotes[0])
    kernel_dataset.drop_runs_shorter_than(15000)
    print(kernel_dataset.df[coverage_short_list_columns])

    cc = ConfigCreator()
    config = cc.create_from_kernel_dataset(kernel_dataset,
                                           emtf_band_file=BANDS_DEFAULT_FILE, )

    for decimation in config.decimations:
        decimation.estimator.engine = "RME"

    show_plot = False
    aurora_tf = process_mth5(config,
                             kernel_dataset,
                             units="MT",
                             show_plot=show_plot,
                             z_file_path=None,
                             )

    aurora_tf.tf_id = f'{station}_aurora'
    print(f'Result of process_mth5:\n{aurora_tf}')
    tf_group_01 = mth5_object.add_transfer_function(aurora_tf)


def debug():
    #file_base = f'{network}_{station}'
    network = "8P"
    station = "CAS04"
    spud_tf = load_tf(station)
    remotes = get_remotes_from_tf(spud_tf)
    if controls["doGetData"]:
        mth5_file = get_data(station, remotes, network)
    else: #test
        mth5_file = pathlib.Path("/home/kkappler/.cache/earthscope/data/8P_CAS04_NVR08.h5")
    print(mth5_file)
    run_aurora(mth5_file)




# mth5_filename = '8P_REV06_NVS12_CAV08_RET03_RER03.h5'

# print(f"Opening mth5_object from {mth5_filename}")
#
# mth5_object.open_mth5(mth5_filename)

## Initialize a MakeMTH5 object
# if control_dict["doInitializeMTH5"]:
#     print("Creating MTH5 object")
#     mth5_object = MTH5(file_version="0.2.0")






#
#
#
# if doRunAurora:
#     # # run aurora on the mth5_object
#     print("Running AURORA")
#     mth5_run_summary = RunSummary()
#     h5_path = default_path.joinpath(mth5_filename)
#     mth5_run_summary.from_mth5s([h5_path,])
#     run_summary = mth5_run_summary.clone()
#     print(run_summary.df)
#
#     coverage_short_list_columns = ['station_id', 'run_id', 'start', 'end', ]
#     kernel_dataset = KernelDataset()
#
#     print(type(remotes[0]), remotes[0])
#     kernel_dataset.from_run_summary(run_summary, station, remotes[0])
#     kernel_dataset.drop_runs_shorter_than(15000)
#     print(kernel_dataset.df[coverage_short_list_columns])
#
#     cc = ConfigCreator()
#     config = cc.create_from_kernel_dataset(kernel_dataset,
#                                         emtf_band_file=BANDS_DEFAULT_FILE,)
#
#     for decimation in config.decimations:
#         decimation.estimator.engine = "RME"
#
#     show_plot = False
#     aurora_tf = process_mth5(config,
#                         kernel_dataset,
#                         units="MT",
#                         show_plot=show_plot,
#                         z_file_path=None,
#                     )
#
#     aurora_tf.tf_id = f'{station}_aurora'
#     print(f'Result of process_mth5:\n{aurora_tf}')
#     tf_group_01 = mth5_object.add_transfer_function(aurora_tf)
#



def get_mth5_data(xml_df, xml_source="data_xml_path"):
    t0 = time.time()
    xml_df["min_period"] = 0.0
    xml_df["max_period"] = 0.0
    xml_df["station"] = ""
    xml_df["survey"] = ""


    for i_row, row in xml_df.iterrows():
        xml_path = pathlib.Path(row[xml_source])
        assert(len(xml_path.stem.split("_"))==3)
        parts = xml_path.stem.split("_")
        xml_id = parts[0]
        network_id = parts[1]
        station_id = parts[2]
        spud_tf = load_xml_tf(xml_path)
        xml_df["min_period"].at[i_row] = float(spud_tf.dataset.period.min())
        xml_df["max_period"].at[i_row] = float(spud_tf.dataset.period.max())
        xml_df["station"].at[i_row] = spud_tf.station
        xml_df["survey"].at[i_row] = spud_tf.survey
        print(f"{i_row}")
    print(f"Took {time.time()-t0}s for second review of spud tfs")
    xml_df.to_csv(SPUD_XML_REVIEW_CSV_02)
    return xml_df


def main():
    """
    2023-05-27
    :return:


    """
    #local_data_coverage_df = pd.DataFrame(columns=["station_id", "network_id", "filename", "filesize"])
    local_data_coverage_csv = "local_data_coverage.csv"
    local_data_coverage_df = pd.read_csv(local_data_coverage_csv)

    xml_source = "data_xml_path"
    spud_csv = "spud_xml_review_2023-05-26_17:10:33.csv"
    spud_df = pd.read_csv(spud_csv)

    for i_row, row in spud_df.iterrows():
        if row[f"{xml_source}_error"] is True:
            print(f"Skipping {row} for now, tf not reading in")
            continue

        xml_path = pathlib.Path(row[xml_source])
        if "__" in xml_path.name:
            print(f"Skipping {row[xml_source]} for now, Station/network unknown")
            continue

        [xml_uid, network_id, station_id] = xml_path.stem.split("_")
        tf = load_xml_tf(xml_path)
        rr_type = get_rr_type(tf)
        remotes = get_remotes_from_tf(tf)
        if remotes:
            print(f"remotes: {remotes} ")
        all_stations = remotes + [station_id,]

        for station in all_stations:
            request_df = build_request_df([station,], network_id, start=None, end=None)
            print(request_df)
            fdsn_object = FDSN(mth5_version='0.2.0')
            fdsn_object.client = "IRIS"

            expected_file_name = DATA_PATH.joinpath(fdsn_object.make_filename(request_df))
            sub_coverage_df = local_data_coverage_df[local_data_coverage_df["filename"] == str(expected_file_name)]
            if len(sub_coverage_df):
                print(f"Already have data for {station}-{network_id}")
                print(f"{sub_coverage_df}")
                continue
            try:
                mth5_filename = fdsn_object.make_mth5_from_fdsn_client(request_df,
                                                                       interact=False,
                                                                       path=DATA_PATH)
                new_row = {"station_id": station,
                           "network_id": network_id,
                           "filename": mth5_filename,
                           "filesize": mth5_filename.stat().st_size,
                           "exception":"",
                           "error_message":""}
                local_data_coverage_df = local_data_coverage_df.append(new_row, ignore_index=True)
                local_data_coverage_df.to_csv(local_data_coverage_csv, index=False)
            except Exception as e:
                print("")
                new_row = {"station_id":station,
                           "network_id":network_id,
                           "filename":"", #expected_file_name
                           "filesize": "", #expected_file_name.stat().st_size,
                           "exception":e.__class__.__name__,
                           "error_message":e.args[0]}
                local_data_coverage_df = local_data_coverage_df.append(new_row, ignore_index=True)
                local_data_coverage_df.to_csv(local_data_coverage_csv, index=False)
            # ADD A ROW TO A DF AS WE GO
        #
        #
        #     print("get metadata")
        #     print("check what the filename should be")
        #     print("Check if the file exists")
        #     print("check if the file contains all the data")
        #     print("if not, pull new data, and add to the file if it exists")

    #
    #     print("NOW get the RRS")
    #
    # if controls["review_spud_xmls_01"]:
    #     results_df = review_spud_tfs(xml_source=SPUD_XML_COLUMN)
    # else:
    #     results_df = pd.read_csv(SPUD_XML_REVIEW_CSV_01)
    # results_df = results_df[results_df.error==False]
    # results_df.reset_index(drop=True, inplace=True)
    # results_df2 = get_mth5_data(results_df)
    # grouper = results_df2.groupby("survey")
    # # get data
    #
    # print("OK")
    # for i_row, row in SPUD_DF.iterrows():
    #     xml_path = pathlib.Path(row[SPUD_XML_COLUMN])
    #     spud_tf = load_xml_tf(xml_path)
    #     print(row[SPUD_XML_COLUMN])
    # pass

if __name__ == "__main__":
    main()