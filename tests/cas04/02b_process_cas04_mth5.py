"""
This may be moved to single_station processing example
2022-02-25
Time to start setting up the TFKernel.  We already have a prototype config class.
What is lacking is a DatasetDefinition
Note 1: Functionality of DatasetDefinition()
1. User can see all possible ways of processing the data
(possibly one list per station in run_summary)
2. User can get a list of local_station options
3. User can select local_station
-this can trigger a reduction of runs to only those that are from the local staion
and simultaneous runs at other stations
4. Given a local station, a list of possible reference stations can be generated
5. Given a remote reference station, a list of all relevent runs, truncated to
maximize coverage of the local station runs is generated
6. Given such a "restricted run list", runs can be dropped
7. Time interval endpoints can be changed

number of channels   5   number of frequencies   25
 orientations and tilts of each channel
    1    13.20     0.00 CAS  Hx
    2   103.20     0.00 CAS  Hy
    3     0.00    90.00 CAS  Hz
    4    13.20     0.00 CAS  Ex
    5   103.20     0.00 CAS  Ey

orientations and tilts of each channel
    1   -13.20     0.00 CAS  Hx
    2    86.70     0.00 CAS  Hy
    3     0.00    90.00 CAS  Hz
    4   -13.20     0.00 CAS  Ex
    5    86.70     0.00 CAS  Ey

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib

from aurora.config import BANDS_DEFAULT_FILE
from aurora.config.config_creator import ConfigCreator
from aurora.general_helper_functions import TEST_PATH
from aurora.pipelines.process_mth5 import process_mth5
from aurora.tf_kernel.dataset import channel_summary_to_run_summary
from aurora.tf_kernel.base import TransferFunctionKernel
from aurora.tf_kernel.dataset import DatasetDefinition
from aurora.tf_kernel.helpers import extract_run_summaries_from_mth5s
from mth5.utils.helpers import initialize_mth5


CAS04_PATH = TEST_PATH.joinpath("cas04")
CONFIG_PATH = CAS04_PATH.joinpath("config")
CONFIG_PATH.mkdir(exist_ok=True)
DATA_PATH = CAS04_PATH.joinpath("data")
H5_PATH = DATA_PATH.joinpath("8P_CAS04.h5")


def main():
    # identify the h5 files that you will use
    relevant_h5_list = [H5_PATH,]

    # get a merged run summary from all h5_list
    run_summary_df = extract_run_summaries_from_mth5s(relevant_h5_list)

    # Pass the run_summary to a DatasetDefinition
    dataset_definition = DatasetDefinition(df=run_summary_df)

    # Here you can show tools that DatasetDefinition could have
    # See Note 1 above: Functionality of DatasetDefinition()

    #To process only run "a":

    # tfk = TransferFunctionKernel(mth5_path=mth5_path)
    #
    # #Make quick channel_summary for rapid testing
    # summary_df = tfk.get_channel_summary(csv_path="channel_summary.csv")
    # print("COnvert Summary DF into a list of runs to process, with start and end times")
    # print(f"summary df={summary_df}")
    #
    # dataset_definition = DatasetDefinition(df=run_summary_df)
    # dataset_definition.from_mth5_channel_summary(summary_df)
    dataset_df = dataset_definition.restrict_runs_by_station("CAS04", ["a",],
                                                             overwrite=False)
    dataset_df["remote"] = False
    input_dataset = DatasetDefinition(df=dataset_df)
    cc = ConfigCreator()
    cc = ConfigCreator(config_path=CONFIG_PATH)
    pc = cc.create_run_processing_object(emtf_band_file=BANDS_DEFAULT_FILE,
                                    sample_rate=1.0
                                    )
    pc.stations.from_dataset_dataframe(input_dataset.df)
    for decimation in pc.decimations:
        decimation.estimator.engine = "RME"
    # a_only = process_mth5(pc, input_dataset, show_plot=False,
    #                   z_file_path="a.zss")
    # print("OKOKOKOKOKOKO")

    dataset_df = dataset_definition.restrict_runs_by_station("CAS04", ["a","b",],
                                                         overwrite=False)
    dataset_df["remote"] = False
    input_dataset = DatasetDefinition(df=dataset_df)
    pc = cc.create_run_processing_object(emtf_band_file=BANDS_DEFAULT_FILE,
                                         sample_rate=1.0
                                         )
    dsdf = input_dataset.df
    pc.stations.from_dataset_dataframe(dsdf)
    for decimation in pc.decimations:
        decimation.estimator.engine = "RME"

    # ab_only = process_mth5(pc, input_dataset, show_plot=False,
    #                       z_file_path="a_b.zss")

    print("A,B OK")

    from aurora.transfer_function.plot.comparison_plots import compare_two_z_files
    emtf_file = "emtf_results/CAS04-CAS04bcd_REV06-CAS04bcd_NVR08.zmm"
    compare_two_z_files(emtf_file,
                        "a_b.zss",
                        label1="emtf",
                        label2="a_b",
                        scale_factor1=1,
                        out_file="aab.png",
                        markersize=3,
                        #rho_ylims=[1e-20, 1e-6],
                        #rho_ylims=[1e-8, 1e6],
                        xlims=[1, 5000],
                        )
#     1    13.20     0.00 CAS  Hx
# 2   103.20     0.00 CAS  Hy
# 3     0.00     0.00 CAS  Hz
# 4    13.20     0.00 CAS  Ex
# 5   103.20     0.00 CAS  Ey

# m = initialize_mth5(mth5_path, mode="r")
    # #How do we know these are the run_ids?
    # for run_id in ["a", "b", "c", "d"]:
    #     process_run(m, run_id)
    # run_ids = ["b", "c"]
    # process_merged_runs(run_ids)
    # run_ids = ["b", "c", "d"]
    # process_merged_runs(run_ids)

    print("OK")

if __name__ == "__main__":
    main()