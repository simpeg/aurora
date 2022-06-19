"""
This may be moved to single_station processing example
2022-02-25
Time to start setting up the TFKernel.  We already have a prototype config class.
What is lacking is a Dataset() class
Note 1: Functionality of Dataset()
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
from aurora.tf_kernel.base import TransferFunctionKernel
from aurora.tf_kernel.dataset import Dataset as TFKDataset
from aurora.tf_kernel.helpers import extract_run_summaries_from_mth5s
from aurora.transfer_function.plot.comparison_plots import compare_two_z_files
from mth5.utils.helpers import initialize_mth5


CAS04_PATH = TEST_PATH.joinpath("cas04")
CONFIG_PATH = CAS04_PATH.joinpath("config")
CONFIG_PATH.mkdir(exist_ok=True)
DATA_PATH = CAS04_PATH.joinpath("data")
H5_PATH = DATA_PATH.joinpath("8P_CAS04.h5")


def process_runlist(run_list, return_collection=False):
    # identify the h5 files that you will use
    relevant_h5_list = [H5_PATH,]

    # get a merged run summary from all h5_list
    run_summary_df = extract_run_summaries_from_mth5s(relevant_h5_list)

    # Pass the run_summary to a Dataset class
    tfk_dataset = TFKDataset(df=run_summary_df)

    # Here you can show tools that TFKDataset could have
    # See Note 1 above: Functionality of TFKDataset()



    # tfk = TransferFunctionKernel(mth5_path=mth5_path)
    # #Make quick channel_summary for rapid testing
    # summary_df = tfk.get_channel_summary(csv_path="channel_summary.csv")

    dataset_df = tfk_dataset.restrict_runs_by_station("CAS04", run_list,
                                                             overwrite=False)
    dataset_df["remote"] = False
    input_dataset = TFKDataset(df=dataset_df)
    cc = ConfigCreator()
    cc = ConfigCreator(config_path=CONFIG_PATH)
    pc = cc.create_run_processing_object(emtf_band_file=BANDS_DEFAULT_FILE,
                                         sample_rate=1.0
                                         )
    pc.stations.from_dataset_dataframe(input_dataset.df)
    pc.validate()
    z_file_name = f"{'_'.join(run_list)}.zss"
    tf_result = process_mth5(pc, input_dataset, show_plot=False,
                             z_file_path=z_file_name,
                          return_collection=return_collection)
    if not return_collection:
        xml_file_name = f"{'_'.join(run_list)}.xml"
        tf_result.write_tf_file(fn=xml_file_name, file_type="emtfxml")
    return tf_result




def compare_results(run_list):
    emtf_file = "emtf_results/CAS04-CAS04bcd_REV06-CAS04bcd_NVR08.zmm"
    z_file_name = f"{'_'.join(run_list)}.zss"
    compare_two_z_files(emtf_file,
                    z_file_name,
                    label1="emtf",
                    label2="a_b",
                    scale_factor1=1,
                    out_file="aab.png",
                    markersize=3,
                    #rho_ylims=[1e-20, 1e-6],
                    #rho_ylims=[1e-8, 1e6],
                    xlims=[1, 5000],
                    )
def process_all_runs_individually():
    all_runs = ["a", "b", "c", "d"]
    #all_runs = ["c",]
    for run in all_runs:
        run_list = [run,]
        process_runlist(run_list)
        compare_results(run_list)

def see_channel_summary():
    h5_path = DATA_PATH.joinpath("8P_CAS04_CAV07_NVR11_REV06.h5")
    mth5_obj = initialize_mth5(h5_path=h5_path, )
    mth5_obj.channel_summary.summarize()
    #mth5_obj.close_mth5()
    summary_df = mth5_obj.channel_summary.to_dataframe()
    print(summary_df)

def main():
    see_channel_summary()
    #process_all_runs_individually()

    run_list = ["b", "c", "d",]
    #process_runlist(run_list)

    compare_results(run_list)
    print("OK")

if __name__ == "__main__":
    main()
