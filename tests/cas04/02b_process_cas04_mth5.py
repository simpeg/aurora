"""
This may be moved to single_station processing example
2022-02-25
Time to start setting up the TFKernel.  We already have a prototype config class.
What is lacking is a DatasetDefinition
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib

from aurora.config.config_creator import ConfigCreator
from aurora.general_helper_functions import TEST_PATH
from aurora.pipelines.process_mth5 import process_mth5
from aurora.tf_kernel.dataset import channel_summary_to_run_summary
from aurora.tf_kernel.base import TransferFunctionKernel
from aurora.tf_kernel.dataset import DatasetDefinition

from mth5.utils.helpers import initialize_mth5


CAS04_PATH = TEST_PATH.joinpath("cas04")
DATA_PATH = CAS04_PATH.joinpath("data")

def test_make_dataset_definition():
    """
    ToDo: talk to Jared about validations here ... what is already being checked in
    mth5?
    Things to make sure of:
    1. That there are the same number of channels (and same channels exactly) in each run,
     otherwise we would need separate processing paramters for separate runs
    Returns
    -------

    """
    summary_csv = pathlib.Path("channel_summary.csv")
    df = pd.read_csv(summary_csv, parse_dates=["start", "end"])
    dataset_definition = channel_summary_to_dataset_definition(df)
    return dataset_definition




def process_merged_runs(run_ids):
    pass

def main():
    defn_df = test_make_dataset_definition()

    #To process only run "a":
    mth5_path = DATA_PATH.joinpath("8P_CAS04.h5")#../backup/data/
    tfk = TransferFunctionKernel(mth5_path=mth5_path)

    #Make quick channel_summary for rapid testing
    summary_df = tfk.get_channel_summary(csv_path="channel_summary.csv")
    print("COnvert Summary DF into a list of runs to process, with start and end times")
    print(f"summary df={summary_df}")

    dataset_definition = DatasetDefinition()
    dataset_definition.from_mth5_channel_summary(summary_df)
    dataset_df = dataset_definition.restrict_runs_by_station("CAS04", ["a",],
                                                             overwrite=True)

    dataset_definition = DatasetDefinition()
    dataset_definition.from_mth5_channel_summary(summary_df)
    run_ids = ["a", "b"]
#    run_ids = ["b", "c", "d"]
    dataset_df = dataset_definition.restrict_runs_by_station("CAS04", run_ids,
                                                             overwrite=True)

    ##NEED A CONFIG HERE, THIS WAS NEVER RUN WITH NEW PROCESSING CONFIG
    ee = process_mth5(config, mth5_path,
                                        dataset_definition, show_plot=False,
                                        z_file_path="a_b.zss")
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