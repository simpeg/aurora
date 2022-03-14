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
from aurora.config.processing_config import RunConfig
from aurora.general_helper_functions import TEST_PATH
from aurora.pipelines.process_mth5 import process_mth5_run
from aurora.pipelines.process_mth5 import process_mth5_runs
from aurora.pipelines.process_mth5 import process_mth5_from_dataset_definition
from aurora.sandbox.mth5_channel_summary_helpers import channel_summary_to_dataset_definition
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


def make_processing_config_a(h5_path):#:station_id, run_id, sample_rate):
    """
    
    Returns
    -------

    """
    station_id = "CAS04"
    run_id = "a"
    sample_rate = 1.0
    config_maker = ConfigCreator()
    #decimation_factors = [1, 4, 4, 4, 4]
    decimation_factors = [1, 4, 4, 4,]
    config_path = config_maker.create_run_config(station_id,
                                                 run_id,
                                                 h5_path,
                                                 sample_rate,
                                                 decimation_factors=decimation_factors,
                                                 channel_scale_factors={
                                                     "CAS04":{"ex":1e6, "ey":1e6}} )
    return config_path

def process_run_a(config_path, mth5_path):
    tf_cls = process_mth5_run(
        config_path,
        "a",
        mth5_path=mth5_path,
        units="MT",
        show_plot=True,
        z_file_path="a.zss",
        return_collection=False,
    )

    tmp = tf_cls.write_tf_file(fn="cas04_ss_a.xml", file_type="emtfxml")
    print(f"would be nice if {tmp} was the xml")
    return

def process_run_a_new(config_path, mth5_path):
    """
    Just like process_run_a but in this case we use
    Parameters
    ----------
    config_path
    mth5_path

    Returns
    -------

    """
    tf_cls = process_mth5_runs(
        config_path,
        "a",
        mth5_path=mth5_path,
        units="MT",
        show_plot=True,
        z_file_path="a.zss",
        return_collection=False,
    )
    tmp = tf_cls.write_tf_file(fn="cas04_ss_a.xml", file_type="emtfxml")
    print(f"would be nice if {tmp} was the xml")
    return

def process_with_dataset_definition(config_path, mth5_path, dataset_dfn,
                                          show_plot=False, z_file_path="test.zss"):
    tf_cls = process_mth5_from_dataset_definition(
        config_path,
        dataset_dfn,
        mth5_path=mth5_path,
        units="MT",
        show_plot=show_plot,
        z_file_path=z_file_path,
        return_collection=False,
    )
    tmp = tf_cls.write_tf_file(fn="cas04_ss_a.xml", file_type="emtfxml")
    print(f"would be nice if {tmp} was the xml")
    return


def process_run(mth5_obj, run_id):
    """
    Parameters
    ----------
    run_id

    Returns
    -------

    """

    print("you need a processing config for this run")
    pass

def create_run_config(mth5_obj, run_id):
    """
    Need a kwarg for single station and for remote reference

    Parameters
    ----------
    mth5_obj
    run_id

    Returns
    -------

    """
    run_group = mth5_obj.get_run("CAS04", run_id)
    print(run_group.metadata)


    return

def process_merged_runs(run_ids):
    pass

def main():
    defn_df = test_make_dataset_definition()

    #To process only run "a":
    mth5_path = DATA_PATH.joinpath("ZU_CAS04.h5")#../backup/data/
    tfk = TransferFunctionKernel(mth5_path=mth5_path)

    #Make quick channel_summary for rapid testing
    summary_df = tfk.get_channel_summary(csv_path="channel_summary.csv")
    print("COnvert Summary DF into a list of runs to process, with start and end times")
    print(f"summary df={summary_df}")

    # Make Slow channel summary for thorough testing
    # summary_df2 = tfk.get_channel_summary()
    # print(f"summary df2={summary_df}")
    # print(f"OK? {summary_df==summary_df2}")

    dataset_definition = DatasetDefinition()
    dataset_definition.from_mth5_channel_summary(summary_df)
    dataset_df = dataset_definition.restrict_runs_by_station("CAS04", ["a",],
                                                             overwrite=True)
    config_path = make_processing_config_a(mth5_path)
    #qq = process_run_a(config_path, mth5_path)
    #ww = process_run_a_new(config_path, mth5_path)
    print("NEW PROCESSING OK")
    # process_with_dataset_definition(config_path, mth5_path,
    #                                       dataset_definition, show_plot=False,
    #                                       z_file_path="a.zss")
    print("DATASET DEFN A OK")
    dataset_definition = DatasetDefinition()
    dataset_definition.from_mth5_channel_summary(summary_df)
    run_ids = ["a", "b"]
#    run_ids = ["b", "c", "d"]
    dataset_df = dataset_definition.restrict_runs_by_station("CAS04", run_ids,
                                                             overwrite=True)
    ee = process_with_dataset_definition(config_path, mth5_path,
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