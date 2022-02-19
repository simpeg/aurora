"""
THis may be moved to single_station processing example
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aurora.config.config_creator import ConfigCreator
from aurora.config.processing_config import RunConfig
from aurora.general_helper_functions import TEST_PATH
from aurora.pipelines.process_mth5 import process_mth5_run

from mth5.utils.helpers import initialize_mth5

CAS04_PATH = TEST_PATH.joinpath("cas04")
DATA_PATH = CAS04_PATH.joinpath("data")

def make_processing_config_a(h5_path):#:station_id, run_id, sample_rate):
    """
    
    Returns
    -------

    """
    station_id = "CAS04"
    run_id = "a"
    sample_rate = 1.0
    config_maker = ConfigCreator()
    decimation_factors = [1, 4, 4]
    config_path = config_maker.create_run_config(station_id,
                                                 run_id,
                                                 h5_path,
                                                 sample_rate,
                                                 decimation_factors=decimation_factors)
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
    mth5_path = DATA_PATH.joinpath("ZU_CAS04.h5")#../backup/data/
    config_path = make_processing_config_a(mth5_path)
    qq = process_run_a(config_path, mth5_path)
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