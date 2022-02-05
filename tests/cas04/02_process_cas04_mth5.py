"""
THis may be moved to single_station processing example
"""
from aurora.config.processing_config import RunConfig
from aurora.general_helper_functions import TEST_PATH

from mth5.utils.helpers import initialize_mth5

CAS04_PATH = TEST_PATH.joinpath("cas04")
DATA_PATH = CAS04_PATH.joinpath("data")

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
    m = initialize_mth5(mth5_path, mode="r")
    #How do we know these are the run_ids?
    for run_id in ["a", "b", "c", "d"]:
        process_run(m, run_id)
    run_ids = ["b", "c"]
    process_merged_runs(run_ids)
    run_ids = ["b", "c", "d"]
    process_merged_runs(run_ids)

    print("OK")

if __name__ == "__main__":
    main()