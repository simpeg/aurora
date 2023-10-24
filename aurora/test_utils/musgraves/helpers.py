# import aurora
# import mt_metadata
# import mth5
# import pandas as pd
# import pathlib
# import socket
# import warnings
#
# from aurora.config.config_creator import ConfigCreator
# from aurora.config import BANDS_TEST_FAST_FILE
# from aurora.pipelines.process_mth5 import process_mth5
# from aurora.pipelines.run_summary import RunSummary
# from aurora.test_utils.earthscope.widescale import WidesScaleTest
# from aurora.transfer_function.kernel_dataset import KernelDataset
# from mth5.mth5 import MTH5
# from mt_metadata.transfer_functions.processing.aurora.channel_nomenclature import CHANNEL_MAPS

import pandas as pd
import pathlib
import socket

def get_results_dir(ss_or_rr):
    hostname = socket.gethostname()
    print(f"hostname: {hostname}")
    if "gadi" in hostname:
        results_path = pathlib.Path("/scratch/tq84/kk9397/musgraves/aurora_results/level_1/single_station")
    elif hostname == "namazu":
        results_path = pathlib.Path("/home/kkappler/.cache/musgraves/aurora_results/level_1/single_station")

    if ss_or_rr.upper()=="SS":
        results_path = results_path.joinpath("single_station")
    elif ss_or_rr.upper()=="RR":
        results_path = results_path.joinpath("remote_reference")
    return results_path
def get_data_dir():

    hostname = socket.gethostname()
    print(f"hostname: {hostname}")
    if "gadi" in hostname:
        my80_path = pathlib.Path("/g/data/my80")
    elif hostname == "namazu":
        my80_path = pathlib.Path("/home/kkappler/data/gadi/g/data/my80")

    au_scope_mt_collection_path = my80_path.joinpath("AuScope_MT_collection")
    auslamp_path = au_scope_mt_collection_path.joinpath("AuScope_AusLAMP")
    musgraves_path = auslamp_path.joinpath("Musgraves_APY")
    data_dir = musgraves_path
    assert data_dir.exists()
    return data_dir

def get_musgraves_availability_df():
    data_dir = get_data_dir()
    all_mth5_files = data_dir.rglob("*h5")
    all_mth5_files_list = list(all_mth5_files)
    num_mth5 = len(all_mth5_files_list)
    print(f"Found {num_mth5} h5 files")
    levels = num_mth5 * [""]
    station_ids = num_mth5 * [""]
    territories = num_mth5 * [""]
    paths = num_mth5 * [""]

    for i_filepath, filepath in enumerate(all_mth5_files_list):
        levels[i_filepath] = str(filepath).split("level_")[1][0]
        station_ids[i_filepath] = filepath.stem
        territories[i_filepath] = str(filepath).split("Musgraves_APY/")[1][0:2]
        paths[i_filepath] = filepath
    df_dict = {"level": levels, "territory": territories, "station_id": station_ids, "path": paths}
    df = pd.DataFrame(data=df_dict)

    return df








def main():
    RESULTS_PATH = get_results_dir()
    df = get_musgraves_availability_df()
    print(df)

if __name__ == "__main__":
    main()