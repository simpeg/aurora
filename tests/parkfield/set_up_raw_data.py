"""
This script reads raw parkfield ascii data from csvs in the data repo and saves
it to an h5 file.  The flow is:

1. read in all the parkfield data to a dataframe,
2  read in all the hollister data to a dataframe,
3. Save the pkd dataframe to compressed h5
4. Save the hollister dataframe to compressed h5

This script is not intended to be run as a test.
The output compressed h5 is to be kept in the repo.

"""
import pandas as pd
from mth5_test_data.util import MTH5_TEST_DATA_DIR as DATA_DIR
from aurora.general_helper_functions import TEST_PATH

RAW_DATA_DIR = DATA_DIR.joinpath("iris/BK/2004/ATS")


TARGET_DATA_DIR = TEST_PATH.joinpath("parkfield", "data")
merged_h5 = TARGET_DATA_DIR.joinpath("pkd_sao_2004_272_00-02.h5")

channel_keys = ["hx", "hy", "ex", "ey"]
pkd_df = pd.read_csv(RAW_DATA_DIR.joinpath("PKD_272_00.csv"))
pkd_df = pkd_df[channel_keys]
sao_df = pd.read_csv(RAW_DATA_DIR.joinpath("SAO_272_00.csv"))
sao_df = sao_df[channel_keys]

h5_fn = merged_h5  # "test.h5"


for ch in channel_keys:
    pkd_df[ch].to_hdf(h5_fn, f"{ch}_pkd", complib="zlib", complevel=5)
    sao_df[ch].to_hdf(h5_fn, f"{ch}_sao", complib="zlib", complevel=5)
pkd_df_hx = pd.read_hdf(h5_fn, "hx_pkd")

with pd.HDFStore(merged_h5) as hdf:
    # This prints a list of all group names:
    print(hdf.keys())
