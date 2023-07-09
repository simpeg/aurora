"""
Based on process mth5, this will build the FC level of the mth5

The overall flow will be to start with a somthing like a processing_summary
and iterate over the rows, creating the FC levels.

We need to start with
1. A list of mth5 files
2. A FC-scheme.


FC-SCheme in the past has come from ConfigCreator
which takes a KernelDataset as input
KernelDataset at its core is a run_summary df with columns:
['survey', 'station_id', 'run_id', 'start', 'end', 'sample_rate',
       'input_channels', 'output_channels', 'channel_scale_factors',
       'mth5_path', 'remote', 'duration']

What I already have are:
['survey', 'station_id', (via grouper)
['id', 'start', 'end', 'components', 'measurement_type', 'sample_rate',
       'hdf5_reference']
['survey', 'station_id', 'run_id', 'start', 'end', 'sample_rate',
       'input_channels', 'output_channels', 'channel_scale_factors',
       'mth5_path', 'remote', 'duration']

Here are the parameters that are defined via the mt_metadata fourier coefficients structures
"anti_alias_filter": "default",
"bands"
"decimation.factor": 4.0,
"decimation.level": 2,
"decimation.method": "default",
"decimation.sample_rate": 0.0625,
"extra_pre_fft_detrend_type": "linear",
"prewhitening_type": "first difference",
"window.clock_zero_type": "ignore",
"window.num_samples": 128,
"window.overlap": 32,
"window.type": "boxcar"

Key to creating the decimations config is the decision about decimation factors and the number of levels
We have been getting this from the EMTF band setup file by default.  It is desireable to continue supporting this, however,
note that the EMTF band setup is really about processing, and not about making STFTs.

What we really want here is control of the decimation config.
This was controlled by decset.cfg which looks like this:
4     0      # of decimation level, & decimation offset
128  32.   1   0   0   7   4   32   1
1.0
128  32.   4   0   0   7   4   32   4
.2154  .1911   .1307   .0705
128  32.   4   0   0   7   4   32   4
.2154  .1911   .1307   .0705
128  32.   4   0   0   7   4   32   4
.2154  .1911   .1307   .0705

This essentially corresponds to a "Decimations Group" which is a list of decimations.
Related to the generation of FCs is the ARMA prewhitening (Issue #60) which was controlled in
EMTF with pwset.cfg
4    5             # of decimation level, # of channels
3 3 3 3 3
3 3 3 3 3
3 3 3 3 3
3 3 3 3 3

For now, lets make a Decimations list,
For now, lets continue supporting this as an interfce.


Note 1: Assumes application of cascading decimation, and that the
decimated data will be accessed from the previous decimation level.

Note 2: We can encounter cases where some runs can be decimated and others can not.
We need a way to handle this. For example, a short run may not yield any data from a
later decimation level. An attempt to handle this has been made in TF Kernel by
adding a is_valid_dataset column, associated with each run-decimation level pair.


Note 3: This point in the loop marks the interface between _generation_ of the FCs and
 their _usage_. In future the code above this comment would be pushed into
 create_fourier_coefficients() and the code below this would access those FCs and
 execute compute_transfer_function()


"""
# =============================================================================
# Imports
# =============================================================================
import copy

import numpy as np
import xarray as xr

from aurora.pipelines.time_series_helpers import calibrate_stft_obj
from aurora.pipelines.time_series_helpers import prototype_decimate
from aurora.pipelines.time_series_helpers import run_ts_to_stft_scipy


from aurora.pipelines.transfer_function_kernel import TransferFunctionKernel

from aurora.transfer_function.transfer_function_collection import (
    TransferFunctionCollection,
)
from aurora.transfer_function.TTFZ import TTFZ

from mth5.mth5 import MTH5
from mt_metadata.transfer_functions.core import TF
# from mt_metadata.transfer_functions.processing.fourier_coefficients import Channel
from mt_metadata.transfer_functions.processing.fourier_coefficients import Decimation
from mt_metadata.transfer_functions.processing.fourier_coefficients import FC


# =============================================================================
from aurora.general_helper_functions import TEST_PATH
from aurora.test_utils.synthetic.paths import DATA_PATH

def get_groupby_columns(m):
    groupby = ["survey", "station", "sample_rate"]
    if m.file_version == "0.2.0":
        groupby.insert(0, "experiement")
    return groupby

def decimation_and_stft_config_creator(initial_sample_rate, max_levels=6, decimation_factors=None):
    """
    Based on the number of samples in the run, we can compute the maximum number of valid decimation levels.
    This would re-use code in processing summary ... or we could just decimate until we cant anymore?

    You can provide soemthing like: decimation_info = {0: 1.0, 1: 4.0, 2: 4.0, 3: 4.0}
    :param initial_sample_rate:
    :param max_levels:
    :return:
    """
    if not decimation_factors:
        # set default values to EMTF default values [1, 4, 4, 4, ..., 4]
        decimation_factors = max_levels * [4]
        decimation_factors[0] = 1
        # add labels for the dec levels ... maybe not needed?
        # decimation_factors = dict(zip(np.arange(max_levels), decimation_factors))

    num_decimations = len(decimation_factors)


    # Refer to processing.Processing.assign_bands()
    decimation_and_stft_config = []
    for i_dec_level , decimation_factor in enumerate(decimation_factors):
        dd = Decimation()
        dd.decimation_level = i_dec_level
        dd.decimation_factor = decimation_factor
        if i_dec_level == 0:
            current_sample_rate = 1.0 * initial_sample_rate
        else:
            current_sample_rate /= decimation_factor
        dd.sample_rate_decimation = current_sample_rate
        print(dd.sample_rate_decimation)
        decimation_and_stft_config.append(dd)
    print("OKOKOK")
    print("WHAT ABOUT TIME PERIOD START AND END??? ")
    return decimation_and_stft_config



def generate_fcs_synthetic(mth5_path, min_num_stft_windows=2):
    m = MTH5()
    m.open_mth5(mth5_path)
    channel_summary_df = m.channel_summary.to_dataframe()
    print(m.channel_summary)
    #try one big groupby:
    groupby = get_groupby_columns(m)
    unique_station_sample_rate_grouper = channel_summary_df.groupby(groupby)
    # I cannot resist calling this a ussr_grouper
    ussr_grouper = unique_station_sample_rate_grouper
    print(f"DETECTED {len(ussr_grouper)} unique station-sample_rate instances")
    print("careful to groupby experiment as well for v0.2.0")
    for (survey, station, sample_rate), ussr_group in ussr_grouper:
        print(f"survey: {survey}, station: {station}, sample_rate {sample_rate}")
        station_obj = m.get_station(station, survey)
        run_summary = station_obj.run_summary

        drop_columns = ["start", "end"]
        for i_run_row, run_row in run_summary.iterrows():
            # Access Time Series Data
            run_obj = m.from_reference(run_row.hdf5_reference)
            runts = run_obj.to_runts()
            run_xrds = runts.dataset
            # access container for FCs
            fc_group = (station_obj.fourier_coefficients_group.add_fc_group(run_obj.metadata.id))
            print(" TIMING CORRECTIONS WOULD GO HERE ")

            # Get the FC schemes
            decimation_and_stft_configs = decimation_and_stft_config_creator(sample_rate)
            decimation_info = {x.decimation_level:x.decimation_factor for x in decimation_and_stft_configs}
            decimation_level_is_valid = True
            for i_dec_level, decimation_stft_obj in enumerate(decimation_and_stft_configs):
                if i_dec_level != 0:
                    print("APPLY DECIMATION")
                    run_xrds = prototype_decimate(decimation_stft_obj, run_xrds)
                    print("OK")
                else:
                    pass

                # Check that decimation_level_is_valid

                #n_samples = run_xrds.time.shape[0]
                required_num_samples = decimation_stft_obj.window.num_samples + (min_num_stft_windows - 1) * decimation_stft_obj.window.num_samples_advance
                if run_xrds.time.shape[0] < required_num_samples:
                    decimation_level_is_valid = False
                if not decimation_level_is_valid:
                    print(f"DECIMATION LEVEL {i_dec_level} found to be invalid")
                    print(f"DECIMATED TS HAS {run_xrds.time.shape[0]} samples")
                    print(f"NOT ENOUGH to get {min_num_stft_windows} of len {decimation_stft_obj.window.num_samples} and overlap {decimation_stft_obj.window.overlap}")
                    continue

                stft_obj = run_ts_to_stft_scipy(decimation_stft_obj, run_xrds)
                stft_obj = calibrate_stft_obj(stft_obj,run_obj)

                print("Pack FCs into h5 and update metadata")
                decimation_level = fc_group.add_decimation_level(f"{i_dec_level}")
                decimation_level.from_xarray(stft_obj)
                decimation_level.update_metadata()
                fc_group.update_metadata()

    m.close_mth5()
    return


def read_back_fcs(mth5_path):
    m = MTH5()
    m.open_mth5(mth5_path)
    channel_summary_df = m.channel_summary.to_dataframe()
    print(channel_summary_df)
    print("do some groupby stuffs")
    groupby = get_groupby_columns(m)
    ussr_grouper = channel_summary_df.groupby(groupby)
    for (survey, station, sample_rate), ussr_group in ussr_grouper:
        print(f"survey: {survey}, station: {station}, sample_rate {sample_rate}")
        station_obj = m.get_station(station, survey)
        print("Here are the fc groups")
        fc_groups = station_obj.fourier_coefficients_group.groups_list
        print(fc_groups)
        for run_id in fc_groups:
            fc_group = station_obj.fourier_coefficients_group.get_fc_group(run_id)
            dec_level_ids = fc_group.groups_list
            print("Expect TIME axis to change from level to level")
            print("Expect FREQ axis to stay same shape (for now -- storing whole enchelada")
            for dec_level_id in dec_level_ids:
                dec_level = fc_group.get_decimation_level(dec_level_id)
                print(f"dec_level {dec_level_id}")# channel_summary {dec_level.channel_summary}")
                xrds = dec_level.to_xarray(["hx", "hy"])
                print(f"TIME {xrds.time.data.shape}")
                print(f"FREQ {xrds.frequency.data.shape}")





def main():
    cfgs = decimation_and_stft_config_creator(1.0)
    synthetic_file_paths = list(DATA_PATH.glob("*.h5"))
    synthetic_file_paths = [x for x in synthetic_file_paths if "nan" not in str(x)]
    for mth5_path in synthetic_file_paths:
        generate_fcs_synthetic(mth5_path)
        read_back_fcs(mth5_path)
    print("se funciona!")

if __name__ == "__main__":
    main()


# Not sure we need this:
# print("MELT  Decimations") # Borrow from tfkernel line 60
# tmp = run_summary.copy(deep=True)
# tmp.drop(drop_columns, axis=1, inplace=True)
# id_vars = list(tmp.columns)
# for i_dec, dec_factor in decimation_info.items():
#     tmp[i_dec] = dec_factor
# tmp = tmp.melt(id_vars=id_vars, value_name="dec_factor", var_name="dec_level")
#
# sortby = ["id", "dec_level"] # might be nice to sort on "start" as well
# tmp.sort_values(by=sortby, inplace=True)
# tmp.reset_index(drop=True, inplace=True)
# tmp.drop("sample_rate", axis=1, inplace=True)  # not valid for decimated data
# sortby = ["survey", "station_id", "run_id", "start", "dec_level"]
# tmp.sort_values(by=sortby, inplace=True)
# tmp.reset_index(drop=True, inplace=True)
# tmp.drop("sample_rate", axis=1, inplace=True)  # not valid for decimated data
