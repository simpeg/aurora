"""
Supporting codes for building the FC level of the mth5

When this is done the following new files exist:
1. tests/synthetic/test_add_fourier_coefficients.py


20230722: will make this into a test on fc branch.

Flow:
1. Assert that synthetic data exist, (and build if they dont)
- you should know what you expect here ...  test1,2,3.h5 and test12rr.h5
2. Two ways to prepare the FC instructions (processing configs)
- a) use the mt_metadata processing fourier_coefficients structures explictly
- b) use the default processing configs you already use for processing, and
extract type (a) cfgs from these (the machinery to do this should exist already)
3. Loop over files and generate FCs
4. Compare fc values against some archived values

We need to start with
1. A list of mth5 files
2. A FC-scheme.

FC-SCheme in the past has come from ConfigCreator, which takes a KernelDataset as input.

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


ToDo:
1. Create tests and put them in aurora/tests/___where_exactly__?
2. Make one test generate the decimation_and_stft_config with default values from
the decimation_and_stft_config_creator method here
3. Make another test take the existing aurora processing config and transform it to
decimation_and_stft_config

Tools for this are already in the FourierCoefficients branch.

Questions:
1. Shouldn;t there be an experiment column in the channel_summary dataframe for a v0.2.0 file?
GROUPBY_COLUMNS = ["survey", "station", "sample_rate"]
If I use ["experiment", "survey", "station", "sample_rate"] instead (for a v0.2.0 file) encounter KeyError.

2. How to assign default values to Decimation.time_period?
Usually we will want to convert the entire run, so these should be assigned
during processing when we knwo the run extents.  Thus the
"""
# =============================================================================
# Imports
# =============================================================================
import copy
import unittest

import mt_metadata.timeseries.time_period
import numpy as np
import xarray as xr

from aurora.test_utils.synthetic.make_processing_configs import create_test_run_config
from aurora.test_utils.synthetic.make_mth5_from_asc import create_test1_h5
from aurora.test_utils.synthetic.make_mth5_from_asc import create_test2_h5
from aurora.test_utils.synthetic.make_mth5_from_asc import create_test3_h5
from aurora.test_utils.synthetic.make_mth5_from_asc import create_test12rr_h5
from aurora.test_utils.synthetic.make_mth5_from_asc import main as make_all_h5

from aurora.pipelines.time_series_helpers import calibrate_stft_obj
from aurora.pipelines.time_series_helpers import prototype_decimate
from aurora.pipelines.time_series_helpers import run_ts_to_stft_scipy
from aurora.pipelines.transfer_function_kernel import TransferFunctionKernel
from aurora.transfer_function.transfer_function_collection import (
    TransferFunctionCollection,
)
from aurora.transfer_function.TTFZ import TTFZ
from mth5.mth5 import MTH5
from mth5.helpers import close_open_files
from mt_metadata.transfer_functions.core import TF
from mt_metadata.transfer_functions.processing.fourier_coefficients import Decimation as FCDecimation


# =============================================================================
from aurora.general_helper_functions import TEST_PATH
from aurora.pipelines.run_summary import RunSummary
from aurora.test_utils.synthetic.paths import DATA_PATH
from aurora.transfer_function.kernel_dataset import KernelDataset

FILE_VERSION = "you need to set this, and ideally cycle over 0.1.0, 0.2.0"
DEFAULT_TIME = "1980-01-01T00:00:00+00:00"
GROUPBY_COLUMNS = ["survey", "station", "sample_rate"] # ["experiment", "survey", "station", "sample_rate"]
# ? Shouldn't we have an "experiment" column here? results in KeyError when using it.

# tests/synthetic/test_add_fourier_coefficients.py
class TestAddFourierCoefficientsToSyntheticData(unittest.TestCase):
    """
    Runs several synthetic processing tests from config creation to tf_cls.

    """

    def setUpClass(self):
        print("make synthetic data")
        close_open_files()
        self.file_version = "0.1.0"
        self.mth5_path_1 = create_test1_h5(file_version=self.file_version)
        # self.mth5_path_2 = create_test2_h5(file_version=self.file_version)
        # mth5_path_3 = create_test3_h5(file_version=self.file_version)
        # mth5_path_12rr = create_test12rr_h5(file_version=self.file_version)
        #self.mth5_paths = [mth5_path_1, mth5_path_2, mth5_path_3, mth5_path_12rr]
        # logging.getLogger("matplotlib.font_manager").disabled = True
        # logging.getLogger("matplotlib.ticker").disabled = True

    def test_1(self):
        mth5_path = self.mth5_path_1
        station_id = "test1"
        mth5_paths = [ mth5_path, ]
        run_summary = RunSummary()
        run_summary.from_mth5s(mth5_paths)
        tfk_dataset = KernelDataset()
        tfk_dataset.from_run_summary(run_summary, station_id)
        processing_config = create_test_run_config(station_id, tfk_dataset)
        fc_decimations = [x.to_fc_decimation("local") for x in processing_config.decimations]
        add_fcs_to_mth5(mth5_path, decimation_and_stft_configs=fc_decimations)
        # Now build the layer:

#        tfc = process_mth5(processing_config, tfk_dataset=tfk_dataset, save_fcs=True)
        #return tfc
        print("OK")
        print("NEXT STEP is add a Tap-Point into existing processing to create these levels")
        print("NEXT STEP AFTER THAT is to try processing data from the FC LEVEL")

        pass

# Belongs in time_series/fourier_coefficients.py
def decimation_and_stft_config_creator(initial_sample_rate, max_levels=6, decimation_factors=None, time_period=None):
    """
    Based on the number of samples in the run, we can compute the maximum number of valid decimation levels.
    This would re-use code in processing summary ... or we could just decimate until we cant anymore?

    You can provide soemthing like: decimation_info = {0: 1.0, 1: 4.0, 2: 4.0, 3: 4.0}

    Note 1:  This does not yet work through the assignment of which bands to keep.  Refer to
    mt_metadata.transfer_functions.processing.Processing.assign_bands() to see how this was done in the past

    Args:
        initial_sample_rate:
        max_levels:
        decimation_factors:
        time_period:

    Returns:
        decimation_and_stft_config: list
            Each element of the list is a Decimation() object.  The order of the list implies the order of the cascading
            decimation (thus no decimation levels are omitted).  This could be changed in future by using a dict
            instead of a list, e.g. decimation_factors = dict(zip(np.arange(max_levels), decimation_factors))

    """
    if not decimation_factors:
        # msg = "No decimation factors given, set default values to EMTF default values [1, 4, 4, 4, ..., 4]")
        # logger.info(msg)
        default_decimation_factor = 4
        decimation_factors = max_levels * [default_decimation_factor]
        decimation_factors[0] = 1

    num_decimations = len(decimation_factors)

    # See Note 1
    decimation_and_stft_config = []
    for i_dec_level , decimation_factor in enumerate(decimation_factors):
        dd = FCDecimation()
        dd.decimation_level = i_dec_level
        dd.decimation_factor = decimation_factor
        if i_dec_level == 0:
            current_sample_rate = 1.0 * initial_sample_rate
        else:
            current_sample_rate /= decimation_factor
        dd.sample_rate_decimation = current_sample_rate

        if time_period:
            if isinstance(mt_metadata.timeseries.time_period.TimePeriod, time_period):
                dd.time_period = time_period
            else:
                print(f"Not sure how to assign time_period with {time_period}")
                raise NotImplementedError

        decimation_and_stft_config.append(dd)

    return decimation_and_stft_config


# MOVE THIS TO pipelines/fourier_coefficients.py
def add_fcs_to_mth5(mth5_path, decimation_and_stft_configs=None):
    """

    Args:
        mth5_path: str or pathlib.Path
            Where the mth5 file is locatid
        decimation_and_stft_configs:

    Returns:

    """
    m = MTH5()
    m.open_mth5(mth5_path)
    channel_summary_df = m.channel_summary.to_dataframe()


    unique_station_sample_rate_grouper = channel_summary_df.groupby(GROUPBY_COLUMNS)# Can't resist calling this ussr_grouper
    ussr_grouper = unique_station_sample_rate_grouper
    print(f"DETECTED {len(ussr_grouper)} unique station-sample_rate instances")
    print("Need to groupby experiment as well for v0.2.0?? See Question 1 at top of module")

    for (survey, station, sample_rate), ussr_group in ussr_grouper:
        print(f"\n\n\nsurvey: {survey}, station: {station}, sample_rate {sample_rate}")
        station_obj = m.get_station(station, survey)
        run_summary = station_obj.run_summary

        # Get the FC schemes
        if not decimation_and_stft_configs:
            print("FC config not supplied, using default, creating on the fly")
            decimation_and_stft_configs = decimation_and_stft_config_creator(sample_rate, time_period=None)
            decimation_info = {x.decimation_level: x.decimation_factor for x in decimation_and_stft_configs}

        # Make this a function that can be done using df.apply()
        # I wonder if daskifiying that will cause issues with multiple threads trying to
        # write to the hdf5 file -- will need testing
        for i_run_row, run_row in run_summary.iterrows():
            print(f"survey: {survey}, station: {station}, sample_rate {sample_rate}, i_run_row {i_run_row}")
            # Access Run
            run_obj = m.from_reference(run_row.hdf5_reference)

            # Set the time period:
            for decimation_and_stft_config in decimation_and_stft_configs:
                decimation_and_stft_config.time_period = run_obj.metadata.time_period

            runts = run_obj.to_runts(start=decimation_and_stft_config.time_period.start,
                                     end=decimation_and_stft_config.time_period.end)
            # runts = run_obj.to_runts() # skip setting time_period explcitly

            run_xrds = runts.dataset
            # access container for FCs
            fc_group = (station_obj.fourier_coefficients_group.add_fc_group(run_obj.metadata.id))

            print(" TIMING CORRECTIONS WOULD GO HERE ")

            for i_dec_level, decimation_stft_obj in enumerate(decimation_and_stft_configs):
                if i_dec_level != 0:
                    # Apply decimation
                    run_xrds = prototype_decimate(decimation_stft_obj, run_xrds)
                print(f"type decimation_stft_obj = {type(decimation_stft_obj)}")
                if not decimation_stft_obj.is_valid_for_time_series_length(run_xrds.time.shape[0]):
                    print(f"Decimation Level {i_dec_level} invalid, TS of {run_xrds.time.shape[0]} samples too short")
                    continue

                stft_obj = run_ts_to_stft_scipy(decimation_stft_obj, run_xrds)
                stft_obj = calibrate_stft_obj(stft_obj,run_obj)

                # print("Pack FCs into h5 and update metadata")
                decimation_level = fc_group.add_decimation_level(f"{i_dec_level}")
                decimation_level.from_xarray(stft_obj)
                decimation_level.update_metadata()
                fc_group.update_metadata()

    m.close_mth5()
    return

# pipelines/fourier_coefficients.py
def read_back_fcs(mth5_path):
    """
    This is mostly a helper function for tests

    Args:
        mth5_path: str or pathlib.Path
            The path to an h5 file that we will scan the fcs from

    Returns:

    """
    m = MTH5()
    m.open_mth5(mth5_path)
    channel_summary_df = m.channel_summary.to_dataframe()
    print(channel_summary_df)
    ussr_grouper = channel_summary_df.groupby(GROUPBY_COLUMNS)
    for (survey, station, sample_rate), ussr_group in ussr_grouper:
        print(f"survey: {survey}, station: {station}, sample_rate {sample_rate}")
        station_obj = m.get_station(station, survey)
        fc_groups = station_obj.fourier_coefficients_group.groups_list
        print(f"FC Groups: {fc_groups}")
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




def test_decimation_and_stft_config_creator():
    cfgs = decimation_and_stft_config_creator(1.0)
    return cfgs

def test_can_add_fcs_to_synthetic_mth5s(decimation_and_stft_configs=None):
    synthetic_file_paths = list(DATA_PATH.glob("*.h5"))
    synthetic_file_paths = [x for x in synthetic_file_paths if "nan" not in str(x)]
    # Here are the synthetic files for which this is currently passing tests
    # [PosixPath('/home/kkappler/software/irismt/aurora/tests/synthetic/data/test1.h5'),
    #  PosixPath('/home/kkappler/software/irismt/aurora/tests/synthetic/data/test2.h5'),
    #  PosixPath('/home/kkappler/software/irismt/aurora/tests/synthetic/data/test3.h5'),
    #  PosixPath('/home/kkappler/software/irismt/aurora/tests/synthetic/data/test12rr.h5')]

    for mth5_path in synthetic_file_paths:
        add_fcs_to_mth5(mth5_path, decimation_and_stft_configs=decimation_and_stft_configs)
        read_back_fcs(mth5_path)
    return

def main():
    # cfgs = test_decimation_and_stft_config_creator()
    # test_can_add_fcs_to_synthetic_mth5s()
    test_case = TestAddFourierCoefficientsToSyntheticData()
    test_case.setUpClass()
    test_case.test_1()
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
