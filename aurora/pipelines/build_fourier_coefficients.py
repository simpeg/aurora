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

I think the way to move this forward is to follow process_mth5,
but try to initialize an mt_metadata FourierCoefficients object from the existing processing config
– this will point at any missing elements. Then replace the

run_ts_to_stft(stft_config, run_xrds)

with

run_ts_to_stft(fourier_coeffs_from_mtmetadata_config, run_xrds)

That is the piece that we can work on together on Friday. Once we have that,
then the “building” of the FC layer should be able to simply follow your test example in mth5.




"""
# =============================================================================
# Imports
# =============================================================================
import copy

import numpy as np
import xarray as xr

from aurora.pipelines.time_series_helpers import apply_prewhitening
from aurora.pipelines.time_series_helpers import apply_recoloring
from aurora.pipelines.time_series_helpers import truncate_to_clock_zero
from aurora.pipelines.time_series_helpers import calibrate_stft_obj
from aurora.pipelines.time_series_helpers import prototype_decimate
from aurora.pipelines.time_series_helpers import run_ts_to_stft
from aurora.pipelines.time_series_helpers import run_ts_to_stft_scipy


from aurora.pipelines.transfer_function_helpers import process_transfer_functions
from aurora.pipelines.transfer_function_kernel import TransferFunctionKernel

from aurora.transfer_function.transfer_function_collection import (
    TransferFunctionCollection,
)
from aurora.transfer_function.TTFZ import TTFZ

from mth5.mth5 import MTH5
from mt_metadata.transfer_functions.core import TF
from mt_metadata.transfer_functions.processing.fourier_coefficients import Channel
from mt_metadata.transfer_functions.processing.fourier_coefficients import Decimation
from mt_metadata.transfer_functions.processing.fourier_coefficients import FC


# =============================================================================
from aurora.general_helper_functions import TEST_PATH
from aurora.test_utils.synthetic.paths import DATA_PATH

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


# decimation_and_stft_config_creator(1.0)

def take_a_look_at_synthetic_data():
    min_num_stft_windows = 2
    synthetic_file_paths = list(DATA_PATH.glob("*.h5"))
    synthetic_file_paths = [x for x in synthetic_file_paths if "nan" not in str(x)]
    for mth5_path in synthetic_file_paths:
        m = MTH5()
        m.open_mth5(mth5_path)
        channel_summary_df = m.channel_summary.to_dataframe()
        print(m.channel_summary)
        unique_station_grouper = channel_summary_df.groupby(["survey", "station"])
        print(f"DETECTED {len(unique_station_grouper)} unique station instances")
        print("careful to groupby experiment as well for v0.2.0")

        for (survey, station), grp in unique_station_grouper:
            print(f"survey: {survey}, station: {station}")

            station_obj = m.get_station(station, survey)
            run_summary = station_obj.run_summary

            print("We should further group these by sample rate...")
            unique_station_sample_rate_grouper = run_summary.groupby(["sample_rate"])
            ussr_grouper = unique_station_sample_rate_grouper
            # I cannot resist calling this a ussr_grouper

            for sample_rate, ussr_group in ussr_grouper:
                drop_columns = ["start", "end"]
                for i_run_row, run_row in run_summary.iterrows():
                    # Access Time Series Data
                    run_obj = m.from_reference(run_row.hdf5_reference)
                    runts = run_obj.to_runts()
                    run_xrds = runts.dataset

                    print(" TIMING CORRECTIONS WOULD GO HERE ")

                    print("GET the FC SCHEMES")
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
                            continue

                        stft_obj = run_ts_to_stft_scipy(decimation_stft_obj, run_xrds)
                        stft_obj = calibrate_stft_obj(stft_obj,run_obj)

                        print("Pack FCs into h5 and update metadata")
                        fc_group = (station_obj.fourier_coefficients_group.add_fc_group(run_obj.metadata.id))
                        decimation_level = fc_group.add_decimation_level(f"{i_dec_level}")
                        decimation_level.from_xarray(stft_obj)
                        decimation_level.update_metadata()
                        fc_group.update_metadata()





        m.close_mth5()
    print("WOWWWWEEEEE")
    return

# decimation_level.channel_summary
# decimation_level.dataset_options
# decimation_level.update_metadata()
# decimation_level.to_xarray(["ex",])
# decimation_level.to_xarray(["ex","ey"])
def make_stft_objects(
    processing_config, i_dec_level, run_obj, run_xrds, units, station_id
):
    """
    Operates on a "per-run" basis

    This method could be modifed in a multiple station code so that it doesn't care
    if the station is "local" or "remote" but rather uses scale factors keyed by
    station_id

    Parameters
    ----------
    processing_config: mt_metadata.transfer_functions.processing.aurora.Processing
        Metadata about the processing to be applied
    i_dec_level: int
        The decimation level to process
    run_obj: mth5.groups.master_station_run_channel.RunGroup
        The run to transform to stft
    run_xrds: xarray.core.dataset.Dataset
        The data time series from the run to transform
    units: str
        expects "MT".  May change so that this is the only accepted set of units
    station_id: str
        To be deprecated, this information is contained in the run_obj as
        run_obj.station_group.metadata.id

    Returns
    -------
    stft_obj: xarray.core.dataset.Dataset
        Time series of calibrated Fourier coefficients per each channel in the run
    """
    stft_config = processing_config.get_decimation_level(i_dec_level)
    stft_obj = run_ts_to_stft(stft_config, run_xrds)
    # if stft_obj is None:
    #    # not enough data to FFT
    #    return stft_obj
    # stft_obj = run_ts_to_stft_scipy(stft_config, run_xrds)
    run_id = run_obj.metadata.id
    if station_id == processing_config.stations.local.id:
        scale_factors = processing_config.stations.local.run_dict[
            run_id
        ].channel_scale_factors
    elif station_id == processing_config.stations.remote[0].id:
        scale_factors = (
            processing_config.stations.remote[0].run_dict[run_id].channel_scale_factors
        )

    stft_obj = calibrate_stft_obj(
        stft_obj,
        run_obj,
        units=units,
        channel_scale_factors=scale_factors,
    )
    return stft_obj

# REMOVE THIS
def process_tf_decimation_level(
    config, i_dec_level, local_stft_obj, remote_stft_obj, units="MT"
):
    """
    Processing pipeline for a single decimation_level

    TODO: Add a check that the processing config sample rates agree with the data
    sampling rates otherwise raise Exception
    This method can be single station or remote based on the process cfg

    Parameters
    ----------
    config: mt_metadata.transfer_functions.processing.aurora.decimation_level.DecimationLevel
        Config for a single decimation level
    i_dec_level: int
        decimation level_id
        ?could we pack this into the decimation level as an attr?
    local_stft_obj: xarray.core.dataset.Dataset
        The time series of Fourier coefficients from the local station
    remote_stft_obj: xarray.core.dataset.Dataset or None
        The time series of Fourier coefficients from the remote station
    units: str
        one of ["MT","SI"]

    Returns
    -------
    transfer_function_obj : aurora.transfer_function.TTFZ.TTFZ
        The transfer function values packed into an object
    """
    frequency_bands = config.decimations[i_dec_level].frequency_bands_obj()
    transfer_function_obj = TTFZ(i_dec_level, frequency_bands, processing_config=config)

    transfer_function_obj = process_transfer_functions(
        config, i_dec_level, local_stft_obj, remote_stft_obj, transfer_function_obj
    )

    return transfer_function_obj

# REMOVE THIS
def export_tf(
    tf_collection,
    channel_nomenclature,
    station_metadata_dict={},
    survey_dict={},
):
    """
    This method may wind up being embedded in the TF class
    Assign transfer_function, residual_covariance, inverse_signal_power, station, survey

    Parameters
    ----------
    tf_collection: aurora.transfer_function.transfer_function_collection
    .TransferFunctionCollection
    station_metadata_dict: dict
    survey_dict: dict

    Returns
    -------
    tf_cls: mt_metadata.transfer_functions.core.TF
        Transfer function container
    """
    from mt_metadata.utils.list_dict import ListDict

    merged_tf_dict = tf_collection.get_merged_dict(channel_nomenclature)
    channel_nomenclature_dict = channel_nomenclature.to_dict()["channel_nomenclature"]
    tf_cls = TF(channel_nomenclature=channel_nomenclature_dict)
    renamer_dict = {"output_channel": "output", "input_channel": "input"}
    tmp = merged_tf_dict["tf"].rename(renamer_dict)
    tf_cls.transfer_function = tmp

    isp = merged_tf_dict["cov_ss_inv"]
    renamer_dict = {"input_channel_1": "input", "input_channel_2": "output"}
    isp = isp.rename(renamer_dict)
    tf_cls.inverse_signal_power = isp

    res_cov = merged_tf_dict["cov_nn"]
    renamer_dict = {"output_channel_1": "input", "output_channel_2": "output"}
    res_cov = res_cov.rename(renamer_dict)
    tf_cls.residual_covariance = res_cov

    tf_cls.station_metadata._runs = ListDict()
    tf_cls.station_metadata.from_dict(station_metadata_dict)
    tf_cls.survey_metadata.from_dict(survey_dict)
    return tf_cls


def update_dataset_df(i_dec_level, tfk):
    """
    This function has two different modes.  The first mode, initializes values in the
    array, and could be placed into TFKDataset.initialize_time_series_data()
    The second mode, decimates. The function is kept in pipelines becasue it calls
    time series operations.


    Notes:
    1. When iterating over dataframe, (i)ndex must run from 0 to len(df), otherwise
    get indexing errors.  Maybe reset_index() before main loop? or push reindexing
    into TF Kernel, so that this method only gets a cleanly indexed df, restricted to
    only the runs to be processed for this specific TF?
    2. When assigning xarrays to dataframe cells, df dislikes xr.Dataset,
    so we convert to DataArray before assignment


    Parameters
    ----------
    i_dec_level: int
        decimation level id, indexed from zero
    config: mt_metadata.transfer_functions.processing.aurora.decimation_level.DecimationLevel
        decimation level config

    Returns
    -------
    dataset_df: pd.DataFrame
        Same df that was input to the function but now has columns:


    """
    if i_dec_level == 0:
        pass
        # replaced with kernel_dataset.initialize_dataframe_for_processing()

        # APPLY TIMING CORRECTIONS HERE
    else:
        print(f"DECIMATION LEVEL {i_dec_level}")
        # See Note 1 top of module
        # See Note 2 top of module
        for i, row in tfk.dataset_df.iterrows():
            if not tfk.is_valid_dataset(row, i_dec_level):
                continue
            run_xrds = row["run_dataarray"].to_dataset("channel")
            decimation = tfk.config.decimations[i_dec_level].decimation
            decimated_xrds = prototype_decimate(decimation, run_xrds)
            tfk.dataset_df["run_dataarray"].at[i] = decimated_xrds.to_array("channel")

    print("DATASET DF UPDATED")
    return


def generate_fcs(
    config,
    tfk_dataset=None,
    units="MT",
    show_plot=False,
    save_fcs=True
):
    """
    This is the main method used to transform a processing_config,
    and a kernel_dataset into a transfer function estimate.



    Parameters
    ----------
    config: mt_metadata.transfer_functions.processing.aurora.Processing or path to json
        All processing parameters
    tfk_dataset: aurora.tf_kernel.dataset.Dataset or None
        Specifies what datasets to process according to config
    units: string
        "MT" or "SI".  To be deprecated once data have units embedded
    show_plot: boolean
        Only used for dev
    z_file_path: string or pathlib.Path
        Target path for a z_file output if desired
    return_collection : boolean
        return_collection=False will return an mt_metadata TF object
        return_collection=True will return
        aurora.transfer_function.transfer_function_collection.TransferFunctionCollection

    Returns
    -------
    tf: TransferFunctionCollection or mt_metadata TF
        The transfer funtion object
    tf_cls: mt_metadata.transfer_functions.TF
        TF object
    """
    # Initialize config and mth5s
    tfk = TransferFunctionKernel(dataset=tfk_dataset, config=config)
    tfk.make_processing_summary()
    tfk.validate()
    mth5_objs = tfk.config.initialize_mth5s()

    # Assign additional columns to dataset_df, populate with mth5_objs and xr_ts
    # ANY MERGING OF RUNS IN TIME DOMAIN WOULD GO HERE
    tfk.dataset.initialize_dataframe_for_processing(mth5_objs)

    print(
        f"Processing config indicates {len(tfk.config.decimations)} "
        f"decimation levels "
    )

    tf_dict = {}

    for i_dec_level, dec_level_config in enumerate(tfk.valid_decimations()):

        update_dataset_df(i_dec_level, tfk)

        # TFK 1: get clock-zero from data if needed
        if dec_level_config.window.clock_zero_type == "data start":
            dec_level_config.window.clock_zero = str(tfk.dataset_df.start.min())

        # Apply STFT to all runs
        local_stfts = []
        remote_stfts = []
        for i, row in tfk.dataset_df.iterrows():

            if not tfk.is_valid_dataset(row, i_dec_level):
                continue

            run_xrds = row["run_dataarray"].to_dataset("channel")
            run_obj = row.mth5_obj.from_reference(row.run_reference)
            stft_obj = make_stft_objects(
                tfk.config, i_dec_level, run_obj, run_xrds, units, row.station_id
            )

            if row.station_id == tfk.config.stations.local.id:
                local_stfts.append(stft_obj)
            elif row.station_id == tfk.config.stations.remote[0].id:
                remote_stfts.append(stft_obj)

        # Merge STFTs
        local_merged_stft_obj = xr.concat(local_stfts, "time")

        if tfk.config.stations.remote:
            remote_merged_stft_obj = xr.concat(remote_stfts, "time")
        else:
            remote_merged_stft_obj = None

        # FC TF Interface here (see Note #3)

        # Could downweight bad FCs here

        tf_obj = process_tf_decimation_level(
            tfk.config,
            i_dec_level,
            local_merged_stft_obj,
            remote_merged_stft_obj,
        )
        tf_obj.apparent_resistivity(tfk.config.channel_nomenclature, units=units)
        tf_dict[i_dec_level] = tf_obj

        if show_plot:
            from aurora.sandbox.plot_helpers import plot_tf_obj

            plot_tf_obj(tf_obj, out_filename="out")

    tf_collection = TransferFunctionCollection(
        tf_dict=tf_dict, processing_config=tfk.config
    )

    # local_run_obj = mth5_obj.get_run(run_config["local_station_id"], run_id)
    local_run_obj = tfk_dataset.get_run_object(0)

    if z_file_path:
        tf_collection.write_emtf_z_file(z_file_path, run_obj=local_run_obj)

    if return_collection:
        # this is now really only to be used for debugging and may be deprecated soon
        tfk_dataset.close_mths_objs()
        return tf_collection
    else:
        local_station_id = tfk.config.stations.local.id
        station_metadata = tfk_dataset.get_station_metadata(local_station_id)
        local_mth5_obj = mth5_objs[local_station_id]

        if local_mth5_obj.file_version == "0.1.0":
            survey_dict = local_mth5_obj.survey_group.metadata.to_dict()
        elif local_mth5_obj.file_version == "0.2.0":
            # this could be a method of tf_kernel.get_survey_dict()
            survey_id = tfk.dataset_df[
                tfk.dataset_df["station_id"] == local_station_id
            ].survey.unique()[0]
            survey_obj = local_mth5_obj.get_survey(survey_id)
            survey_dict = survey_obj.metadata.to_dict()

        tf_cls = export_tf(
            tf_collection,
            tfk.config.channel_nomenclature,
            station_metadata_dict=station_metadata.to_dict(),
            survey_dict=survey_dict,
        )
        tfk_dataset.close_mths_objs()
        return tf_cls


def main():
    take_a_look_at_synthetic_data()

if __name__ == "__main__":
    main()
# tmp.drop("sample_rate", axis=1, inplace=True)  # not valid for decimated data
# sortby = ["survey", "station_id", "run_id", "start", "dec_level"]
# tmp.sort_values(by=sortby, inplace=True)
# tmp.reset_index(drop=True, inplace=True)
# tmp.drop("sample_rate", axis=1, inplace=True)  # not valid for decimated data


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