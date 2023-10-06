"""
Supporting codes for building the FC level of the mth5


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

Key to creating the decimations config is the decision about decimation factors and the number of levels.
We have been getting this from the EMTF band setup file by default.  It is desireable to continue supporting this,
however, note that the EMTF band setup is really about processing, and not about making STFTs.

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


from aurora.pipelines.time_series_helpers import calibrate_stft_obj
from aurora.pipelines.time_series_helpers import prototype_decimate
from aurora.pipelines.time_series_helpers import run_ts_to_stft_scipy
from mth5.mth5 import MTH5
import mt_metadata.timeseries.time_period
from mt_metadata.transfer_functions.processing.fourier_coefficients import (
    Decimation as FCDecimation,
)


# =============================================================================
FILE_VERSION = "you need to set this, and ideally cycle over 0.1.0, 0.2.0"
DEFAULT_TIME = "1980-01-01T00:00:00+00:00"
GROUPBY_COLUMNS = ["survey", "station", "sample_rate"]  # See Question 1


def decimation_and_stft_config_creator(
    initial_sample_rate, max_levels=6, decimation_factors=None, time_period=None
):
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

    # See Note 1
    decimation_and_stft_config = []
    for i_dec_level, decimation_factor in enumerate(decimation_factors):
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


def add_fcs_to_mth5(mth5_path, decimation_and_stft_configs=None):
    """
    usssr_grouper: output of a groupby on unique {survey, station, sample_rate} tuples

    Args:
        mth5_path: str or pathlib.Path
            Where the mth5 file is locatid
        decimation_and_stft_configs:

    Returns:

    """
    m = MTH5()
    m.open_mth5(mth5_path)
    channel_summary_df = m.channel_summary.to_dataframe()

    usssr_grouper = channel_summary_df.groupby(GROUPBY_COLUMNS)
    print(f"DETECTED {len(usssr_grouper)} unique station-sample_rate instances")

    for (survey, station, sample_rate), usssr_group in usssr_grouper:
        print(f"\n\n\nsurvey: {survey}, station: {station}, sample_rate {sample_rate}")
        station_obj = m.get_station(station, survey)
        run_summary = station_obj.run_summary

        # Get the FC schemes
        if not decimation_and_stft_configs:
            msg = "FC config not supplied, using default, creating on the fly"
            print(f"{msg}")
            decimation_and_stft_configs = decimation_and_stft_config_creator(
                sample_rate, time_period=None
            )

        # Make this a function that can be done using df.apply()
        # I wonder if daskifiying that will cause issues with multiple threads trying to
        # write to the hdf5 file -- will need testing
        for i_run_row, run_row in run_summary.iterrows():
            print(
                f"survey: {survey}, station: {station}, sample_rate {sample_rate}, i_run_row {i_run_row}"
            )
            # Access Run
            run_obj = m.from_reference(run_row.hdf5_reference)

            # Set the time period:
            for decimation_and_stft_config in decimation_and_stft_configs:
                decimation_and_stft_config.time_period = run_obj.metadata.time_period

            runts = run_obj.to_runts(
                start=decimation_and_stft_config.time_period.start,
                end=decimation_and_stft_config.time_period.end,
            )
            # runts = run_obj.to_runts() # skip setting time_period explcitly

            run_xrds = runts.dataset
            # access container for FCs
            fc_group = station_obj.fourier_coefficients_group.add_fc_group(
                run_obj.metadata.id
            )

            print(" TIMING CORRECTIONS WOULD GO HERE ")

            for i_dec_level, decimation_stft_obj in enumerate(
                decimation_and_stft_configs
            ):
                if i_dec_level != 0:
                    # Apply decimation
                    run_xrds = prototype_decimate(decimation_stft_obj, run_xrds)
                print(f"type decimation_stft_obj = {type(decimation_stft_obj)}")
                if not decimation_stft_obj.is_valid_for_time_series_length(
                    run_xrds.time.shape[0]
                ):
                    print(
                        f"Decimation Level {i_dec_level} invalid, TS of {run_xrds.time.shape[0]} samples too short"
                    )
                    continue

                stft_obj = run_ts_to_stft_scipy(decimation_stft_obj, run_xrds)
                stft_obj = calibrate_stft_obj(stft_obj, run_obj)

                # print("Pack FCs into h5 and update metadata")
                decimation_level = fc_group.add_decimation_level(f"{i_dec_level}")
                decimation_level.from_xarray(stft_obj)
                decimation_level.update_metadata()
                fc_group.update_metadata()

    m.close_mth5()
    return


def read_back_fcs(mth5_path):
    """
    This is mostly a helper function for tests.  It was used as a sanity check while debugging the FC files, and
    also is a good example for how to access the data at each level for each channel.

    The Time axis of the FC array will change from level to level, but the frequency axis will stay the same shape
    (for now -- storing all fcs by default)

    Args:
        mth5_path: str or pathlib.Path
            The path to an h5 file that we will scan the fcs from

    Returns:

    """
    m = MTH5()
    m.open_mth5(mth5_path)
    channel_summary_df = m.channel_summary.to_dataframe()
    print(channel_summary_df)
    usssr_grouper = channel_summary_df.groupby(GROUPBY_COLUMNS)
    for (survey, station, sample_rate), usssr_group in usssr_grouper:
        print(f"survey: {survey}, station: {station}, sample_rate {sample_rate}")
        station_obj = m.get_station(station, survey)
        fc_groups = station_obj.fourier_coefficients_group.groups_list
        print(f"FC Groups: {fc_groups}")
        for run_id in fc_groups:
            fc_group = station_obj.fourier_coefficients_group.get_fc_group(run_id)
            dec_level_ids = fc_group.groups_list
            for dec_level_id in dec_level_ids:
                dec_level = fc_group.get_decimation_level(dec_level_id)
                print(
                    f"dec_level {dec_level_id}"
                )  # channel_summary {dec_level.channel_summary}")
                xrds = dec_level.to_xarray(["hx", "hy"])
                print(f"Time axis shape {xrds.time.data.shape}")
                print(f"Freq axis shape {xrds.frequency.data.shape}")
    return True


def main():
    pass


if __name__ == "__main__":
    main()
