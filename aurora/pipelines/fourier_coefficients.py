"""
Supporting codes for building the FC level of the mth5

Here are the parameters that are defined via the mt_metadata fourier coefficients structures:
"anti_alias_filter": "default",
"bands",
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

Creating the decimations config requires a decision about decimation factors and the number of levels.
We have been getting this from the EMTF band setup file by default.  It is desirable to continue supporting this,
however, note that the EMTF band setup is really about processing, and not about making STFTs.

For the record, here is the legacy decimation config from EMTF, a.k.a. decset.cfg:
```
4     0      # of decimation level, & decimation offset
128  32.   1   0   0   7   4   32   1
1.0
128  32.   4   0   0   7   4   32   4
.2154  .1911   .1307   .0705
128  32.   4   0   0   7   4   32   4
.2154  .1911   .1307   .0705
128  32.   4   0   0   7   4   32   4
.2154  .1911   .1307   .0705
```

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


"""

# =============================================================================
# Imports
# =============================================================================

import mt_metadata.timeseries.time_period
import mth5.mth5
import pathlib

from aurora.pipelines.time_series_helpers import calibrate_stft_obj
from aurora.pipelines.time_series_helpers import prototype_decimate
from aurora.pipelines.time_series_helpers import run_ts_to_stft_scipy
from loguru import logger
from mth5.mth5 import MTH5
from mth5.utils.helpers import path_or_mth5_object
from mt_metadata.transfer_functions.processing.fourier_coefficients import (
    Decimation as FCDecimation,
)
from typing import List, Optional, Union

# =============================================================================
GROUPBY_COLUMNS = ["survey", "station", "sample_rate"]


def fc_decimations_creator(
    initial_sample_rate: float,
    decimation_factors: Optional[Union[list, None]] = None,
    max_levels: Optional[int] = 6,
    time_period: mt_metadata.timeseries.TimePeriod = None,
) -> list:
    """

    Creates mt_metadata FCDecimation objects that parameterize Fourier coefficient decimation levels.

    Note 1:  This does not yet work through the assignment of which bands to keep.  Refer to
    mt_metadata.transfer_functions.processing.Processing.assign_bands() to see how this was done in the past

    Parameters
    ----------
    initial_sample_rate: float
        Sample rate of the "level0" data -- usually the sample rate during field acquisition.
    decimation_factors: list (or other iterable)
        The decimation factors that will be applied at each FC decimation level
    max_levels: int
        The maximum number of decimation levels to allow
    time_period:

    Returns
    -------
    fc_decimations: list
        Each element of the list is an object of type
        mt_metadata.transfer_functions.processing.fourier_coefficients.Decimation,
        (a.k.a. FCDecimation).

        The order of the list corresponds the order of the cascading decimation
          - No decimation levels are omitted.
          - This could be changed in future by using a dict instead of a list,
          - e.g. decimation_factors = dict(zip(np.arange(max_levels), decimation_factors))

    """
    if not decimation_factors:
        # msg = "No decimation factors given, set default values to EMTF default values [1, 4, 4, 4, ..., 4]")
        # logger.info(msg)
        default_decimation_factor = 4
        decimation_factors = max_levels * [default_decimation_factor]
        decimation_factors[0] = 1

    # See Note 1
    fc_decimations = []
    for i_dec_level, decimation_factor in enumerate(decimation_factors):
        fc_dec = FCDecimation()
        fc_dec.decimation_level = i_dec_level
        fc_dec.id = f"{i_dec_level}"
        fc_dec.decimation_factor = decimation_factor
        if i_dec_level == 0:
            current_sample_rate = 1.0 * initial_sample_rate
        else:
            current_sample_rate /= decimation_factor
        fc_dec.sample_rate_decimation = current_sample_rate

        if time_period:
            if isinstance(time_period, mt_metadata.timeseries.time_period.TimePeriod):
                fc_dec.time_period = time_period
            else:
                msg = (
                    f"Not sure how to assign time_period with type {type(time_period)}"
                )
                logger.info(msg)
                raise NotImplementedError(msg)

        fc_decimations.append(fc_dec)

    return fc_decimations


@path_or_mth5_object
def add_fcs_to_mth5(
    m: MTH5, fc_decimations: Optional[Union[list, None]] = None
) -> None:
    """
    Add Fourier Coefficient Levels ot an existing MTH5.

    **Notes:**

    - This module computes the FCs differently than the legacy aurora pipeline. It uses scipy.signal.spectrogram. There is a test in Aurora to confirm that there are equivalent if we are not using fancy pre-whitening.

    - Nomenclature: "usssr_grouper" is the output of a group-by on unique {survey, station, sample_rate} tuples.

    Parameters
    ----------
    m: MTH5 object
        The mth5 file, open in append mode.
    fc_decimations: Union[str, None, List]
        This specifies the scheme to use for decimating the time series when building the FC layer.
        None: Just use default (something like four decimation levels, decimated by 4 each time say.
        String: Controlled Vocabulary, values are a work in progress, that will allow custom definition of the fc_decimations for some common cases. For example, say you have stored already decimated time
        series, then you want simply the zeroth decimation for each run, because the decimated time series live
        under another run container, and that will get its own FCs.  This is experimental.
        List: (**UNTESTED**) -- This means that the user thought about the decimations that they want to create and is
        passing them explicitly.  -- probably will need to be a dictionary actually, since this
        would get redefined at each sample rate.

    """
    # Group the channel summary by survey, station, sample_rate
    channel_summary_df = m.channel_summary.to_dataframe()
    usssr_grouper = channel_summary_df.groupby(GROUPBY_COLUMNS)
    logger.debug(f"Detected {len(usssr_grouper)} unique station-sample_rate instances")

    # loop over groups
    for (survey, station, sample_rate), usssr_group in usssr_grouper:
        msg = f"\n\n\nsurvey: {survey}, station: {station}, sample_rate {sample_rate}"
        logger.info(msg)
        station_obj = m.get_station(station, survey)
        run_summary = station_obj.run_summary

        # Get the FC decimation schemes if not provided
        if not fc_decimations:
            msg = "FC Decimations not supplied, creating defaults on the fly"
            logger.info(f"{msg}")
            fc_decimations = fc_decimations_creator(
                initial_sample_rate=sample_rate, time_period=None
            )
        elif isinstance(fc_decimations, str):
            if fc_decimations == "degenerate":
                fc_decimations = get_degenerate_fc_decimation(sample_rate)

        # TODO: Make this a function that can be done using df.apply()
        for i_run_row, run_row in run_summary.iterrows():
            logger.info(
                f"survey: {survey}, station: {station}, sample_rate {sample_rate}, i_run_row {i_run_row}"
            )
            # Access Run
            run_obj = m.from_reference(run_row.hdf5_reference)

            # Set the time period:
            # TODO: Should this be over-writing time period if it is already there?
            for fc_decimation in fc_decimations:
                fc_decimation.time_period = run_obj.metadata.time_period

            # Access the data to Fourier transform
            runts = run_obj.to_runts(
                start=fc_decimation.time_period.start,
                end=fc_decimation.time_period.end,
            )
            run_xrds = runts.dataset

            # access container for FCs
            fc_group = station_obj.fourier_coefficients_group.add_fc_group(
                run_obj.metadata.id
            )

            # If timing corrections were needed they could go here, right before STFT

            for i_dec_level, fc_decimation in enumerate(fc_decimations):
                if i_dec_level != 0:
                    # Apply decimation
                    run_xrds = prototype_decimate(fc_decimation, run_xrds)

                # check if this decimation level yields a valid spectrogram
                if not fc_decimation.is_valid_for_time_series_length(
                    run_xrds.time.shape[0]
                ):
                    logger.info(
                        f"Decimation Level {i_dec_level} invalid, TS of {run_xrds.time.shape[0]} samples too short"
                    )
                    continue

                stft_obj = run_ts_to_stft_scipy(fc_decimation, run_xrds)
                stft_obj = calibrate_stft_obj(stft_obj, run_obj)

                # Pack FCs into h5 and update metadata
                decimation_level = fc_group.add_decimation_level(
                    f"{i_dec_level}", decimation_level_metadata=fc_decimation
                )
                decimation_level.from_xarray(
                    stft_obj, decimation_level.metadata.sample_rate_decimation
                )
                decimation_level.update_metadata()
                fc_group.update_metadata()
    return


def get_degenerate_fc_decimation(sample_rate: float) -> list:
    """

    Makes a default fc_decimation list. WIP
    This "degnerate" config will only operate on the first decimation level.
    This is useful for testing but could be used in future if an MTH5 stored time series in decimation
    levels already as separate runs.

    Parameters
    ----------
    sample_rate: float
        The sample rate assocaiated with the time-series to convert to Spectrogram

    Returns
    -------
    output: list
        List has only one element which is of type mt_metadata.transfer_functions.processing.fourier_coefficients.Decimation.
    """
    output = fc_decimations_creator(
        sample_rate,
        decimation_factors=[
            1,
        ],
        max_levels=1,
    )
    return output


@path_or_mth5_object
def read_back_fcs(m: Union[MTH5, pathlib.Path, str], mode="r"):
    """
    This is mostly a helper function for tests.  It was used as a sanity check while debugging the FC files, and
    also is a good example for how to access the data at each level for each channel.

    The Time axis of the FC array will change from level to level, but the frequency axis will stay the same shape
    (for now -- storing all fcs by default)

    Args:
        m: pathlib.Path, str or an MTH5 object
            The path to an h5 file that we will scan the fcs from


    """
    channel_summary_df = m.channel_summary.to_dataframe()
    logger.debug(channel_summary_df)
    usssr_grouper = channel_summary_df.groupby(GROUPBY_COLUMNS)
    for (survey, station, sample_rate), usssr_group in usssr_grouper:
        logger.info(f"survey: {survey}, station: {station}, sample_rate {sample_rate}")
        station_obj = m.get_station(station, survey)
        fc_groups = station_obj.fourier_coefficients_group.groups_list
        logger.info(f"FC Groups: {fc_groups}")
        for run_id in fc_groups:
            fc_group = station_obj.fourier_coefficients_group.get_fc_group(run_id)
            dec_level_ids = fc_group.groups_list
            for dec_level_id in dec_level_ids:
                dec_level = fc_group.get_decimation_level(dec_level_id)
                xrds = dec_level.to_xarray(["hx", "hy"])
                msg = f"dec_level {dec_level_id}"
                msg = f"{msg} \n Time axis shape {xrds.time.data.shape}"
                msg = f"{msg} \n Freq axis shape {xrds.frequency.data.shape}"
                logger.debug(msg)

    return
