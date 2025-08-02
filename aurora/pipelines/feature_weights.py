from loguru import logger
from mt_metadata.transfer_functions.processing.aurora.decimation_level import (
    DecimationLevel as AuroraDecimationLevel,
)
from mth5.processing import KernelDataset

import pandas as pd
import xarray as xr


def extract_features(
    dec_level_config: AuroraDecimationLevel, tfk_dataset: KernelDataset
) -> pd.DataFrame:

    """
        Temporal place holder.

        In the future, features will be stored in the MTH5 and accessible as df or xr.
        Here we compute them in place -- its inefficient, but will allow as faster
        end-to-end prototype.

        Note that the container with the

        Development Note #4:
            Features that will be used for FC-weighting are expected to share a time axis with the FC data.
            (This could possibly be relaxed in the future with some interpolation, but for now, enforce it).
            This means that striding window features derived from time series should be implemented using the
            WindowedTimeSeries methods as in run_ts_to_stft (including truncation to clock-zero) etc.

    Parameters
    ----------
    dec_level_config
    tfk_dataset

    Returns
    -------

    """
    try:
        features = read_features_from_mth5()
        return features
    except Exception as e:
        msg = f"Features could not be accessed from MTH5 -- {e}\n"
        msg += "Calculating features on the fly (development only)"
        logger.warning(msg)

    for (
        chws
    ) in dec_level_config.channel_weight_specs:  # This refers to solving a TF equation
        # Loop over features and compute them
        msg = f"channel weight spec:\n {chws}"
        logger.info(msg)
        for fws in chws.feature_weight_specs:
            msg = f"feature weight spec: {fws}"
            logger.info(msg)
            feature = fws.feature
            msg = f"feature: {feature}"
            logger.info(msg)
            feature_chunks = []
            if feature.name == "coherence":
                msg = f"{feature.name} is not supported as a data weighting feature"
                logger.warning(msg)
            elif feature.name == "multiple_coherence":
                msg = f"{feature.name} is not supported as a data weighting feature"
                logger.warning(msg)
                # raise NotImplementedError(msg)
            elif feature.name == "striding_window_coherence":
                # TODO: review logic in time_series_helpers.run_ts_to_stft.  This logic could be
                #  modified to work similarly.  Importantly, we need to map the time axis of stft
                #  onto the features
                feature.validate_station_ids(
                    local_station_id=tfk_dataset.local_station_id,
                    remote_station_id=tfk_dataset.remote_station_id,
                )

                # Make main window scheme agree with the one used for the FC calculation
                feature.window = dec_level_config.stft.window
                feature.set_subwindow_from_window(fraction=0.25)

                # get data required for computing the feature.
                # Loop the runs (or run-pairs) ... this should be equivalent to grouping on start time.
                # TODO: consider mixing in valid run info from processing_summary here, (avoid window too long for data)
                #  Desirable to have some "processing_run" iterator supplied by KernelDataset.
                from aurora.pipelines.time_series_helpers import (
                    truncate_to_clock_zero,
                )  # TODO: consider storing clock-zero-truncated data

                tmp = tfk_dataset.df.copy(deep=True)
                group_by = [
                    "start",
                ]
                grouper = tmp.groupby(
                    group_by
                )  # these should have 1 or two rows per group
                for start, df in grouper:
                    end = df.end.unique()[0]  # nice to have this for info log
                    logger.debug("Access ch1 and ch2 ")
                    ch1_row = df[df.station == feature.station1].iloc[0]
                    ch1_data = ch1_row.run_dataarray.to_dataset("channel")[feature.ch1]
                    ch1_data = truncate_to_clock_zero(
                        decimation_obj=dec_level_config, run_xrds=ch1_data
                    )
                    ch2_row = df[df.station == feature.station2].iloc[0]
                    ch2_data = ch2_row.run_dataarray.to_dataset("channel")[feature.ch2]
                    ch2_data = truncate_to_clock_zero(
                        decimation_obj=dec_level_config, run_xrds=ch2_data
                    )
                    msg = f"Data for computing {feature.name} on {start} -- {end} ready"
                    logger.info(msg)
                    # Compute the feature.
                    freqs, coherence_spectrogram = feature.compute(ch1_data, ch2_data)
                    # TODO: consider making get_time_axis() a method of the feature class
                    # Get Time Axis (See Development Note #4)
                    # now create a WindowedTimeSeries and get the time axis")
                    from aurora.time_series.windowing_scheme import (
                        window_scheme_from_decimation,
                    )

                    windowing_scheme = window_scheme_from_decimation(
                        decimation=dec_level_config
                    )
                    ch1_dataset = ch1_data.to_dataset()
                    windowed_obj = windowing_scheme.apply_sliding_window(
                        ch1_dataset, dt=1.0 / dec_level_config.decimation.sample_rate
                    )
                    # Now, take the time axis off the windowed obj, and bind it to coherence_spectrogram
                    # Check the logic in time_series_helpers.fft_xr_ds for how to do this, and note there is a model
                    # there for multivariate.
                    coherence_spectrogram_xr = xr.DataArray(
                        coherence_spectrogram,
                        dims=["time", "frequency"],
                        coords={"frequency": freqs, "time": windowed_obj.time.data},
                    )
                    feature_chunks.append(coherence_spectrogram_xr)
                feature_data = xr.concat(feature_chunks, "time")
                feature.data = feature_data  # bind feature data to feature instance (maybe temporal workaround)

    return


def read_features_from_mth5():
    raise NotImplementedError


def calculate_weights(
    dec_level_config: AuroraDecimationLevel, tfk_dataset: KernelDataset
):
    """
        Placeholder.

        Development Note #5.
            The calculation of weights from features is in general not a simple operation.
            Naive calculation by multiplying data with weight functions may be relatively simple, but even this is
            complicated by the selection of weight functions, and transition regions.  Moreover, when using hard
            thresholds (weights that can go to zero) it is possible to unintentionally "zero-out" the whole dataset.

            Such problems can be partly addressed by implementing strategies such as; if all weights are zero, keep
            the "best 20%" of the data, say.  However, when multiple features are being used, we then need to handle
            ranking of data that may fail to qualify based on one feature, but OK based on another. General handling
            for this may involve case studies of features and their relationship to data quality, and should be done
            with visualization capability.

            This analysis can be further complicated by the fact that some features are dynamic.  Consider
            for example the Mahalnobis Distance (MD).  This feature gives a reasonable way to down-weight data based on
            the empirical distributions, but, say that another feature is also employed (multiple coherence (MC) for
            example). The MD values of the data will change depending on whether the MC was applied before or
            after the MC trimming of the dataset.  In the MD case, a reasonable workaround would be to add an MD option
            in the regression, rather than in the preliminary weighting, since the robust regression repeatedly
            recomputes weights based on updated distributions.  Thus features can be divided into 'static' and
            'dynamic'.  Static features can be computed before regression, and used to pre-weight the data, whereas
            dynamic features are better worked into the regression loops.

            There may not be an ideal general solution, but it is likely that there will be a few "plays" that we
            can devise that will work to improve the data most of the time.

            For the reasons outlined above, the weight calculation will be in development for some time, and will
            likely require feature-visualization tools before we settle on a fixed set of strategies.  In the meantime,
            we can support

    Parameters
    ----------
    features
    chws

    Returns
    -------

    """

    # loop the channel weight specs
    for chws in dec_level_config.channel_weight_specs:

        msg = f"{chws}"
        logger.info(msg)
        # TODO: Consider calculating all the weight kernels in advance, case switching on the combination style.
        if chws.combination_style == "multiplication":
            print(f"chws.combination_style {chws.combination_style}")

            weights = None
            # loop the feature weight specs
            for fws in chws.feature_weight_specs:
                msg = f"feature weight spec: {fws}"
                logger.info(msg)
                feature = fws.feature
                msg = f"feature: {feature}"
                logger.info(msg)
                # TODO: confirm that the feature object has its data
                print("feature.data", feature.data, len(feature.data))

                # TODO: Now apply the fws weighting to the feature data
                #  Hopefully this is independent of the feature.
                weights = None
                for wk in fws.weight_kernels:
                    if weights is None:
                        weights = wk.evaluate(feature.data)
                    else:
                        weights *= wk.evaluate(feature.data)
                # chws.weights[fws.feature.name] = weights
            chws.weights = weights

        else:
            msg = f"chws.combination_style {chws.combination_style} not implemented"
            raise ValueError(msg)

    return
