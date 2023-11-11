# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 16:03:21 2021

@author: jpeacock

This module is concerned with creating mth5 files from the synthetic test data
 that originally came from EMTF -- test1.asc and test2.asc.  Each ascii file
 represents five channels of data sampled at 1Hz at a synthetic station.

Mirroring the original ascii files are:
data/test1.h5
data/test2.h5
data/test12rr.h5

Also created are some files with the same data but other channel_nomenclature schemes:
data/test12rr_LEMI34.h5
data/test1_LEMI12.h5

- 20231103: Added an 8Hz upsampled version of test1.  No spectral content was added
so the band between the old and new Nyquist frequencies is bogus.

"""

import numpy as np
import pandas as pd
import pathlib
import scipy.signal as ssig

from loguru import logger

from aurora.test_utils.synthetic.paths import SyntheticTestPaths
from aurora.test_utils.synthetic.station_config import make_filters
from aurora.test_utils.synthetic.station_config import make_station_01
from aurora.test_utils.synthetic.station_config import make_station_02
from aurora.test_utils.synthetic.station_config import make_station_03
from aurora.test_utils.synthetic.station_config import make_station_04
from mth5.timeseries import ChannelTS, RunTS
from mth5.mth5 import MTH5
from mt_metadata.transfer_functions.processing.aurora import ChannelNomenclature

np.random.seed(0)

synthetic_test_paths = SyntheticTestPaths()
MTH5_PATH = synthetic_test_paths.mth5_path


def create_run_ts_from_synthetic_run(run, df, channel_nomenclature="default"):
    """
    Loop over stations and make ChannelTS objects.
    Need to add a tag in the channels
    so that when you call a run it will get all the filters with it.

    Parameters
    ----------
    run: aurora.test_utils.synthetic.station_config.SyntheticRun
        One-off data structure with information mth5 needs to initialize
        Specifically sample_rate, filters,
    df : pandas.DataFrame
        time series data in columns labelled from ["ex", "ey", "hx", "hy", "hz"]
    channel_nomenclature : string
        Keyword corresponding to channel nomenclature mapping in CHANNEL_MAPS variable
        from channel_nomenclature.py module in mt_metadata.
        Supported values include ['default', 'lemi12', 'lemi34', 'phoenix123']

    Returns
    -------

    """
    channel_nomenclature_obj = ChannelNomenclature()
    channel_nomenclature_obj.keyword = channel_nomenclature
    EX, EY, HX, HY, HZ = channel_nomenclature_obj.unpack()
    ch_list = []
    for col in df.columns:
        data = df[col].values
        meta_dict = {
            "component": col,
            "sample_rate": run.sample_rate,
            "filter.name": run.filters[col],
            "time_period.start": run.start,
        }
        if col in [EX, EY]:

            chts = ChannelTS(
                channel_type="electric", data=data, channel_metadata=meta_dict
            )
            # add metadata to the channel here
            chts.channel_metadata.dipole_length = 50
            if col == EY:
                chts.channel_metadata.measurement_azimuth = 90.0

        elif col in [HX, HY, HZ]:
            chts = ChannelTS(
                channel_type="magnetic", data=data, channel_metadata=meta_dict
            )
            if col == HY:
                chts.channel_metadata.measurement_azimuth = 90.0

        ch_list.append(chts)

    # make a RunTS object
    runts = RunTS(array_list=ch_list)

    # add in metadata
    runts.run_metadata.id = run.id
    return runts


def add_filters(active_filters, m, survey_id):
    """

    Parameters
    ----------
    active_filters: list of filters
    m: mth5.mth5.MTH5
    survey_id: string

    Returns
    -------

    """
    for fltr in active_filters:
        if m.file_version == "0.1.0":
            m.filters_group.add_filter(fltr)
        elif m.file_version == "0.2.0":
            survey = m.get_survey(survey_id)
            survey.filters_group.add_filter(fltr)
        else:
            msg = f"unexpected MTH5 file_version = {m.file_version}"
            raise NotImplementedError(msg)
    return m


def get_set_survey_id(m):
    if m.file_version == "0.1.0":
        survey_id = None
    elif m.file_version == "0.2.0":
        survey_id = "EMTF Synthetic"
        m.add_survey(survey_id)
    else:
        msg = f"unexpected MTH5 file_version = {m.file_version}"
        raise NotImplementedError(msg)
    return m, survey_id


def create_mth5_synthetic_file(
    station_cfgs,
    mth5_name,
    target_folder="",
    source_folder="",
    plot=False,
    add_nan_values=False,
    file_version="0.1.0",
    channel_nomenclature="default",
    force_make_mth5=True,
    upsample_factor=0,
):
    """

    Parameters
    ----------
    station_cfgs: list of dicts
        The dicts are one-off data structure used to hold information mth5 needs to
        initialize, specifically sample_rate, filters, etc.
    mth5_name: string or pathlib.Path()
        Where the mth5 will be stored.  This is generated by the station_config,
        but may change in this method based on add_nan_values or channel_nomenclature
    plot: bool
        Set to false unless you want to look at a plot of the time series
    add_nan_values: bool
        If true, some np.nan are sprinkled into the time series.  Intended to be used for tests.
    file_version: string
        One of ["0.1.0", "0.2.0"], corresponding to the version of mth5 to create
    channel_nomenclature : string
        Keyword corresponding to channel nomenclature mapping in CHANNEL_MAPS variable
        from channel_nomenclature.py module in mt_metadata.
        Supported values are ['default', 'lemi12', 'lemi34', 'phoenix123']
    force_make_mth5: bool
        If set to true, the file will be made, even if it already exists.
        If false, and file already exists, skip the make job.
    upsample_factor: int
        Integer, only tested for 8, to make 8Hz data for testing.  If upsample_factor is set to
        default (zero), then no upsampling takes place.


    Returns
    -------
    mth5_path: pathlib.Path
        The path to the stored h5 file.
    """
    if not target_folder:
        msg = f"No target folder provided for making {mth5_name}"
        logger.warning("No target folder provided for making {}")
        msg = f"Setting target folder to {MTH5_PATH}"
        logger.info(msg)
        target_folder = MTH5_PATH

    mth5_path = target_folder.joinpath(mth5_name)
    # set name for output h5 file
    if add_nan_values:
        mth5_path = pathlib.Path(mth5_path.__str__().replace(".h5", "_nan.h5"))
    if channel_nomenclature != "default":
        mth5_path = pathlib.Path(
            mth5_path.__str__().replace(".h5", f"_{channel_nomenclature}.h5")
        )
    if not force_make_mth5:
        if mth5_path.exists():
            return mth5_path

    # open output h5
    m = MTH5(file_version=file_version)
    m.open_mth5(mth5_path, mode="w")
    m, survey_id = get_set_survey_id(m)

    for station_cfg in station_cfgs:
        station_group = m.add_station(station_cfg.id, survey=survey_id)
        for run in station_cfg.runs:
            if source_folder:
                run.raw_data_path = source_folder.joinpath(run.raw_data_path.name)

            # read in data
            df = pd.read_csv(run.raw_data_path, names=run.channels, sep="\s+")

            # generate upsampled data if requested, store in df
            if upsample_factor:
                df_orig = df.copy(deep=True)
                new_data_dict = {}
                for i_ch, ch in enumerate(run.channels):
                    data = df_orig[ch].to_numpy()
                    new_data_dict[ch] = ssig.resample(
                        data, upsample_factor * len(df_orig)
                    )
                df = pd.DataFrame(data=new_data_dict)

            # add noise
            for col in run.channels:
                if run.noise_scalars[col]:
                    df[col] += run.noise_scalars[col] * np.random.randn(len(df))

            # add nan
            if add_nan_values:
                for col in run.channels:
                    for [ndx, num_nan] in run.nan_indices[col]:
                        df[col].loc[ndx : ndx + num_nan] = np.nan

            # cast to run_ts
            runts = create_run_ts_from_synthetic_run(
                run, df, channel_nomenclature=channel_nomenclature
            )
            runts.station_metadata.id = station_cfg.id

            # plot the data
            if plot:
                runts.plot()

            run_group = station_group.add_run(run.id)
            run_group.from_runts(runts)

    # add filters
    active_filters = make_filters(as_list=True)
    add_filters(active_filters, m, survey_id)
    m.close_mth5()
    return mth5_path


def create_test1_h5(
    file_version="0.1.0",
    channel_nomenclature="default",
    target_folder=MTH5_PATH,
    source_folder="",
    force_make_mth5=True,
):
    station_01_params = make_station_01(channel_nomenclature=channel_nomenclature)
    mth5_name = station_01_params.mth5_name
    station_params = [
        station_01_params,
    ]
    mth5_path = create_mth5_synthetic_file(
        station_params,
        mth5_name,
        plot=False,
        file_version=file_version,
        channel_nomenclature=channel_nomenclature,
        target_folder=target_folder,
        source_folder=source_folder,
        force_make_mth5=force_make_mth5,
    )
    return mth5_path


def create_test2_h5(
    file_version="0.1.0",
    channel_nomenclature="default",
    force_make_mth5=True,
    target_folder=MTH5_PATH,
    source_folder="",
):
    station_02_params = make_station_02(channel_nomenclature=channel_nomenclature)
    mth5_name = station_02_params.mth5_name
    station_params = [
        station_02_params,
    ]
    mth5_path = create_mth5_synthetic_file(
        station_params,
        mth5_name,
        plot=False,
        file_version=file_version,
        force_make_mth5=force_make_mth5,
        target_folder=target_folder,
        source_folder=source_folder,
    )
    return mth5_path


def create_test1_h5_with_nan(
    file_version="0.1.0",
    channel_nomenclature="default",
    target_folder=MTH5_PATH,
    source_folder="",
):
    station_01_params = make_station_01(channel_nomenclature=channel_nomenclature)
    mth5_name = station_01_params.mth5_name
    station_params = [
        station_01_params,
    ]
    mth5_path = create_mth5_synthetic_file(
        station_params,
        mth5_name,
        plot=False,
        add_nan_values=True,
        file_version=file_version,
        target_folder=target_folder,
        source_folder=source_folder,
    )
    return mth5_path


def create_test12rr_h5(
    file_version="0.1.0",
    channel_nomenclature="default",
    target_folder=MTH5_PATH,
    source_folder=None,
):
    station_01_params = make_station_01(channel_nomenclature=channel_nomenclature)
    station_02_params = make_station_02(channel_nomenclature=channel_nomenclature)
    station_params = [station_01_params, station_02_params]
    mth5_name = "test12rr.h5"
    mth5_path = create_mth5_synthetic_file(
        station_params,
        mth5_name,
        file_version=file_version,
        channel_nomenclature=channel_nomenclature,
        target_folder=target_folder,
        source_folder=source_folder,
    )
    mth5_path = pathlib.Path(mth5_path)
    return mth5_path


def create_test3_h5(
    file_version="0.1.0",
    channel_nomenclature="default",
    force_make_mth5=True,
    target_folder=MTH5_PATH,
    source_folder="",
):

    station_03_params = make_station_03(channel_nomenclature=channel_nomenclature)
    station_params = [
        station_03_params,
    ]
    mth5_path = create_mth5_synthetic_file(
        station_params,
        station_params[0].mth5_name,
        file_version=file_version,
        force_make_mth5=force_make_mth5,
        target_folder=target_folder,
        source_folder=source_folder,
    )
    return mth5_path


def create_test4_h5(
    file_version="0.1.0",
    channel_nomenclature="default",
    target_folder=MTH5_PATH,
    source_folder="",
):
    """8Hz data kluged from the 1Hz ... only freqs below 0.5Hz will make sense (100 Ohmm and 45deg)"""
    station_04_params = make_station_04(channel_nomenclature=channel_nomenclature)
    mth5_path = create_mth5_synthetic_file(
        [
            station_04_params,
        ],
        station_04_params.mth5_name,
        plot=False,
        file_version=file_version,
        channel_nomenclature=channel_nomenclature,
        target_folder=target_folder,
        source_folder=source_folder,
        upsample_factor=8,
    )
    return mth5_path


def main(file_version="0.1.0"):
    file_version = "0.2.0"
    create_test1_h5(file_version=file_version)
    create_test1_h5_with_nan(file_version=file_version)
    create_test2_h5(file_version=file_version)
    create_test12rr_h5(file_version=file_version)
    create_test3_h5(file_version=file_version)
    create_test4_h5(file_version=file_version)


if __name__ == "__main__":
    main()
