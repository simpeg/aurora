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

- 20231103: Added an 8Hz up-sampled version of test1.  No spectral content was added
 so the band between the old and new Nyquist frequencies is bogus.

Notes: Work in progress -- this module is being migrated to MTH5.

"""
# import inspect
import numpy as np
import pandas as pd
import pathlib
import scipy.signal as ssig

from aurora.test_utils.synthetic.paths import SyntheticTestPaths
from aurora.test_utils.synthetic.station_config import make_filters
from aurora.test_utils.synthetic.station_config import make_station_01
from aurora.test_utils.synthetic.station_config import make_station_02
from aurora.test_utils.synthetic.station_config import make_station_03
from aurora.test_utils.synthetic.station_config import make_station_04
from loguru import logger
from mth5.mth5 import MTH5
from mth5.timeseries import ChannelTS, RunTS
from mt_metadata.transfer_functions.processing.aurora import ChannelNomenclature
from mt_metadata.timeseries import Electric
from mt_metadata.timeseries import Magnetic
from mth5.utils.helpers import add_filters

np.random.seed(0)

synthetic_test_paths = SyntheticTestPaths()
MTH5_PATH = synthetic_test_paths.mth5_path


def create_run_ts_from_synthetic_run(run, df, channel_nomenclature="default") -> RunTS:
    """
    Loop over channels of synthetic data in df and make ChannelTS objects.

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
    runts: RunTS
        MTH5 run time series object, data and metadata bound into one.
    """

    channel_nomenclature_obj = ChannelNomenclature()
    channel_nomenclature_obj.keyword = channel_nomenclature
    ch_list = []
    for col in df.columns:

        data = df[col].values
        if col in channel_nomenclature_obj.ex_ey:
            channel_metadata = Electric()
            channel_metadata.component = col
            channel_metadata.units = "millivolts per kilometer"
            chts = ChannelTS(
                channel_type="electric",
                data=data,
                channel_metadata=channel_metadata.to_dict(),
            )

            # add metadata to the channel here
            chts.channel_metadata.dipole_length = 50
            if col == channel_nomenclature_obj.ey:
                chts.channel_metadata.measurement_azimuth = 90.0

        elif col in channel_nomenclature_obj.hx_hy_hz:
            channel_metadata = Magnetic()
            channel_metadata.units = "nanotesla"
            channel_metadata.component = col
            channel_metadata.channel_number = 0
            channel_metadata.sample_rate = 1.0
            chts = ChannelTS(
                channel_type=channel_metadata.type,
                data=data,
                channel_metadata=channel_metadata.to_dict(),
            )
            chts.component = col

            if col == channel_nomenclature_obj.ey:
                chts.channel_metadata.measurement_azimuth = 90.0

        chts.channel_metadata.component = col
        chts.channel_metadata.sample_rate = run.run_metadata.sample_rate
        chts.channel_metadata.filter.name = run.filters[col]
        chts.channel_metadata.filter.applied = len(run.filters[col]) * [
            True,
        ]
        chts.channel_metadata.start = run.run_metadata.time_period.start

        ch_list.append(chts)

    # make a RunTS object
    runts = RunTS(array_list=ch_list)

    # add in metadata
    runts.run_metadata.id = run.run_metadata.id
    return runts


def get_time_series_dataframe(run, source_folder, add_nan_values):
    """
    Returns time series data in a dataframe with columns named for EM field component.

    Parameters
    ----------
    run: aurora.test_utils.synthetic.station_config.SyntheticRun
        Information needed to define/create the run
    source_folder: pathlib.Path, or null
        Where to load the ascii time series from
    add_nan_values: bool
        If True, add some NaN, if False, do not add Nan.

    Up-samples data to run.sample_rate, which is treated as in integer.
    Only tested for 8, to make 8Hz data for testing.  If run.sample_rate is default (1.0)
    then no up-sampling takes place.

    Returns
    -------
    df: pandas.DataFrame
        The time series data for the synthetic run
    """
    # point to the ascii time series
    if source_folder:
        run.raw_data_path = source_folder.joinpath(run.raw_data_path.name)

    # read in data
    df = pd.read_csv(run.raw_data_path, names=run.channels, sep="\s+")

    # upsample data if requested,
    if run.run_metadata.sample_rate != 1.0:
        df_orig = df.copy(deep=True)
        new_data_dict = {}
        for i_ch, ch in enumerate(run.channels):
            data = df_orig[ch].to_numpy()
            new_data_dict[ch] = ssig.resample(
                data, int(run.run_metadata.sample_rate) * len(df_orig)
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
    return df


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
):
    """
    Creates an MTH5 from synthetic data

    Parameters
    ----------
    station_cfgs: list of dicts
        The dicts are one-off data structure used to hold information mth5 needs to
        initialize, specifically sample_rate, filters, etc.
    mth5_name: string or pathlib.Path()
        Where the mth5 will be stored.  This is generated by the station_config,
        but may change in this method based on add_nan_values or channel_nomenclature
    target_folder: str or path, optional
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

    Returns
    -------
    mth5_path: pathlib.Path
        The path to the stored h5 file.
    """

    def update_mth5_path(mth5_path, add_nan_values, channel_nomenclature):
        """set name for output h5 file"""
        path_str = mth5_path.__str__()
        if add_nan_values:
            path_str = path_str.replace(".h5", "_nan.h5")
        if channel_nomenclature != "default":
            path_str = path_str.replace(".h5", f"_{channel_nomenclature}.h5")
        return pathlib.Path(path_str)

    if not target_folder:
        msg = f"No target folder provided for making {mth5_name}"
        logger.warning("No target folder provided for making {}")
        msg = f"Setting target folder to {MTH5_PATH}"
        logger.info(msg)
        target_folder = MTH5_PATH

    try:
        target_folder.mkdir(exist_ok=True, parents=True)
    except OSError:
        msg = "Aurora maybe installed on a read-only file system"
        msg = f"{msg}: try setting target_path argument when calling create_mth5_synthetic_file"
        logger.error(msg)

    mth5_path = target_folder.joinpath(mth5_name)
    mth5_path = update_mth5_path(mth5_path, add_nan_values, channel_nomenclature)

    if not force_make_mth5:
        if mth5_path.exists():
            return mth5_path

    # open output h5
    m = MTH5(file_version=file_version)
    m.open_mth5(mth5_path, mode="w")
    m, survey_id = _get_set_survey_id(m)

    for station_cfg in station_cfgs:
        station_group = m.add_station(station_cfg.id, survey=survey_id)

        for run in station_cfg.runs:
            df = get_time_series_dataframe(run, source_folder, add_nan_values)

            # cast to run_ts
            runts = create_run_ts_from_synthetic_run(
                run, df, channel_nomenclature=channel_nomenclature
            )
            runts.station_metadata.id = station_cfg.id

            # plot the data
            if plot:
                runts.plot()

            run_group = station_group.add_run(run.run_metadata.id)
            run_group.from_runts(runts)

    # add filters
    active_filters = make_filters(as_list=True)
    add_filters(m, active_filters, survey_id)
    m.close_mth5()
    return mth5_path


def create_test1_h5(
    file_version="0.1.0",
    channel_nomenclature="default",
    target_folder=MTH5_PATH,
    source_folder="",
    force_make_mth5=True,
):
    """
    Creates an MTH5 file for a single station named "test1".

    Parameters
    ----------
    file_version
    channel_nomenclature
    target_folder
    source_folder
    force_make_mth5

    Returns
    -------

    """
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
    """
    Creates an MTH5 file for a single station named "test2".
    """
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
    """
    Creates an MTH5 file for a single station named "test1" with some nan values.
    """
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
    """
    Creates an MTH5 file with data from two stations station named "test1" and "test2".
    """
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
    """
    Creates an MTH5 file for a single station named "test3".
    This example has several runs and can be used to test looping over runs.
    """
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
    """
    Creates an MTH5 file for a single station named "test1", data are up-sampled to 8Hz from
    original 1 Hz.

    Note: Because the 8Hz data are derived from the 1Hz, only frequencies below 0.5Hz
    will have valid TFs that yield the apparent resistivity of the synthetic data (100 Ohm-m).
    """
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
    )
    return mth5_path


def _get_set_survey_id(m: MTH5) -> tuple:
    """
    Given an open mth5 file (m) set the survey ID and return it as a string,
    as well a the (modified) mth5 object.

    Parameters
    ----------
    m: MTH5
        The mth5 object to set the survey ID for.

    Returns
    -------
    (m, survey_id): tuple
        m is the (modified) MTH5
        survey_id is a string
    """
    if m.file_version == "0.1.0":
        survey_id = None
    elif m.file_version == "0.2.0":
        survey_id = "EMTF Synthetic"
        m.add_survey(survey_id)
    else:
        msg = f"unexpected MTH5 file_version = {m.file_version}"
        raise NotImplementedError(msg)
    return m, survey_id


def main(file_version="0.1.0"):
    """Allow the module to be called from the command line"""
    file_version = "0.2.0"
    #    create_test1_h5(file_version=file_version)
    #     create_test1_h5_with_nan(file_version=file_version)
    #     create_test2_h5(file_version=file_version)
    #     create_test12rr_h5(file_version=file_version, channel_nomenclature="lemi12")
    #     create_test3_h5(file_version=file_version)
    create_test4_h5(file_version=file_version)


if __name__ == "__main__":
    main()
