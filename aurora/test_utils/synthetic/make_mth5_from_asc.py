# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 16:03:21 2021

@author: jpeacock

This module is concerned with creating mth5 files from the synthetic test data
 that origianlly came from EMTF.

Want to create a station1.h5, station2.h5 and array.h5
where array has 2 stations.
"""

import numpy as np
from pathlib import Path

import pandas as pd
from mth5.timeseries import ChannelTS, RunTS
from mth5.mth5 import MTH5

from aurora.config.metadata.channel_nomenclature import ChannelNomenclature
from aurora.test_utils.synthetic.synthetic_station_config import make_filters
from aurora.test_utils.synthetic.synthetic_station_config import (
    make_station_01,
)
from aurora.test_utils.synthetic.synthetic_station_config import (
    make_station_02,
)
from aurora.test_utils.synthetic.synthetic_station_config import (
    make_station_03,
)

np.random.seed(0)


def create_run_ts_from_synthetic_run(run, df, channel_nomenclature="default"):
    """
    Loop over stations and make ChannelTS objects.
    Need to add a tag in the channels
    so that when you call a run it will get all the filters with it.
    Parameters
    ----------
    config: dict
        one-off data structure used to hold information mth5 needs to initialize
        Specifically sample_rate, filters,
    df : pandas.DataFrame
        time series data in columns labelled from ["ex", "ey", "hx", "hy", "hz"]

    Returns
    -------

    """
    channel_nomenclature_obj = ChannelNomenclature()
    channel_nomenclature_obj.keyword = channel_nomenclature
    EX, EY, HX, HY, HZ = channel_nomenclature_obj.unpack()
    ch_list = []
    for col in df.columns:
        data = df[col].values

        if col in [EX, EY]:
            meta_dict = {
                "component": col,
                "sample_rate": run.sample_rate,
                "filter.name": run.filters[col],
            }
            chts = ChannelTS(
                channel_type="electric", data=data, channel_metadata=meta_dict
            )
            # add metadata to the channel here
            chts.channel_metadata.dipole_length = 50
            if col == EY:
                chts.channel_metadata.measurement_azimuth = 90.0

        elif col in [HX, HY, HZ]:
            meta_dict = {
                "component": col,
                "sample_rate": run.sample_rate,
                "filter.name": run.filters[col],
            }
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


def create_mth5_synthetic_file(
    station_cfgs,
    mth5_path,
    plot=False,
    add_nan_values=False,
    file_version="0.1.0",
    channel_nomenclature="default",
):
    """

    Parameters
    ----------
    station_cfg: dict
        one-off data structure used to hold information mth5 needs to initialize
        Specifically sample_rate, filters,
    plot : bool
        set to false unless you want to look at a plot of the time series

    Returns
    -------

    """

    # set name for output h5 file
    if add_nan_values:
        mth5_path = Path(mth5_path.__str__().replace(".h5", "_nan.h5"))
    if channel_nomenclature != "default":
        mth5_path = Path(
            mth5_path.__str__().replace(".h5", f"_{channel_nomenclature}.h5")
        )

    # open output h5
    m = MTH5(file_version=file_version)
    m.open_mth5(mth5_path, mode="w")
    if file_version == "0.2.0":
        survey_id = "EMTF Synthetic"
        m.add_survey(survey_id)
    else:
        survey_id = None

    for station_cfg in station_cfgs:
        station_group = m.add_station(station_cfg.id, survey=survey_id)
        for run in station_cfg.runs:

            # read in data
            df = pd.read_csv(run.raw_data_path, names=run.channels, sep="\s+")

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
    for fltr in active_filters:
        if file_version == "0.1.0":
            m.filters_group.add_filter(fltr)
        elif file_version == "0.2.0":
            survey = m.get_survey(survey_id)
            survey.filters_group.add_filter(fltr)
        else:
            print(f"unexpected file_version = {file_version}")
            raise NotImplementedError

    m.close_mth5()
    return mth5_path


def create_test1_h5(file_version="0.1.0", channel_nomenclature="default"):
    station_01_params = make_station_01(channel_nomenclature=channel_nomenclature)
    mth5_path = station_01_params.mth5_path
    mth5_path = create_mth5_synthetic_file(
        [
            station_01_params,
        ],
        mth5_path,
        plot=False,
        file_version=file_version,
        channel_nomenclature=channel_nomenclature,
    )
    return mth5_path


def create_test2_h5(file_version="0.1.0", channel_nomenclature="default"):
    station_02_params = make_station_02(channel_nomenclature=channel_nomenclature)
    mth5_path = station_02_params.mth5_path
    mth5_path = create_mth5_synthetic_file(
        [
            station_02_params,
        ],
        mth5_path,
        plot=False,
        file_version=file_version,
    )
    return mth5_path


def create_test1_h5_with_nan(file_version="0.1.0", channel_nomenclature="default"):
    station_01_params = make_station_01(channel_nomenclature=channel_nomenclature)
    mth5_path = station_01_params.mth5_path  # DATA_PATH.joinpath("test1.h5")
    mth5_path = create_mth5_synthetic_file(
        [
            station_01_params,
        ],
        mth5_path,
        plot=False,
        add_nan_values=True,
        file_version=file_version,
    )
    return mth5_path


def create_test12rr_h5(file_version="0.1.0", channel_nomenclature="default"):
    station_01_params = make_station_01(channel_nomenclature=channel_nomenclature)
    station_02_params = make_station_02(channel_nomenclature=channel_nomenclature)
    station_params = [station_01_params, station_02_params]
    mth5_path = station_01_params.mth5_path.__str__().replace("test1.h5", "test12rr.h5")
    mth5_path = create_mth5_synthetic_file(
        station_params,
        mth5_path,
        file_version=file_version,
        channel_nomenclature=channel_nomenclature,
    )
    return mth5_path


def create_test3_h5(file_version="0.1.0", channel_nomenclature="default"):
    station_params = make_station_03(channel_nomenclature=channel_nomenclature)
    mth5_path = create_mth5_synthetic_file(
        [
            station_params,
        ],
        station_params.mth5_path,
        file_version=file_version,
    )
    return mth5_path


def main():
    file_version = "0.1.0"
    # file_version="0.2.0"
    create_test1_h5(file_version=file_version)
    create_test1_h5_with_nan(file_version=file_version)
    create_test2_h5(file_version=file_version)
    create_test12rr_h5(file_version=file_version)
    create_test3_h5(file_version=file_version)


if __name__ == "__main__":
    main()
