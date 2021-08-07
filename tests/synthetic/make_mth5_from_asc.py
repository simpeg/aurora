# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 16:03:21 2021

@author: jpeacock

Want to create a station1.h5, station2.h5 and array.h5
where array has 2 stations.
"""

import numpy as np
from pathlib import Path
from random import seed
import pandas as pd
from mth5.timeseries import ChannelTS, RunTS
from mth5.mth5 import MTH5

from synthetic_station_config import ACTIVE_FILTERS

seed(0)


def create_run_ts_from_station_config(config, df):
    """
    Loop over stations and make them ChannelTS objects.
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
    ch_list = []
    for col in df.columns:
        data = df[col].values

        if col in ["ex", "ey"]:
            meta_dict = {"component": col,
                         "sample_rate": config["sample_rate"],
                         "filter.name": config["filters"][col],
                         }
            chts = ChannelTS(channel_type="electric", data=data,
                             channel_metadata=meta_dict)
            # add metadata to the channel here
            chts.channel_metadata.dipole_length = 50
            if col == "ey":
                chts.channel_metadata.measurement_azimuth = 90.0


        elif col in ["hx", "hy", "hz"]:
            meta_dict = {"component": col,
                         "sample_rate": config["sample_rate"],
                         "filter.name": config["filters"][col],
                         }
            chts = ChannelTS(channel_type="magnetic", data=data,
                             channel_metadata=meta_dict)
            if col == "hy":
                chts.channel_metadata.measurement_azimuth = 90.0

        ch_list.append(chts)

    # make a RunTS object
    runts = RunTS(array_list=ch_list)

    # add in metadata
    runts.station_metadata.id = config["station_id"]
    runts.run_metadata.id = config["run_id"]
    return runts

def create_mth5_synthetic_file(station_cfg, plot=False, add_nan_values=False):
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
    #read in data
    df = pd.read_csv(station_cfg["raw_data_path"],
                     names=station_cfg["columns"], sep="\s+")
    #add noise
    for col in station_cfg["columns"]:
        if station_cfg["noise_scalar"][col]:
            df[col] += station_cfg["noise_scalar"][col]*np.random.randn(len(df))

    if add_nan_values:
        new_path = Path(station_cfg["mth5_path"].__str__().replace(".h5", "_nan.h5"))
        station_cfg["mth5_path"] = new_path
        for col in station_cfg["columns"]:
            for [ndx,num_nan] in station_cfg["nan_indices"][col]:
                df[col].loc[ndx:ndx+num_nan] = np.nan
    #cast to run_ts
    runts = create_run_ts_from_station_config(station_cfg, df)

    # plot the data
    if plot:
        runts.plot()


    #survey = Survey()

    # make an MTH5
    m = MTH5()
    m.open_mth5(station_cfg["mth5_path"], mode="w")
    station_group = m.add_station(station_cfg["station_id"])

    #<try assign location>
    from mt_metadata.timeseries.location import Location
    location = Location()
    location.latitude = station_cfg["latitude"]
    station_group.metadata.location = location
    print("DEBUG: setting latitude in the above line does not wind up being "
          "in the run, but it is in the station_group")
    run_group = station_group.add_run(station_cfg["run_id"])
    run_group.station_group.metadata.location = location
    print("DEBUG: setting latitude in the above line does not wind up being "
          "in the run either")
    # </try assign location>
    run_group.from_runts(runts)

    #add filters
    for fltr in ACTIVE_FILTERS:
        cf_group = m.filters_group.add_filter(fltr)

    m.close_mth5()
    return

def create_mth5_synthetic_file_for_array(station_cfgs,
                                         h5_name=Path("data","test12rr.h5"),
                                         plot=False):
    # open an MTH5
    m = MTH5()
    m.open_mth5(h5_name, mode="w")


    run_ts_dict = {}
    for station_cfg in station_cfgs:
        #read in data
        df = pd.read_csv(station_cfg["raw_data_path"],
                         names=station_cfg["columns"], sep="\s+")
        #add noise
        for col in station_cfg["columns"]:
            df[col] += station_cfg["noise_scalar"][col]*np.random.randn(len(df))
        #cast to run_ts
        runts = create_run_ts_from_station_config(station_cfg, df)

        # plot the data
        if plot:
            runts.plot()
        station_run = f"{station_cfg['station_id']}_{station_cfg['run_id']}"
        #run_ts_dict[station_run] = runts

        station_group = m.add_station(station_cfg["station_id"])
        run_group = station_group.add_run(station_cfg["run_id"])
        run_group.from_runts(runts)

    #add filters
    for fltr in ACTIVE_FILTERS:
        cf_group = m.filters_group.add_filter(fltr)
    m.close_mth5()


def main():
    from synthetic_station_config import STATION_01_CFG
    from synthetic_station_config import STATION_02_CFG
    create_mth5_synthetic_file(STATION_01_CFG, plot=False, add_nan_values=True)
    create_mth5_synthetic_file(STATION_01_CFG, plot=False)
    create_mth5_synthetic_file(STATION_02_CFG)
    create_mth5_synthetic_file_for_array([STATION_01_CFG, STATION_02_CFG])

if __name__ == '__main__':
    main()

