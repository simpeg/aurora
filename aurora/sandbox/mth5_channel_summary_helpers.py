import pandas as pd


def channel_summary_to_make_mth5(df, network="ZU"):
    """
    Context is say you have a station_xml that has come from somewhere and you want
    to make an mth5 from it, with all the relevant data.  Then you should use
    make_mth5.  But make_mth5 wants a df with a particular schema (which should be
    written down somewhere!)

    This returns a dataframe with the schema that MakeMTH5() expects.

    TODO: This method could be an option for output format of mth5.channel_summary()

    Parameters
    ----------
    df: the output from mth5_obj.channel_summary

    Returns
    -------

    """
    ch_map = {"ex": "LQN", "ey": "LQE", "hx": "LFN", "hy": "LFE", "hz": "LFZ"}
    number_of_runs = len(df["run"].unique())
    num_rows = 5 * number_of_runs
    networks = num_rows * [network]
    stations = num_rows * [None]
    locations = num_rows * [""]
    channels = num_rows * [None]
    starts = num_rows * [None]
    ends = num_rows * [None]

    i = 0
    for group_id, group_df in df.groupby("run"):
        print(group_id, group_df.start.unique(), group_df.end.unique())
        for index, row in group_df.iterrows():
            stations[i] = row.station
            channels[i] = ch_map[row.component]
            starts[i] = row.start
            ends[i] = row.end
            print("OK")
            i += 1

    out_dict = {
        "network": networks,
        "station": stations,
        "location": locations,
        "channel": channels,
        "start": starts,
        "end": ends,
    }
    out_df = pd.DataFrame(data=out_dict)
    return out_df


def channel_summary_to_dataset_definition(df):
    """

    Parameters
    ----------
    df

    Returns
    -------

    """
    grouper = df.groupby(["station", "run"])
    n_station_runs = len(grouper)
    station_ids = n_station_runs * [None]
    run_ids = n_station_runs * [None]
    start_times = n_station_runs * [None]
    end_times = n_station_runs * [None]
    i = 0
    for (station_id, run_id), group in grouper:
        print(f"{i} {station_id} {run_id}")
        #print(group)
        station_ids[i] = station_id
        run_ids[i] = run_id
        start_times[i] = group.start.iloc[0]
        end_times[i] = group.end.iloc[0]
        i += 1

    data_dict = {}
    data_dict["station_ids"] = station_ids
    data_dict["run_ids"] = run_ids
    data_dict["start_times"] = start_times
    data_dict["end_times"] = end_times
    dataset_definition = pd.DataFrame(data=data_dict)
    print("? Cast as dataset.DatasetDefinition() here?")
    return dataset_definition

