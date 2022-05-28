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