"""
    This module contains methods to help make mth5 files.
    TODO: Review and move to make_mth5 if relevant
"""
import pandas as pd
from loguru import logger


def channel_summary_to_make_mth5(
    df: pd.DataFrame, network="", channel_nomenclature=None, verbose=False
):
    """
    Makes a dataframe that can be passed to make_mth5.

    Context is say you have a station_xml that has come from somewhere and you want
    to make an mth5 from it, with all the relevant data.  Then you should use
    make_mth5.  But make_mth5 wants a df with a particular schema (which should be
    written down somewhere!) This method returns a dataframe with the schema that
    MakeMTH5() expects.  Specifically, there is one row for each channel-run
    combination.

    This method could be an option for output format of mth5.channel_summary()

    TODO: Add support for channel_nomenclature

    This method could be an option for output format of mth5.channel_summary()

    Parameters
    ----------
    df: pd.DataFrame
        Output from mth5_obj.channel_summary
    network: str
        Usually two characters, the network code specifies the network on which
        the data were acquired.
    verbose: bool
        Set to true to see some strings describing what is happening


    Returns
    -------
    out_df: pd.DataFrame
        The make_mth5 formatted dataframe


    """
    if not network:
        logger.warning("Network not specified")
        logger.warning("this will cause IRIS data pull to fail")
        raise ValueError

    ch_map = {"ex": "LQN", "ey": "LQE", "hx": "LFN", "hy": "LFE", "hz": "LFZ"}
    if channel_nomenclature is not None:
        logger.info("Need to add handling for channel nomenclature")
        # either replace the keys in ch_map, or create a new ch_map

    num_rows = len(df)
    networks = num_rows * [network]
    stations = num_rows * [None]
    locations = num_rows * [""]
    channels = num_rows * [None]
    starts = num_rows * [None]
    ends = num_rows * [None]

    i = 0
    for group_id, group_df in df.groupby(["station", "run"]):
        if verbose:
            logger.info(
                f"{group_id}, from "
                f"{group_df.start.unique()[0]}, to "
                f"{group_df.end.unique()[0]}"
            )
        for index, row in group_df.iterrows():
            stations[i] = row.station
            channels[i] = ch_map[row.component]
            starts[i] = row.start
            ends[i] = row.end
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
