"""
    This module contains helper functions for working with obspy objects.

    Development Notes
    - TODO: Most of this could likely be moved into MTH5

"""
import datetime

from obspy import UTCDateTime
from loguru import logger


def trim_streams_to_common_timestamps(streams):
    """
    Slices streams to common time stamps.

    Parameters
    ----------
    streams

    Returns
    -------

    """
    start_times = sorted(list(set([tr.stats.starttime.isoformat() for tr in streams])))
    end_times = sorted(list(set([tr.stats.endtime.isoformat() for tr in streams])))
    run_stream = streams.slice(UTCDateTime(start_times[0]), UTCDateTime(end_times[-1]))
    return run_stream


def align_streams(streams, clock_start):
    """
    Shift stream timestamps so that they are aligned.

    This is a hack around to handle data that are asynchronously sampled.
    It should not be used in general.  It is only appropriate for datasets that have
    been tested with it.
    PKD, SAO only at this point.

    Parameters
    ----------
    streams : iterable of types obspy.core.stream.Stream
    clock_start : obspy UTCDateTime
        this is a reference time that we set the first sample to be

    Returns
    -------
    streams
    """
    for stream in streams:
        logger.info(
            f"{stream.stats['station']}  {stream.stats['channel']} N="
            f"{len(stream.data)}  startime {stream.stats.starttime}"
        )
        dt_seconds = stream.stats.starttime - clock_start
        logger.info(f"stream offset seconds {dt_seconds}")
        dt = datetime.timedelta(seconds=dt_seconds)
        logger.info(f"stream timedelta {dt}")
        stream.stats.starttime = stream.stats.starttime - dt
    return streams


FDSN_CHANNEL_MAP = {}
FDSN_CHANNEL_MAP["BQ2"] = "BQ1"
FDSN_CHANNEL_MAP["BQ3"] = "BQ2"
FDSN_CHANNEL_MAP["BT1"] = "BF1"
FDSN_CHANNEL_MAP["BT2"] = "BF2"
FDSN_CHANNEL_MAP["BT3"] = "BF3"
FDSN_CHANNEL_MAP["LQ2"] = "LQ1"
FDSN_CHANNEL_MAP["LQ3"] = "LQ2"
FDSN_CHANNEL_MAP["LT1"] = "LF1"
FDSN_CHANNEL_MAP["LT2"] = "LF2"
FDSN_CHANNEL_MAP["LT3"] = "LF3"
FDSN_CHANNEL_MAP["LFE"] = "LF1"
FDSN_CHANNEL_MAP["LFN"] = "LF2"
FDSN_CHANNEL_MAP["LFZ"] = "LF3"
FDSN_CHANNEL_MAP["LQE"] = "LQ1"
FDSN_CHANNEL_MAP["LQN"] = "LQ2"


def make_channel_labels_fdsn_compliant(streams):
    """
    Renames channels to FDSN compliant channel names.

    Workaround because NCEDC channel nomenclature is not FDSN Compliant for PKD, SAO
    Specifically, re-assign non-conventional channel labels
    Q2 --> Q1
    Q3 --> Q2
    T1 --> F1
    T2 --> F2
    T3 --> F3

    Parameters
    ----------
    streams : iterable of types obspy.core.stream.Stream

    Returns
    -------
    streams : iterable of types obspy.core.stream.Stream
        Same as input but updated with new channel names.
    """
    for stream in streams:
        stream.stats["channel"] = FDSN_CHANNEL_MAP[stream.stats["channel"]]
    return streams
