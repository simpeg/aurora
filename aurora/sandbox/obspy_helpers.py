import datetime

from obspy import UTCDateTime


def trim_streams_to_acquisition_run(streams):
    """
    Rename as? TRIM DATA STREAMS TO COMMON TIME STAMPS
    TODO: add doc here.  It looks like we are slicing from the earliest starttime to
    the latest endtime
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

    """
    for stream in streams:
        print(
            f"{stream.stats['station']}  {stream.stats['channel']} N="
            f"{len(stream.data)}  startime {stream.stats.starttime}"
        )
        dt_seconds = stream.stats.starttime - clock_start
        print(f"dt_seconds {dt_seconds}")
        dt = datetime.timedelta(seconds=dt_seconds)
        print(f"dt = {dt}")
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
    Workaround because NCEDC channel nomenclature is not FDSN Compliant for PKD, SAO
    Parameters
    ----------
    streams : iterable of types obspy.core.stream.Stream

    Returns
    -------

    """
    # <REASSIGN NON-CONVENTIONAL CHANNEL LABELS (Q2, Q3, T1, T2)>
    for stream in streams:
        stream.stats["channel"] = FDSN_CHANNEL_MAP[stream.stats["channel"]]
    # </REASSIGN NON-CONVENTIONAL CHANNEL LABELS (Q2, Q3, T1, T2)>
    return streams
