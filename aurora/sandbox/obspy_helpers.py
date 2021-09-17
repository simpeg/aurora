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
