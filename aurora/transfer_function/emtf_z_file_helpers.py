"""
These methods can possibly be moved under mt_metadata, or mth5

They extract info needed to setup emtf_z files.
"""
EMTF_CHANNEL_ORDER = ["hx", "hy", "hz", "ex", "ey"]
def make_orientation_block_of_z_file(run_obj):
    """
    Replicates emtz z-file metadata about orientation like this:
    1     0.00     0.00 tes  Hx
    2    90.00     0.00 tes  Hy
    3     0.00     0.00 tes  Hz
    4     0.00     0.00 tes  Ex
    5    90.00     0.00 tes  Ey

    based on this fortran snippet:
            write(3, 115) k, orient(1, k), orient(2, k), stname(1: 3), chid(k)
    Parameters
    ----------
    run_obj

    Returns
    -------

    """
    output_strings = []
    for channel_id in EMTF_CHANNEL_ORDER:
        try:
            channel = run_obj.get_channel(channel_id)
            azimuth = channel.metadata.measurement_azimuth
            tilt = channel.metadata.measurement_tilt
            station_id = run_obj.station_group.name
            emtf_channel_id = channel_id.capitalize()
            out_str = f"{channel_id}     {azimuth}     {tilt} " \
                f"{station_id[0:3]} {emtf_channel_id}\n"
            output_strings.append(out_str)
        except:
            print(f"No channel {channel_id} in run")
            pass
    return output_strings
