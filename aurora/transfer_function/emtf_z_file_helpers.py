"""
These methods can possibly be moved under mt_metadata, or mth5

They extract info needed to setup emtf_z files.
"""
import fortranformat as ff

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
    for i_ch, channel_id in enumerate(EMTF_CHANNEL_ORDER):
        try:
            channel = run_obj.get_channel(channel_id)
            azimuth = channel.metadata.measurement_azimuth
            tilt = channel.metadata.measurement_tilt
            station_id = run_obj.station_group.name
            emtf_channel_id = channel_id.capitalize()
            # format(i5, 1x, f8.2, 1x, f8.2, 1x, a3, 2x, a6) #Fortran Format
            ff_format = ff.FortranRecordWriter(
                "(i5, 1x, f8.2, 1x, f8.2, 1x, " "a3, 1x, a3)"
            )
            fortran_str = ff_format.write(
                [i_ch + 1, azimuth, tilt, station_id, emtf_channel_id]
            )
            out_str = f"{fortran_str}\n"
            # out_str = f"{i_ch + 1}     {azimuth}     {tilt} " \
            #     f"{station_id[0:3]} {emtf_channel_id}\n"
            output_strings.append(out_str)
        except:
            print(f"No channel {channel_id} in run")
            pass
    return output_strings


def merge_tf_collection_to_match_z_file(aux_data, tf_collection):
    """
    Currently this is only used for the synthtetic test, but maybe useful for
    other tests.  Given data from a z_file, and a tf_collection,
    the tf_collection may have several TF estimates at the same frequency
    from multiple decimation levels.  This tries to make a single array as a
    function of period for all rho and phi
    Parameters
    ----------
    aux_data
    tf_collection

    Returns
    -------

    """
    import numpy as np

    rxy = np.full(len(aux_data.decimation_levels), np.nan)
    ryx = np.full(len(aux_data.decimation_levels), np.nan)
    pxy = np.full(len(aux_data.decimation_levels), np.nan)
    pyx = np.full(len(aux_data.decimation_levels), np.nan)
    dec_levels = list(set(aux_data.decimation_levels))
    dec_levels = [int(x) for x in dec_levels]
    dec_levels.sort()
    for dec_level in dec_levels:
        aurora_tf = tf_collection.tf_dict[dec_level - 1]
        indices = np.where(aux_data.decimation_levels == dec_level)[0]
        for ndx in indices:
            period = aux_data.periods[ndx]
            # find the nearest period in aurora_tf
            aurora_ndx = np.argmin(np.abs(aurora_tf.periods - period))
            rxy[ndx] = aurora_tf.rho[aurora_ndx, 0]
            ryx[ndx] = aurora_tf.rho[aurora_ndx, 1]
            pxy[ndx] = aurora_tf.phi[aurora_ndx, 0]
            pyx[ndx] = aurora_tf.phi[aurora_ndx, 1]

    return rxy, ryx, pxy, pyx
