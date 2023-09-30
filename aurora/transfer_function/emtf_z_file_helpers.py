"""
These methods can possibly be moved under mt_metadata, or mth5

They extract info needed to setup emtf_z files.
"""
import numpy as np

EMTF_CHANNEL_ORDER = ["hx", "hy", "hz", "ex", "ey"]


def get_default_orientation_block(n_ch=5):
    """
    Helper function used when working with matlab structs which do not have enough
    info to make headers

    Parameters
    ----------
    n_ch: int
        number of channels at the station

    Returns
    -------
    orientation_strs: list
        List of text strings, one per channel
    """
    orientation_strs = []
    orientation_strs.append("    1     0.00     0.00 tes  Hx\n")
    orientation_strs.append("    2    90.00     0.00 tes  Hy\n")
    if n_ch == 5:
        orientation_strs.append("    3     0.00     0.00 tes  Hz\n")
    orientation_strs.append("    4     0.00     0.00 tes  Ex\n")
    orientation_strs.append("    5    90.00     0.00 tes  Ey\n")
    return orientation_strs


def merge_tf_collection_to_match_z_file(aux_data, tf_collection):
    """
    Currently this is only used for the synthtetic test, but maybe useful for
    other tests.  Given data from a z_file, and a tf_collection,
    the tf_collection may have several TF estimates at the same frequency
    from multiple decimation levels.  This tries to make a single array as a
    function of period for all rho and phi

    Parameters
    ----------
    aux_data: merge_tf_collection_to_match_z_file
        Object representing a z-file
    tf_collection: aurora.transfer_function.transfer_function_collection
    .TransferFunctionCollection
        Object representing the transfer function returnd from the aurora processing


    Returns
    -------
    result: dict of dicts
        Keyed by ["rho", "phi"], below each of these is an ["xy", "yx",] entry.  The
        lowest level entries are numpy arrays.
    """
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

    result = {}
    result["rho"] = {}
    result["phi"] = {}
    result["rho"]["xy"] = rxy
    result["phi"]["xy"] = pxy
    result["rho"]["yx"] = ryx
    result["phi"]["yx"] = pyx
    return result


def clip_bands_from_z_file(z_path, n_bands_clip, output_z_path=None, n_sensors=5):
    """
    This function takes a z_file and clips periods off the end of it.
    It can come in handy sometimes -- specifically for manipulating matlab
    results of synthetic data.

    Parameters
    ----------
    z_path: Path or str
        path to the z_file to read in and clip periods from
    n_periods_clip: integer
        how many periods to clip from the end of the zfile
    overwrite: bool
        whether to overwrite the zfile or rename it
    n_sensors

    Returns
    -------

    """
    if not output_z_path:
        output_z_path = z_path

    if n_sensors == 5:
        n_lines_per_period = 13
    elif n_sensors == 4:
        n_lines_per_period = 11
        print("WARNING n_sensors==4 NOT TESTED")

    f = open(z_path, "r")
    lines = f.readlines()
    f.close()

    for i in range(n_bands_clip):
        lines = lines[:-n_lines_per_period]
    n_bands_str = lines[5].split()[-1]
    n_bands = int(n_bands_str)
    new_n_bands = n_bands - n_bands_clip
    new_n_bands_str = str(new_n_bands)
    lines[5] = lines[5].replace(n_bands_str, new_n_bands_str)

    f = open(output_z_path, "w")
    f.writelines(lines)
    f.close()
    return
