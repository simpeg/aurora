"""

This module contains methods associated with legacy EMTF z-file TF format.

Development notes:
They extract info needed to setup emtf_z files.
These methods can possibly be moved under mt_metadata, or deprecated.

"""
import pathlib
from typing import Optional, Union

from loguru import logger


EMTF_CHANNEL_ORDER = ["hx", "hy", "hz", "ex", "ey"]


def get_default_orientation_block(n_ch: int = 5) -> list:
    """
    creates a text block like the part of the z-file that holds channel orientations.

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


def clip_bands_from_z_file(
    z_path: Union[str, pathlib.Path],
    n_bands_clip: int,
    output_z_path: Optional[Union[str, pathlib.Path, None]] = None,
    n_sensors: Optional[int] = 5,
):
    """
    This function clips periods off the end of an EMTF legacy z_file.

    Development Notes:
    It can come in handy for manipulating matlab results of synthetic data.

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
        logger.info("WARNING n_sensors==4 NOT TESTED")

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
