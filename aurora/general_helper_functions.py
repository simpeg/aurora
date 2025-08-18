"""
This module contains from miscellaneous functions and some global paths.
"""
import inspect

# import os
import pathlib
import scipy.io as sio
import subprocess

from loguru import logger
from pathlib import Path

import aurora
import mt_metadata
import mth5


init_file = inspect.getfile(aurora)
AURORA_PATH = Path(init_file).parent.parent
DATA_PATH = AURORA_PATH.joinpath("data")
DOCS_PATH = AURORA_PATH.joinpath("docs")
TEST_PATH = AURORA_PATH.joinpath("tests")
CONFIG_PATH = AURORA_PATH.joinpath("aurora", "config")
BAND_SETUP_PATH = CONFIG_PATH.joinpath("emtf_band_setup")
PROCESSING_TEMPLATES_PATH = CONFIG_PATH.joinpath("templates")

mt_metadata_init_file = inspect.getfile(mt_metadata)
MT_METADATA_PATH = pathlib.Path(mt_metadata_init_file).parent.parent
MT_METADATA_FEATURES_TEST_HELPERS_PATH = MT_METADATA_PATH.joinpath(
    "mt_metadata", "features", "test_helpers"
)


def get_test_path() -> pathlib.Path:
    """
    Gets the path to where the test are.

    Returns
    -------
    test_path: pathlib.Path
        Object that points to where aurora's tests are.
    """
    test_path = AURORA_PATH.joinpath("tests")
    if not test_path.exists():
        msg = (
            f"Could not locate test directory {TEST_PATH}\n "
            f"This is most likely because aurora was installed from pypi or conda forge\n"
            f"TEST_PATH should be replaced with DATA_PATH"
        )
        logger.warning(msg)
    return test_path


try:
    FIGURES_PATH = DATA_PATH.joinpath("figures")
    FIGURES_PATH.mkdir(exist_ok=True, parents=True)
except OSError:
    FIGURES_PATH = None
mt_metadata_init = inspect.getfile(mt_metadata)
MT_METADATA_DATA = Path(mt_metadata_init).parent.parent.joinpath("data")


def count_lines(file_name):
    """
    acts like wc -l in unix,
    raise FileNotFoundError: if file_name does not exist.

    Parameters
    ----------
    file_name: str or pathlib.Path
        The file to apply line counting to

    Returns
    -------
    num_lines: int
        Number of lines present in fileName or -1 if file does not exist

    """
    i = -1
    with open(file_name) as f:
        for i, l in enumerate(f):
            pass
    num_lines = i + 1
    return num_lines


def execute_subprocess(cmd, **kwargs):
    """
    A wrapper for subprocess.call

    Parameters
    ----------
    cmd : string
        command as it would be typed in a terminal
    kwargs: denotes keyword arguments that would be passed to subprocess

    """
    exit_status = subprocess.call([cmd], shell=True, **kwargs)
    if exit_status != 0:
        raise Exception("Failed to execute \n {}".format(cmd))
    return


def replace_in_file(file_path: pathlib.Path, old: str, new: str) -> None:
    """
        Replace all instances of 'old' with 'new' in the given file.
        :param file_path: Path to the file where replacements should be made.

    """
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        lines: list[str] = f.readlines()

    updated: list[str] = [line.replace(old, new) for line in lines]

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(updated)

    logger.info(f"Updated: {file_path}")


def save_to_mat(data, variable_name, filename):
    """
    Saves numpy array in matlab format.

    Example Usage:
    x = X.to_array(dim="channel")
    save_to_mat(x.data, "x", "x.mat")

    Reading into matlab or Octave:
    tmp = load("x.mat");
    data = tmp.x;

    Parameters
    ----------
    data : numpy array
        the data to save to file.  its fine if this is complex-valued.
    variable_name : string
        The name that we use to reference the variable within the struct in the matfile.
    filename : string
        The filepath to output

    """
    sio.savemat(filename, {variable_name: data})
    return


class DotDict(dict):
    """
    Helper function for debugging, casts a dict so that its values
    can be accessed via dict.key as well as dict["key"]

    Usage:
    dot_dict = DotDict(basic_dict)
    """

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name, value):
        self[name] = value
