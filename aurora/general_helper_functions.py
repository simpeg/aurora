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
TEST_PATH = AURORA_PATH.joinpath("tests")
CONFIG_PATH = AURORA_PATH.joinpath("aurora", "config")
BAND_SETUP_PATH = CONFIG_PATH.joinpath("emtf_band_setup")


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


# TODO: Add test for execute_command or delete.
# def execute_command(cmd, **kwargs):
#     """
#     Executes command in terminal from script.
#
#     Parameters:
#     ----------
#     cmd : str
#         command to execute from a terminal
#     kwargs: exec_dir (str): the directory from which to execute
#     kwargs: no_exception: suppress output if exception
#
#     Other Parameters:
#         exit_status: :code:`0` is good, otherwise there is some problem
#
#     .. note:: When executing :code:`rm *` this crashes if the directory we are removing
#         from is empty
#
#     .. note:: if you can you should probably use execute_subprocess() instead
#     """
#     exec_dir = kwargs.get("exec_dir", os.path.expanduser("~/"))
#     allow_exception = kwargs.get("allow_exception", True)
#     logger.info("executing from {}".format(exec_dir))
#     cwd = os.getcwd()
#     os.chdir(exec_dir)
#     exit_status = os.system(cmd)
#     if exit_status != 0:
#         logger.info(f"exit_status of {cmd} = {exit_status}")
#         if allow_exception:
#             raise Exception(f"Failed to successfully execute \n {cmd}")
#     os.chdir(cwd)


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
