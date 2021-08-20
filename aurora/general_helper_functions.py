import inspect
import os
import subprocess
import xarray as xr

from pathlib import Path

import aurora

init_file = inspect.getfile(aurora)
AURORA_PATH = Path(init_file).parent.parent
TEST_PATH = AURORA_PATH.joinpath("tests")
SANDBOX = AURORA_PATH.joinpath("aurora", "sandbox")
DATA_PATH = SANDBOX.joinpath("data")
DATA_PATH.mkdir(exist_ok=True, parents=True)
FIGURES_PATH = DATA_PATH.joinpath("figures")
FIGURES_PATH.mkdir(exist_ok=True, parents=True)
TEST_BAND_FILE = DATA_PATH.joinpath("bandtest.nc")


def execute_subprocess(cmd, **kwargs):
    """

    Parameters
    ----------
    cmd : string
        command as it would be typed in a terminal
    kwargs

    Returns
    -------

    """
    """
    A wrapper for subprocess.call
    """
    exit_status = subprocess.call([cmd], shell=True, **kwargs)
    if exit_status != 0:
        raise Exception("Failed to execute \n {}".format(cmd))
    return


def execute_command(cmd, **kwargs):
    """
    Executes command in terminal from script.

    Parameters:
        cmd (str): command to exectute from a terminal
        kwargs: exec_dir (str): the directory from which to execute
        kwargs: no_exception: suppress output if exception

    Other Parameters:
        exit_status: :code:`0` is good, otherwise there is some problem

    .. note:: When executing :code:`rm *` this crashes if the directory we are removing
        from is empty

    .. note:: if you can you should probably use execute_subprocess() instead
    """
    exec_dir = kwargs.get("exec_dir", os.path.expanduser("~/"))
    allow_exception = kwargs.get("allow_exception", True)
    print("executing from {}".format(exec_dir))
    cwd = os.getcwd()
    os.chdir(exec_dir)
    exit_status = os.system(cmd)
    if exit_status != 0:
        print(f"exit_status of {cmd} = {exit_status}")
        if allow_exception:
            raise Exception(f"Failed to successfully execute \n {cmd}")
    os.chdir(cwd)


# <NETCDF DOESN'T HANDLE COMPLEX>
# https://stackoverflow.com/questions/47162983/how-to-save-xarray-dataarray
# -with-complex128-data-to-netcdf


def save_complex(data_array, *args, **kwargs):
    ds = xr.Dataset({"real": data_array.real, "imag": data_array.imag})
    return ds.to_netcdf(*args, **kwargs)


def read_complex(*args, **kwargs):
    ds = xr.open_dataset(*args, **kwargs)
    return ds["real"] + ds["imag"] * 1j


# Usage:
# band_da is an xarray
# save_complex(band_da, TEST_BAND_FILE)
# band_da = read_complex(TEST_BAND_FILE)
# </NETCDF DOESN'T HANDLE COMPLEX>
