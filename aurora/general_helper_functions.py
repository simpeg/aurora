import inspect
import os
import scipy.io as sio
import subprocess
import xarray as xr

from pathlib import Path

import aurora
import mt_metadata

init_file = inspect.getfile(aurora)
AURORA_PATH = Path(init_file).parent.parent
TEST_PATH = AURORA_PATH.joinpath("tests")
SANDBOX = AURORA_PATH.joinpath("aurora", "sandbox")
CONFIG_PATH = AURORA_PATH.joinpath("aurora", "config")
BAND_SETUP_PATH = CONFIG_PATH.joinpath("emtf_band_setup")
DATA_PATH = SANDBOX.joinpath("data")
DATA_PATH.mkdir(exist_ok=True, parents=True)
FIGURES_PATH = DATA_PATH.joinpath("figures")
FIGURES_PATH.mkdir(exist_ok=True, parents=True)
TEST_BAND_FILE = DATA_PATH.joinpath("bandtest.nc")
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


# <HDF5 save/load complex valued data>
def save_complex(data_array, *args, **kwargs):
    """
    netcdf and h5 do not handle complex values.  This method is a workaround.
    https://stackoverflow.com/questions/47162983/how-to-save-xarray-dataarray-with-complex128-data-to-netcdf
    Example Usage:
    band_da is an xarray
    save_complex(band_da, TEST_BAND_FILE)
    band_da = read_complex(TEST_BAND_FILE)

    Parameters
    ----------
    data_array
    args
    kwargs

    Returns
    -------

    """
    ds = xr.Dataset({"real": data_array.real, "imag": data_array.imag})
    return ds.to_netcdf(*args, **kwargs)


def read_complex(*args, **kwargs):
    ds = xr.open_dataset(*args, **kwargs)
    return ds["real"] + ds["imag"] * 1j


# </HDF5 save/load complex valued data>


def save_to_mat(data, variable_name, filename):
    """
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

    Returns
    -------

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
