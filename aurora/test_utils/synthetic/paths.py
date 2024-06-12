"""
Sets up paths for synthetic data testing.

The DATA_PATH from general_helper_functions has traditionally had the
synthetic ascii data, but this is now stored in MTH5.
"""
import pathlib

from aurora.general_helper_functions import DATA_PATH
from aurora.general_helper_functions import get_mth5_ascii_data_path
from loguru import logger

DEFAULT_SANDBOX_PATH = DATA_PATH.joinpath("synthetic")


class SyntheticTestPaths:
    """
    sandbox path must be a place that has write access.  This class was created because on some
    installations we only have read access.  Originally there was a data/ folder with the synthetic
    ascii data stored there, and we created the mth5 and other data products in the same place.

    Here we have a ascii_data_path which points at the ascii files (which may be read only), but we
    also accept a kwarg for "sandbox_path" which is writable and this is where the mth5 and etc. will
    get built.

    TODO: consider creating a symlink in aurora's legacy data path that points at the mth5 ascii files.

    """

    def __init__(self, sandbox_path=None, ascii_data_path=None):
        """

        Parameters
        ----------
        sandbox_path: pathlib.Path
            A writable path where test results are stored

        ivars:
        - ascii_data_path: where the ascii data are stored
        - mth5_path: this is where the mth5 files get written to.
        - config_path: this is where the config files get saved while tests are running
        - aurora_results_path: this is where the processing results get saved during test execution
        - emtf_results_path: stores some legacy results from EMTF processing for tests/comparison.

        """
        # READ ONLY OK
        if ascii_data_path is None:
            self.ascii_data_path = get_mth5_ascii_data_path()

        # NEED WRITE ACCESS
        # Consider using an environment variable for sandbox_path
        if sandbox_path is None:
            logger.debug(
                f"synthetic sandbox path is being set to {DEFAULT_SANDBOX_PATH}"
            )
            self._sandbox_path = DEFAULT_SANDBOX_PATH
        else:
            self._sandbox_path = sandbox_path

        self.mth5_path = self._sandbox_path.joinpath("mth5")
        self.aurora_results_path = self._sandbox_path.joinpath("aurora_results")
        self.emtf_results_path = self._sandbox_path.joinpath("emtf_results")
        self.config_path = self._sandbox_path.joinpath("config")
        self.writability_check()
        # assert self.ascii_data_path.exists()

    def writability_check(self):
        """
        Placeholder
        Should check that sandbox and dirs below have write access
        If dirs are not writeable, consider
        HOME = pathlib.Path().home()
        workaround_sandbox = HOME.joinpath(".cache", "aurora", "sandbox")
        """
        pass

    def mkdirs(self):
        """
        Makes the directories that the tests will write results to.

        """
        self.aurora_results_path.mkdir(parents=True, exist_ok=True)
        self.config_path.mkdir(parents=True, exist_ok=True)
        self.emtf_results_path.mkdir(parents=True, exist_ok=True)
        self.mth5_path.mkdir(parents=True, exist_ok=True)
