"""
Sets up paths for synthetic data testing.

"""
import pathlib

from aurora.general_helper_functions import DATA_PATH
from loguru import logger


class SyntheticTestPaths:
    def __init__(self, sandbox_path=None):
        """
        ivars:
        - ascii_data_path, where the ascii data are stored
        - mth5_path: this is where the mth5 files get written to.
        - config folder: this is where the config files get saved while tests are running
        - aurora_results folder: this is where the processing results get saved during test execution



        Parameters
        ----------
        sandbox_path: None or pathlib.Path
        """

        self._root_path = DATA_PATH.joinpath("synthetic")
        if sandbox_path is None:
            logger.debug(f"synthetic sandbox path is being set to {self._root_path}")
            self._sandbox_path = self._root_path

        # READ ONLY OK
        self.ascii_data_path = self._root_path.joinpath("ascii")

        # NEED WRITE ACCESS
        # Consider using an environment variable for sandbox_path
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
        self.aurora_results_path.mkdir(parents=True, exist_ok=True)
        self.config_path.mkdir(parents=True, exist_ok=True)
        self.emtf_results_path.mkdir(parents=True, exist_ok=True)
        self.mth5_path.mkdir(parents=True, exist_ok=True)
