# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 10:38:08 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
import warnings

from aurora.config import BANDS_DEFAULT_FILE
from aurora.config.config_creator import ConfigCreator
from aurora.pipelines.process_mth5 import process_mth5
from aurora.transfer_function.kernel_dataset import KernelDataset
from aurora.pipelines.run_summary import RunSummary

warnings.filterwarnings("ignore")

# =============================================================================


class EstimateTransferFunction:
    """convenience class to process MT data"""

    def __init__(self, **kwargs):
        self.local_mth5_path = None
        self.remote_mth5_path = None

        self.local_station = None
        self.remote_station = None

        self.minimum_run_duration_in_seconds = 10000
        self.runs_to_process = None
        self.runs_to_drop = None

        self.bands_file_path = None

        self.config = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def local_mth5_path(self):
        """ """
        return self._local_mth5_path

    @local_mth5_path.setter
    def local_mth5_path(self, local_path):
        """local MTH5 path, make sure its a :class:`pathlib.Path` object

        Parameters
        ----------
        local_path :


        Returns
        -------

        """

        if local_path is None:
            self._local_mth5_path = None
        elif isinstance(local_path, (str, Path)):
            self._local_mth5_path = Path(local_path)
        else:
            raise TypeError(
                "local_mth5_path must be a string or Path object, "
                f"not {type(local_path)}"
            )

    @property
    def remote_mth5_path(self):
        """ """
        return self._remote_mth5_path

    @remote_mth5_path.setter
    def remote_mth5_path(self, remote_path):
        """remote MTH5 path, make sure its a :class:`pathlib.Path` object

        Parameters
        ----------
        remote_path :


        Returns
        -------

        """

        if remote_path is None:
            self._remote_mth5_path = None
        elif isinstance(remote_path, (str, Path)):
            self._remote_mth5_path = Path(remote_path)
        else:
            raise TypeError(
                "remote_mth5_path must be a string or Path object, "
                f"not {type(remote_path)}"
            )

    @property
    def bands_file_path(self):
        """ """
        return self._band_file_path

    @bands_file_path.setter
    def bands_file_path(self, band_file_path):
        """set band file path

        check to make sure it exists

        If none use BANDS_DEFAULT_FILE

        Parameters
        ----------
        band_file_path : path to bands file

        """

        if band_file_path is not None:
            if isinstance(band_file_path, (str, Path)):
                band_file_path = Path(band_file_path)
            if not band_file_path.exists():
                raise IOError(f"Could not find {band_file_path}, check path.")
            self._band_file_path = band_file_path
        else:
            self._band_file_path = Path(BANDS_DEFAULT_FILE)

    @property
    def run_summary(self):
        """create a run summary of the local mth5

        Parameters
        ----------

        Returns
        -------
        run_summary: RunSummary object


        """

        mth5_run_summary = RunSummary()
        if self.local_mth5_path is not None:
            h5_list = [self.local_mth5_path]
            if self.remote_mth5_path is not None:
                if self.local_mth5_path == self.remote_mth5_path:
                    h5_list = [self.local_mth5_path]
                else:
                    h5_list = [self.local_mth5_path, self.remote_mth5_path]
        else:
            raise ValueError(
                "'local_mth5_path' is None, cannot create a RunSummary."
            )

        mth5_run_summary.from_mth5s(h5_list)
        return mth5_run_summary.clone()

    @property
    def kernel_dataset(self):
        """create a kernel dataset

        Parameters
        ----------

        Returns
        -------
        kernel_dataset: KernelDataset Object

        """

        kernel_dataset = KernelDataset()
        kernel_dataset.from_run_summary(self.run_summary, self.local_station)

        if self.minimum_run_duration_in_seconds is not None:
            kernel_dataset.drop_runs_shorter_than(
                self.minimum_run_duration_in_seconds
            )

        if self.runs_to_process is not None:
            kernel_dataset.select_station_runs(
                self._get_station_runs_dict(self.runs_to_process), "keep"
            )
        if self.runs_to_drop is not None:
            kernel_dataset.select_station_runs(
                self._get_station_runs_dict(self.runs_to_drop), "drop"
            )

        return kernel_dataset

    def _get_station_runs_dict(self, runs):
        """
        get a dictionary of runs to process

        Parameters
        ----------
        runs : list of runs to process or skip


        Returns
        -------
        dictionary
            dictionary for each station of which runs to process

        """

        if runs is not None:
            if isinstance(runs, (str)):
                return {self.local_station: [runs]}
            elif isinstance(runs, (list, tuple)):
                return {self.local_station: list(runs)}
            else:
                raise TypeError("runs must be a string or list of strings")

    def create_configuration_object(self):
        """
        create configuration object as an attribute

        Parameters
        ----------

        Returns
        -------
        config object
            Configuration on how to run Aurora

        """

        cc = ConfigCreator()
        self.config = cc.create_from_kernel_dataset(
            self.kernel_dataset, emtf_band_file=self.bands_file_path
        )

    def _save_tf_in_mth5(self, tf_obj):
        """
        save TransferFunction object in the local MTH5 file

        Parameters
        ----------
        tf_obj : TF Object
            Transfer Function object

        """

        from mth5.mth5 import MTH5

        m = MTH5()
        m.open_mth5(self.local_mth5_path)
        m.add_tf(tf_obj)
        m.close_mth5()

    def estimate_tf(self, show_plot=True, save_in_mth5=False):
        """

        Parameters
        ----------
        show_plot : bool, optional
            True to show plots False to not, defaults to True
        save_in_mth5 : bool, optional
            True to save TF in local MTH5 file, defaults to False

        Returns
        -------
        TF object
            Transfer Function object

        """
        if self.config is None:
            self.create_configuration_object()

        tf_object = process_mth5(
            self.config,
            self.kernel_dataset,
            units="MT",
            show_plot=show_plot,
            z_file_path=None,
        )

        if save_in_mth5:
            self._save_tf_in_mth5(tf_object)

        return tf_object
