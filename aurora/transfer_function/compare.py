"""
Module to compare two transfer functions.

"""

import pathlib
from typing import Union

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from mt_metadata.transfer_functions.core import TF
from scipy.interpolate import interp1d


class CompareTF:
    def __init__(
        self,
        tf_01: Union[str, pathlib.Path, TF],
        tf_02: Union[str, pathlib.Path, TF],
    ):
        """
        Class to compare two transfer functions.

        Parameters
        ----------
        tf_01
            First transfer function (file path or TF object)
        tf_02
            Second transfer function (file path or TF object)
        """
        if isinstance(tf_01, (str, pathlib.Path)):
            self.tf_01 = TF()
            self.tf_01.read(tf_01)
        elif isinstance(tf_01, TF):
            self.tf_01 = tf_01
        else:
            raise TypeError("tf_01 must be a file path or TF object")

        if isinstance(tf_02, (str, pathlib.Path)):
            self.tf_02 = TF()
            self.tf_02.read(tf_02)
        elif isinstance(tf_02, TF):
            self.tf_02 = tf_02
        else:
            raise TypeError("tf_02 must be a file path or TF object")

    def plot_two_transfer_functions(
        self,
        label_01="emtf",
        label_02="aurora",
        save_plot_path=None,
    ):
        """
        Plots two transfer functions for comparison.

        Parameters
        ----------
        label_01
            Label for the first transfer function
        label_02
            Label for the second transfer function
        save_plot_path
            Path to save the plot (optional)

        Returns
        -------

        """
        fig = plt.figure(figsize=(12, 6))

        comp_dict = {1: "$Z_{xx}$", 2: "$Z_{xy}$", 3: "$Z_{yx}$", 4: "$Z_{yy}$"}

        for ii in range(2):
            for jj in range(2):
                plot_num_res = 1 + ii * 2 + jj
                plot_num_phase = 5 + ii * 2 + jj
                ax = fig.add_subplot(2, 4, plot_num_res)
                ax.loglog(
                    self.tf_01.period,
                    0.2
                    * self.tf_01.period
                    * np.abs(self.tf_01.impedance.data[:, ii, jj]) ** 2,
                    label=label_01,
                    marker="s",
                    markersize=7,
                    color="k",
                )
                ax.loglog(
                    self.tf_02.period,
                    0.2
                    * self.tf_02.period
                    * np.abs(self.tf_02.impedance.data[:, ii, jj]) ** 2,
                    label=label_02,
                    marker="o",
                    markersize=4,
                    color="r",
                )
                ax.set_title(comp_dict[plot_num_res])
                # ax.set_xlabel("Period (s)")
                if plot_num_res == 1:
                    ax.set_ylabel("Apparent Resistivity ($\Omega \cdot m$)")
                    ax.legend()
                ax.grid(True, which="both", ls="--", lw=0.5, color="gray")

                ax2 = fig.add_subplot(2, 4, plot_num_phase)
                ax2.semilogx(
                    self.tf_01.period,
                    np.degrees(np.angle(self.tf_01.impedance.data[:, ii, jj])),
                    label=label_01,
                    marker="s",
                    markersize=7,
                    color="k",
                )
                ax2.semilogx(
                    self.tf_02.period,
                    np.degrees(np.angle(self.tf_02.impedance.data[:, ii, jj])),
                    label=label_02,
                    marker="o",
                    markersize=4,
                    color="r",
                )
                ax2.set_xlabel("Period (s)")
                if plot_num_phase == 5:
                    ax2.set_ylabel("Phase (degrees)")
                    ax2.legend()
                ax2.grid(True, which="both", ls="--", lw=0.5, color="gray")

        fig.tight_layout()
        plt.show()

        if save_plot_path is not None:
            fig.savefig(save_plot_path, dpi=300)
            logger.info(f"Saved comparison plot to {save_plot_path}")
            plt.close(fig)

    def _interpolate_complex_array(
        self,
        source_periods: np.ndarray,
        source_array: np.ndarray,
        target_periods: np.ndarray,
    ) -> np.ndarray:
        """Interpolate complex array onto target periods."""
        interp_array = np.zeros(
            (len(target_periods),) + source_array.shape[1:], dtype=complex
        )

        for i in range(source_array.shape[1]):
            for j in range(source_array.shape[2]):
                real_interp = interp1d(
                    source_periods,
                    source_array[:, i, j].real,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                imag_interp = interp1d(
                    source_periods,
                    source_array[:, i, j].imag,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                interp_array[:, i, j] = real_interp(target_periods) + 1j * imag_interp(
                    target_periods
                )

        return interp_array

    def interpolate_tf_to_common_periods(self):
        """
        Interpolate two transfer functions onto common period range.

        Uses the overlapping period range and creates a common grid for comparison.

        Parameters
        ----------
        tf1 : TF
            First transfer function
        tf2 : TF
            Second transfer function

        Returns
        -------
        periods_common : ndarray
            Common period array
        z1_interp : ndarray
            Interpolated impedance from tf1, shape (n_periods, 2, 2)
        z2_interp : ndarray
            Interpolated impedance from tf2, shape (n_periods, 2, 2)
        z1_err_interp : ndarray
            Interpolated impedance errors from tf1
        z2_err_interp : ndarray
            Interpolated impedance errors from tf2
        """
        # Get period arrays
        p1 = self.tf_01.period
        p2 = self.tf_02.period

        # Find overlapping range
        p_min = max(p1.min(), p2.min())
        p_max = min(p1.max(), p2.max())

        # Create common period grid (logarithmic spacing)
        n_periods = min(len(p1), len(p2))
        periods_common = np.logspace(np.log10(p_min), np.log10(p_max), n_periods)

        if self.tf_01.has_impedance() and self.tf_02.has_impedance():
            # Interpolate tf1 impedance (log-log for real and imag separately)
            z1_interp = self._interpolate_complex_array(
                p1, self.tf_01.impedance, periods_common
            )
            z1_err_interp = self._interpolate_complex_array(
                p1, self.tf_01.impedance_error, periods_common
            )

            z2_interp = self._interpolate_complex_array(
                p2, self.tf_02.impedance, periods_common
            )
            z2_err_interp = self._interpolate_complex_array(
                p2, self.tf_02.impedance_error, periods_common
            )
        else:
            z1_interp = None
            z2_interp = None
            z1_err_interp = None
            z2_err_interp = None

        if self.tf_01.has_tipper() and self.tf_02.has_tipper():
            t1_interp = self._interpolate_complex_array(
                p1, self.tf_01.tipper, periods_common
            )
            t2_interp = self._interpolate_complex_array(
                p2, self.tf_02.tipper, periods_common
            )
            t1_err_interp = self._interpolate_complex_array(
                p1, self.tf_01.tipper_error, periods_common
            )
            t2_err_interp = self._interpolate_complex_array(
                p2, self.tf_02.tipper_error, periods_common
            )
        else:
            t1_interp = None
            t2_interp = None
            t1_err_interp = None
            t2_err_interp = None

        return (
            periods_common,
            z1_interp,
            z2_interp,
            z1_err_interp,
            z2_err_interp,
            t1_interp,
            t2_interp,
            t1_err_interp,
            t2_err_interp,
        )

    def compare_transfer_functions(
        self,
        rtol: float = 1e-2,
        atol: float = 1e-2,
    ) -> dict:
        """
        Compare transfer functions between two transfer_functions objects.

        Compares transfer_functions, sigma_e, and sigma_s arrays. If periods
        don't match, interpolates one onto the other.

        Parameters
        ----------
        rtol: float
            Relative tolerance for np.allclose, defaults to 1e-2
        atol: float
            Absolute tolerance for np.allclose, defaults to 1e-2

        Returns
        -------
        comparison: dict
            Dictionary containing:
            - "periods_match": bool, whether periods are identical
            - "transfer_functions_close": bool
            - "sigma_e_close": bool
            - "sigma_s_close": bool
            - "max_tf_diff": float, max absolute difference in transfer functions
            - "max_sigma_e_diff": float
            - "max_sigma_s_diff": float
            - "periods_used": np.ndarray of periods used for comparison
        """

        (
            periods_common,
            z1,
            z2,
            z1_err,
            z2_err,
            t1,
            t2,
            t1_err,
            t2_err,
        ) = self.interpolate_tf_to_common_periods()

        result = {}
        result["impedance_amplitude_close"] = None
        result["impedance_amplitude_max_diff"] = None
        result["impedance_phase_close"] = None
        result["impedance_phase_max_diff"] = None
        result["impedance_error_close"] = None
        result["impedance_error_max_diff"] = None
        result["impedance_ratio"] = None
        result["tipper_amplitude_close"] = None
        result["tipper_amplitude_max_diff"] = None
        result["tipper_phase_close"] = None
        result["tipper_phase_max_diff"] = None
        result["tipper_error_close"] = None
        result["tipper_error_max_diff"] = None
        result["tipper_ratio"] = None

        result["periods_used"] = periods_common

        # Compare arrays
        if z1 is not None and z2 is not None:
            result["impedance_amplitude_close"] = np.allclose(
                np.abs(z1), np.abs(z2), rtol=rtol, atol=atol
            )
            result["impedance_amplitude_max_diff"] = np.max(
                np.abs(np.abs(z1) - np.abs(z2))
            )

            result["impedance_phase_close"] = np.allclose(
                np.angle(z1), np.angle(z2), rtol=rtol, atol=atol
            )
            result["impedance_phase_max_diff"] = np.max(
                np.abs(np.angle(z1) - np.angle(z2))
            )

            result["impedance_error_close"] = np.allclose(
                np.abs(z1_err), np.abs(z2_err), rtol=rtol, atol=atol
            )
            result["impedance_error_max_diff"] = np.max(
                np.abs(np.abs(z1_err) - np.abs(z2_err))
            )

            result["impedance_ratio"] = {}
            for ii in range(2):
                for jj in range(2):
                    if ii != jj:
                        ratio = np.median(z1[:, ii, jj] / z2[:, ii, jj])
                        key = f"Z_{ii}{jj}"
                        result["impedance_ratio"][key] = ratio

        else:
            result["tipper_amplitude_close"] = np.allclose(
                np.abs(t1), np.abs(t2), rtol=rtol, atol=atol
            )
            result["tipper_amplitude_max_diff"] = np.max(
                np.abs(np.abs(t1) - np.abs(t2))
            )

            result["tipper_phase_close"] = np.allclose(
                np.angle(t1), np.angle(t2), rtol=rtol, atol=atol
            )
            result["tipper_phase_max_diff"] = np.max(
                np.abs(np.angle(t1) - np.angle(t2))
            )

            result["tipper_error_close"] = np.allclose(
                np.abs(t1_err), np.abs(t2_err), rtol=rtol, atol=atol
            )
            result["tipper_error_max_diff"] = np.max(
                np.abs(np.abs(t1_err) - np.abs(t2_err))
            )

            result["tipper_ratio"] = {}
            for ii in range(2):
                for jj in range(2):
                    if ii != jj:
                        ratio = np.median(t1[:, ii, jj] / t2[:, ii, jj])
                        key = f"T_{ii}{jj}"
                        result["tipper_ratio"][key] = ratio

        return result
