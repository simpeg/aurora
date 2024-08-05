"""
This module contains an extension of aurora's TransferFunction base class.
This class can return estimates of standard error, apparent resistivity and phase.

Development Notes:
This class follows  Gary's legacy matlab code  TTFZ.m from
iris_mt_scratch/egbert_codes-20210121T193218Z-001/egbert_codes/matlabPrototype_10-13-20/TF/classes
TODO: This should be replaced by methods in mtpy.
"""
import numpy as np
import xarray as xr
from loguru import logger

from aurora.transfer_function.base import TransferFunction


class TTFZ(TransferFunction):
    """
    subclass to support some more MT impedance specific functions  --
    initially just apparent resistivity and phase for diagonal elements.
    + rotation/fixed coordinate system

    TODO: This class should be deprecated and mt_metadata TF object should be used instead.

    """

    def __init__(self, *args, **kwargs):
        """
        Constructor

        Parameters
        ----------
        args: passed through to base class
        kwargs: passed through to base class
        """
        super(TTFZ, self).__init__(*args, **kwargs)

    def standard_error(self):
        """
        estimate the standard error, used for error bars and inversion.

        Development Notes:
        The standard error is normally thought of as the sqrt of the error variance.
        since the code here sets std_err = np.sqrt(np.abs(cov_ss_inv * cov_nn))
        that means the inverse signal covariance times the noise covariance is like the error variance.

        Returns
        -------
        standard_error: xr.DataArray
        """
        stderr = np.zeros(self.tf.data.shape)
        standard_error = xr.DataArray(
            stderr,
            dims=["output_channel", "input_channel", "period"],
            coords={
                "output_channel": self.tf_header.output_channels,
                "input_channel": self.tf_header.input_channels,
                "period": self.periods,
            },
        )
        for out_ch in self.tf_header.output_channels:
            for inp_ch in self.tf_header.input_channels:
                for T in self.periods:
                    cov_ss_inv = self.cov_ss_inv.loc[inp_ch, inp_ch, T]
                    cov_nn = self.cov_nn.loc[out_ch, out_ch, T]
                    std_err = np.sqrt(np.abs(cov_ss_inv * cov_nn))
                    standard_error.loc[out_ch, inp_ch, T] = std_err

        return standard_error

    def apparent_resistivity(self, channel_nomenclature, units="SI"):
        """
        Computes the apparent resistivity and phase.

        Development notes:
        Original Matlab Documentation:
        ap_res(...) : computes app. res., phase, errors, given imped., cov.
        %USAGE: [rho,rho_se,ph,ph_se] = ap_res(z,sig_s,sig_e,periods) ;
        % Z = array of impedances (from Z_***** file)
        % sig_s = inverse signal covariance matrix (from Z_****** file)
        % sig_e = residual covariance matrix (from Z_****** file)
        % periods = array of periods (sec)

        Parameters
        ----------
        units: str
            one of ["MT","SI"]
        channel_nomenclature:
        mt_metadata.transfer_functions.processing.aurora.channel_nomenclature.ChannelNomenclature
            has a dict that maps the channel names in TF to the standard channel labellings.

        """
        ex, ey, hx, hy, hz = channel_nomenclature.unpack()
        rad_deg = 180 / np.pi
        # off - diagonal impedances
        self.rho = np.zeros((self.num_bands, 2))
        self.rho_se = np.zeros((self.num_bands, 2))
        self.phi = np.zeros((self.num_bands, 2))
        self.phi_se = np.zeros((self.num_bands, 2))
        Zxy = self.tf.loc[ex, hy, :].data
        Zyx = self.tf.loc[ey, hx, :].data

        # standard deviation of real and imaginary parts of impedance
        Zxy_se = self.standard_error().loc[ex, hy, :].data / np.sqrt(2)
        Zyx_se = self.standard_error().loc[ey, hx, :].data / np.sqrt(2)

        if units == "SI":
            rxy = 2e-7 * self.periods * (abs(Zxy) ** 2)
            ryx = 2e-7 * self.periods * (abs(Zyx) ** 2)
            # print("Correct the standard errors for SI units")
            Zxy_se *= 1e-3
            Zyx_se *= 1e-3
            rxy_se = 2 * np.sqrt(self.periods * rxy / 5) * Zxy_se
            ryx_se = 2 * np.sqrt(self.periods * ryx / 5) * Zyx_se
        elif units == "MT":
            rxy = 2e-1 * self.periods * (abs(Zxy) ** 2)
            ryx = 2e-1 * self.periods * (abs(Zyx) ** 2)
            rxy_se = 2 * np.sqrt(self.periods * rxy / 5) * Zxy_se
            ryx_se = 2 * np.sqrt(self.periods * ryx / 5) * Zyx_se
        else:
            logger.error("ERROR: only SI and MT units supported")
            raise Exception

        self.rho[:, :] = np.vstack((rxy, ryx)).T
        self.rho_se[:, :] = np.vstack((rxy_se, ryx_se)).T

        # phases
        pxy = rad_deg * np.arctan(np.imag(Zxy) / np.real(Zxy))
        pyx = rad_deg * np.arctan(np.imag(Zyx) / np.real(Zyx))
        self.phi[:, :] = np.vstack((pxy, pyx)).T

        pxy_se = rad_deg * Zxy_se / np.abs(Zxy)
        pyx_se = rad_deg * Zyx_se / np.abs(Zyx)

        self.phi_se = np.vstack((pxy_se, pyx_se)).T
        return
