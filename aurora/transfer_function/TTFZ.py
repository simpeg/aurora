"""
follows Gary's TTFZ.m in
iris_mt_scratch/egbert_codes-20210121T193218Z-001/egbert_codes/matlabPrototype_10-13-20/TF/classes
"""
import numpy as np

# from aurora.transfer_function.TTF import TTF
from aurora.transfer_function.base import TransferFunction


class TTFZ(TransferFunction):
    """
    subclass to support some more MT impedance specficic functions  --
    initially just apparent resistivity and pbase for diagonal elements
    + rotation/fixed coordinate system

    properties
    rho
    rho_se
    phi
    phi_se
    """

    def __init__(self, *args, **kwargs):
        super(TTFZ, self).__init__(*args, **kwargs)

    def apparent_resistivity(self, units="SI"):
        """
        ap_res(...) : computes app. res., phase, errors, given imped., cov.
        %USAGE: [rho,rho_se,ph,ph_se] = ap_res(z,sig_s,sig_e,periods) ;
        % Z = array of impedances (from Z_***** file)
        % sig_s = inverse signal covariance matrix (from Z_****** file)
        % sig_e = residual covariance matrix (from Z_****** file)
        % periods = array of periods (sec)
        Returns
        -------

        """

        rad_deg = 180 / np.pi
        # off - diagonal impedances
        self.rho = np.zeros((self.num_bands, 2))
        self.rho_se = np.zeros((self.num_bands, 2))
        self.phi = np.zeros((self.num_bands, 2))
        self.phi_se = np.zeros((self.num_bands, 2))
        Zxy = self.tf.loc["ex", "hy", :].data
        Zyx = self.tf.loc["ey", "hx", :].data

        # standard deviation of real and imaginary parts of impedance
        Zxy_se = self.standard_error().loc["ex", "hy", :].data / np.sqrt(2)
        Zyx_se = self.standard_error().loc["ey", "hx", :].data / np.sqrt(2)
        # apparent resistivities
        # rxy = self.T * (abs(Zxy) ** 2) / 5.
        # ryx = self.T * (abs(Zyx) ** 2) / 5.
        if units == "SI":
            rxy = 2e-7 * self.periods * (abs(Zxy) ** 2)
            ryx = 2e-7 * self.periods * (abs(Zyx) ** 2)
            print("STANDARD ERRORS NOT CORRECT FOR SI")
            rxy_se = 2 * np.sqrt(self.periods * rxy / 5) * Zxy_se
            ryx_se = 2 * np.sqrt(self.periods * ryx / 5) * Zyx_se
        elif units == "MT":
            rxy = self.periods * (abs(Zxy) ** 2) / 5.0
            ryx = self.periods * (abs(Zyx) ** 2) / 5.0
            rxy_se = 2 * np.sqrt(self.periods * rxy / 5) * Zxy_se
            ryx_se = 2 * np.sqrt(self.periods * ryx / 5) * Zyx_se
        else:
            print("ERROR: only SI and MT units supported")
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
