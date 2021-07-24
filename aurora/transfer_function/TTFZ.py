"""
follows Gary's TTFZ.m in
iris_mt_scratch/egbert_codes-20210121T193218Z-001/egbert_codes/matlabPrototype_10-13-20/TF/classes
"""
import numpy as np

#from aurora.transfer_function.TTF import TTF
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
        if self.num_channels_out == 2:
            zRows = [0,1]
        elif self.num_channels_out==3:
            zRows = [1,2]
        else:
            print('ap_res only works for 2 or 3 output channels')
            raise Exception

        rad_deg = 180 / np.pi
        # off - diagonal impedances
        self.rho = np.zeros((self.num_bands, 2))
        self.rho_se = np.zeros((self.num_bands, 2))
        self.phi = np.zeros((self.num_bands, 2))
        self.phi_se = np.zeros((self.num_bands, 2))
        Zxy = self.TF[zRows[0], 1,:].squeeze()
        Zyx = self.TF[zRows[1], 0,:].squeeze()

        # standard deviation of real and imaginary parts of impedance
        Zxy_se = self.standard_error()[zRows[0], 1,:].squeeze() / np.sqrt(2)
        Zyx_se = self.standard_error()[zRows[1], 0,:].squeeze() / np.sqrt(2)

        # apparent resistivities
        #rxy = self.T * (abs(Zxy) ** 2) / 5.
        #ryx = self.T * (abs(Zyx) ** 2) / 5.
        if units=="SI":
            rxy = 2e-7 * self.T * (abs(Zxy) ** 2)
            ryx = 2e-7*  self.T * (abs(Zyx) ** 2)
            print("ERRORS NOT CORRECT FOR SI")
            rxy_se = 2 * np.sqrt(self.T * rxy / 5) * Zxy_se
            ryx_se = 2 * np.sqrt(self.T * ryx / 5) * Zyx_se
        elif units=="MT":
            rxy = self.T * (abs(Zxy) ** 2) / 5.
            ryx = self.T * (abs(Zyx) ** 2) / 5.
            rxy_se = 2 * np.sqrt(self.T * rxy / 5) * Zxy_se
            ryx_se = 2 * np.sqrt(self.T * ryx / 5) * Zyx_se
        else:
            print("ERROR: only SI and MT units supported")
            raise Exception

        self.rho[:,:] = np.vstack((rxy, ryx)).T
        self.rho_se[:,:] = np.vstack((rxy_se, ryx_se)).T;

        # phases
        pxy = rad_deg * np.arctan(np.imag(Zxy) / np.real(Zxy))
        pyx = rad_deg * np.arctan(np.imag(Zyx) / np.real(Zyx))
        self.phi[:,:] = np.vstack((pxy, pyx)).T

        pxy_se = rad_deg * Zxy_se / np.abs(Zxy)
        pyx_se = rad_deg * Zyx_se / np.abs(Zyx)

        self.phi_se  = np.vstack((pxy_se, pyx_se)).T
        return


def test_ttfz():
    from aurora.transfer_function.transfer_function_header \
        import TransferFunctionHeader
    tfh = TransferFunctionHeader()
    ttfz = TTFZ(tfh, 32)


def main():
    test_ttfz()

if __name__ == '__main__':
    main()