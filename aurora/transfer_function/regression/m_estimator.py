"""
An extension of RegressionEstimator with properties common to both TRME and TRME_RR
"""
import numpy as np
from scipy.linalg import solve_triangular
import xarray as xr

from aurora.transfer_function.regression.base import RegressionEstimator
from aurora.transfer_function.regression.helper_functions import rme_beta


class MEstimator(RegressionEstimator):
    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        kwargs
        expectation_psi_prime : numpy array
            Expectation value of psi' (derivative of psi) -- ?? OR rho'=psi????
            same number of entries as there are output channels
            default to 1.0
            Recall start with the loss function rho. The derivative of the
            loss function is the "influence function" psi.
            Think about the Huber loss function (quadratic out to r0,
            after which it is linear). Psi is -1 until you get to -r0,
            then it increases linearly to 1 (r0)
            Psi' is something like 1 between -r0, and r0
            Psi' is zero outside
            So the expectiaon value of psi' is the number of points outside
            its the number of points that didnt get weighted /total number of points

        """
        super(MEstimator, self).__init__(**kwargs)

    @property
    def r0(self):
        return self.iter_control.r0

    @property
    def u0(self):
        return self.iter_control.u0

    # @property
    # def beta(self):
    #     return rme_beta(self.r0)

    # def error_variances(self, Y_or_Yc, correction_factor=1.0):
    #     """
    #     These are the error variances.
    #     TODO: Move this method to the base class, or a QR decorator.
    #     Computes the squared norms difference of the output channels from the
    #     "output channels inner-product with QQH"
    #
    #     ToDo: Rename this to sigma_squared, or residual_variance rather than sigma.
    #     It is a variance.  Especially in the context or the redecnd using
    #     it's sqrt to normalize the residual amplitudes
    #
    #     Parameters
    #     ----------
    #     QHY : numpy array
    #         QHY[i,j] = Q.H * Y[i,j] = <Q[:,i], Y[:,j]>
    #         So when we sum columns of norm(QHY) we are get in the zeroth position
    #         <Q[:,0], Y[:,0]> +  <Q[:,1], Y[:,0]>, that is the 0th channel of Y
    #         projected onto each of the Q-basis vectors
    #     Y_or_Yc : numpy array
    #         The output channels (self.Y) or the cleaned output channels self.Yc
    #     correction_factor : float
    #         See doc in IterControl.correction_factor
    #
    #     Returns
    #     -------
    #     sigma : numpy array
    #         One entry per output channel.
    #
    #     """
    #     Y2 = np.linalg.norm(Y_or_Yc, axis=0) ** 2  # variance?
    #     QHY2 = np.linalg.norm(self.QHY, axis=0) ** 2
    #     sigma = correction_factor * (Y2 - QHY2) / self.n_data
    #
    #     try:
    #         assert (sigma > 0).all()
    #     except AssertionError:
    #         print("WARNING - Negative error variances observed")
    #         print(sigma)
    #         print("Setting sigma to zero - Negative sigma_squared observed")
    #         sigma *= 0
    #         # raise Exception
    #     return sigma

    # def apply_huber_weights(self, sigma, YP):
    #     """
    #     Updates the values of self.Yc and self.expectation_psi_prime
    #
    #     Parameters
    #     ----------
    #     sigma : numpy array
    #         1D array, the same length as the number of output channels
    #         see self.sigma() method for its calculation
    #     YP : numpy array
    #         The predicted data, usually from QQHY
    #
    #     Returns
    #     -------
    #
    #     function [YC,E_psiPrime] = HuberWt(Y,YP,sig,r0)
    #
    #     inputs are data (Y) and predicted (YP), estiamted
    #     error variances (for each column) and Huber parameter r0
    #     allows for multiple columns of data
    #     """
    #     # Y_cleaned = np.zeros(self.Y.shape, dtype=np.complex128)
    #     for k in range(self.n_channels_out):
    #         r0s = self.r0 * np.sqrt(sigma[k])
    #         residuals = np.abs(self.Y[:, k] - YP[:, k])
    #         w = np.minimum(r0s / residuals, 1.0)
    #         self.Yc[:, k] = w * self.Y[:, k] + (1 - w) * YP[:, k]
    #         self.expectation_psi_prime[k] = 1.0 * np.sum(w == 1) / self.n_data
    #     return

    # def update_predicted_data(self):
    #     pass
    #
    # def redescend(
    #     self,
    #     Y_predicted,
    #     sigma,
    # ):
    #     """
    #     % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    #     function[YC, E_psiPrime] = RedescendWt(Y, YP, sig, u0)
    #
    #     % inputs
    #     are
    #     data(Y) and predicted(YP), estiamted
    #     % error
    #     variances(
    #     for each column) and Huber parameter u0
    #     % allows
    #     for multiple columns of data
    #     """
    #     # Y_cleaned = np.zeros(self.Y.shape, dtype=np.complex128)
    #     for k in range(self.n_channels_out):
    #
    #         r = np.abs(self.Y[:, k] - Y_predicted[:, k]) / np.sqrt(sigma[k])
    #         t = -np.exp(self.u0 * (r - self.u0))
    #         w = np.exp(t)
    #
    #         # cleaned data
    #         self.Yc[:, k] = w * self.Y[:, k] + (1 - w) * Y_predicted[:, k]
    #
    #         # computation of E(psi')
    #         t = self.u0 * (t * r)
    #         t = w * (1 + t)
    #         self.expectation_psi_prime[k] = np.sum(t[t > 0]) / self.n_data
    #     return

    def estimate(self):
        """
        function that does the actual regression - M estimate
        Returns
        -------

        """
        print(
            "this method is not defined for RME - you must instantiate either a "
            "single station or remote reference RME estimator"
        )
        raise Exception

    # def compute_squared_coherence(self, Y_hat):
    #     """
    #     res: Residuals: The original data minus the predicted data.
    #     #SSR : Sum of squares of the residuals.  Diagonal is real
    #     This method could use some cleanup for readability
    #     see aurora issue #78.
    #     Parameters
    #     ----------
    #     YP
    #
    #     Returns
    #     -------
    #
    #     """
    #     res = self.Y - Y_hat
    #     SSR = np.conj(res.conj().T @ res)
    #     Yc2 = np.abs(self.Yc) ** 2
    #     SSYC = np.sum(Yc2, axis=0)
    #     R2 = 1 - np.diag(np.real(SSR)).T / SSYC
    #     R2[R2 < 0] = 0
    #
    #     self.R2 = xr.DataArray(
    #         R2,
    #         dims=[
    #             "output_channel",
    #         ],
    #         coords={
    #             "output_channel": list(self._Y.data_vars),
    #         },
    #     )
    #
    #     return