"""
An extension of RegressionEstimator with properties common to both TRME and TRME_RR
"""
import numpy as np
import xarray as xr

from copy import deepcopy
from scipy.linalg import solve_triangular

from aurora.transfer_function.regression.base import RegressionEstimator
from aurora.transfer_function.regression.helper_functions import rme_beta


class MEstimator(RegressionEstimator):
    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        kwargs
        expectation_psi_prime : numpy array
            Same number of entries as there are output channels (columns of Y)
            Default is a vector of 1.0's

            Expectation value of psi' (derivative of psi) -- ?? OR rho'=psi????
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
        self.expectation_psi_prime = np.ones(self.n_channels_out)
        self.Yc = deepcopy(self.Y)
        self._Y_hat = None
        self._residual_variance = None

    @property
    def QHYc(self):
        if self._QHYc is None:
            self.update_QHYc()
        return self._QHYc

    def update_QHYc(self):
        self._QHYc = self.QH @ self.Yc

    @property
    def Y_hat(self):
        if self._Y_hat is None:
            self.update_y_hat()
        return self._Y_hat

    def update_y_hat(self):
        print("Y_hat update method is not defined for abstract MEstimator class")
        print("Try using RME or RME_RR class instead")
        raise Exception

    def update_residual_variance(self, correction_factor=1):
        print("update_residual_variance method not defined in abstract MEstimator")
        print("Try using RME or RME_RR class instead")
        raise Exception

    @property
    def residual_variance(self):
        if self._residual_variance is None:
            self.update_residual_variance()
        return self._residual_variance


    @property
    def r0(self):
        return self.iter_control.r0

    @property
    def u0(self):
        return self.iter_control.u0

    # @property
    # def beta(self):
    #     return rme_beta(self.r0)

    @property
    def correction_factor(self):
        """
        *May want to move this out of iter_control, it is really only controlled by r0.

        Returns
        -------
        correction_factor: float
            See doc in iter_control.IterControl.correction_factor()
        """
        return self.iter_control.correction_factor

    def residual_variance_method1(self):
        """
        This is the method that was originally in TRME_RR.m.  It seems more correct 
        than the one in TRME, but also has more computational overhead.
        """
        res = self.Yc - self.Y_hat  # intial estimate of error variance
        residual_variance = np.sum(np.abs(res * np.conj(res)), axis=0) / self.n_data
        return residual_variance
    
    def residual_variance_method2(self):
        """
        These are the error variances.

        Computes the squared norms difference of the output channels from the
        "output channels inner-product with QQH"

        This method used to take either QHY, Y as arguments or QHYc, Yc
        But now that Yc initializes to Y, we can just use with QHYc, Yc always
        (Note QHYc is updated everytime Yc is updated)

        Parameters
        ----------
        QHY : numpy array
            QHY[i,j] = Q.H * Y[i,j] = <Q[:,i], Y[:,j]>
            So when we sum columns of norm(QHY) we are get in the zeroth position
            <Q[:,0], Y[:,0]> +  <Q[:,1], Y[:,0]>, that is the 0th channel of Y
            projected onto each of the Q-basis vectors
        Y_or_Yc : numpy array
            The output channels (self.Y) or the cleaned output channels self.Yc
        correction_factor : float
            See doc in IterControl.correction_factor

        Returns
        -------
        residual_variance : numpy array
            One entry per output channel.

        """
        Y2 = np.linalg.norm(self.Yc, axis=0) ** 2  # variance?
        QHY2 = np.linalg.norm(self.QHYc, axis=0) ** 2
        residual_variance = (Y2 - QHY2) / self.n_data

        try:
            assert (residual_variance > 0).all()
        except AssertionError:
            print("WARNING - Negative error variances observed")
            print(residual_variance)
            print("Setting residual_variance to zero - Negative values observed")
            residual_variance *= 0

        return residual_variance

    

    def update_y_cleaned_via_huber_weights(self):
        """
        Updates the values of self.Yc and self.expectation_psi_prime

        Parameters
        ----------
        residual_variance : numpy array
            1D array, the same length as the number of output channels
            see self.residual_variance() method for its calculation
        Y_hat : numpy array
            The predicted data, usually from QQHY

        Returns
        -------
        None

        Original matlab documenation:
        function [YC,E_psiPrime] = HuberWt(Y,YP,sig,r0)
        inputs are data (Y) and predicted (YP), estiamted
        error variances (for each column) and Huber parameter r0
        allows for multiple columns of data
        """
        for k in range(self.n_channels_out):
            r0s = self.r0 * np.sqrt(self.residual_variance[k])
            residuals = np.abs(self.Y[:, k] - self.Y_hat[:, k])
            w = np.minimum(r0s / residuals, 1.0)
            self.Yc[:, k] = w * self.Y[:, k] + (1 - w) * self.Y_hat[:, k]
            self.expectation_psi_prime[k] = 1.0 * np.sum(w == 1) / self.n_data
        self.update_QHYc() #note the QH is different in TRME_RR vs TRME
        return


    # def update_predicted_data(self):
    #     pass
    #
    def update_y_cleaned_via_redescend_weights(self):
        """
        Updates estimate for self.Yc as a match-filtered sum of Y and Y_hat.
        
        
        Parameters
        ----------
        Y_hat: numpy array
            An estimate of the output data Y obtained by
            self.Q @ self.QHY
            or
            self.Q @ self.QHYc
        residual_variance

        Returns
        -------

        Matlab documentation:
        function[YC, E_psiPrime] = RedescendWt(Y, YP, sig, u0):
        inputs are data(Y) and predicted(YP), estimated error variances (for each
        column) and Huber parameter u0.  Allows for multiple columns of data
        """
        # Y_cleaned = np.zeros(self.Y.shape, dtype=np.complex128)
        for k in range(self.n_channels_out):
            sigma = np.sqrt(self.residual_variance[k])
            r = np.abs(self.Y[:, k] - self.Y_hat[:, k]) / sigma
            t = -np.exp(self.u0 * (r - self.u0))
            w = np.exp(t)

            # cleaned data
            self.Yc[:, k] = w * self.Y[:, k] + (1 - w) * self.Y_hat[:, k]

            # computation of E(psi')
            t = self.u0 * (t * r)
            t = w * (1 + t)
            self.expectation_psi_prime[k] = np.sum(t[t > 0]) / self.n_data
        self.update_QHYc()
        return

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

    def compute_squared_coherence(self):
        """
        res: Residuals: The original data minus the predicted data.
        SSR : Sum of squares of the residuals.  Diagonal is real
        This method could use some cleanup for readability
        see aurora issue #78.
        Parameters
        ----------
        Y_hat

        Returns
        -------

        """
        res = self.Y - self.Y_hat
        SSR = np.conj(res.conj().T @ res)
        Yc2 = np.abs(self.Yc) ** 2
        SSYC = np.sum(Yc2, axis=0)
        R2 = 1 - np.diag(np.real(SSR)).T / SSYC
        R2[R2 < 0] = 0

        self.R2 = xr.DataArray(
            R2,
            dims=[
                "output_channel",
            ],
            coords={
                "output_channel": list(self._Y.data_vars),
            },
        )

        return


    def compute_noise_covariance(self):
        """
        res_clean: The cleaned data minus the predicted data. The residuals
        SSR_clean: Sum of squares of the residuals.  Diagonal is real
        Parameters
        ----------
        Y_hat

        Returns
        -------

        """
        res_clean = self.Yc - self.Y_hat
        SSR_clean = np.conj(res_clean.conj().T @ res_clean)
        inv_psi_prime2 = np.diag(1.0 / (self.expectation_psi_prime ** 2))
        cov_nn = inv_psi_prime2 @ SSR_clean / self.degrees_of_freedom

        self.cov_nn = xr.DataArray(
            cov_nn,
            dims=["output_channel_1", "output_channel_2"],
            coords={
                "output_channel_1": list(self._Y.data_vars),
                "output_channel_2": list(self._Y.data_vars),
            },
        )
        return