"""
This module contains the MEstimator class - an extension of RegressionEstimator.

 MEstimator has the class methods that are common to both RME and RME_RR.
 See Notes in RME, RME_RR for more details.

"""
import numpy as np
import xarray as xr

from aurora.transfer_function.regression.base import RegressionEstimator
from copy import deepcopy
from loguru import logger


class MEstimator(RegressionEstimator):
    def __init__(self, **kwargs):
        """
        Constructor.

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
            So the expectation value of psi' is the number of points outside, (that didn't get weighted) divided by the total number of points

        """
        super(MEstimator, self).__init__(**kwargs)
        self.expectation_psi_prime = np.ones(self.n_channels_out)
        self.Yc = deepcopy(self.Y)
        self._Y_hat = None
        self._residual_variance = None

    @property
    def QHYc(self) -> np.ndarray:
        """
        Returns the matrix QH @ Yc
        This is Q.conj().T from the QR decomposition multiplied with the cleaned data.


        Returns
        -------
        QHYc: np.ndarray
            A convenience matrix that makes computing the predicted data (QQHYc) more efficient.
        """
        if self._QHYc is None:
            self.update_QHYc()
        return self._QHYc

    def update_QHYc(self) -> np.ndarray:
        """Updates the QHYc matrix with the most recent cleaned data"""
        self._QHYc = self.QH @ self.Yc

    @property
    def Y_hat(self) -> np.ndarray:
        """returns the most recent predicted data"""
        if self._Y_hat is None:
            self.update_y_hat()
        return self._Y_hat

    def update_y_hat(self):
        """updates the predicted data from the most recent cleaned data"""
        logger.error("Y_hat update method is not defined for abstract MEstimator class")
        logger.error("Try using RME or RME_RR class instead")
        raise Exception

    def update_residual_variance(self, correction_factor=1):
        """updates residual variance from most recent cleaned and predicted data"""
        logger.error(
            "update_residual_variance method not defined in abstract MEstimator"
        )
        logger.error("Try using RME or RME_RR class instead")
        raise Exception

    @property
    def residual_variance(self) -> np.ndarray:
        """returns the residual variance"""
        if self._residual_variance is None:
            self.update_residual_variance()
        return self._residual_variance

    @property
    def r0(self) -> float:
        """returns the Huber r0 value"""
        return self.iter_control.r0

    @property
    def u0(self) -> float:
        """returns the u0 threshold for redescending regression step"""
        return self.iter_control.u0

    @property
    def correction_factor(self) -> float:
        """
        Return teh correction factor for the residual variance.

        Returns
        -------
        correction_factor: float
            See doc in iter_control.IterControl.correction_factor()
        """
        return self.iter_control.correction_factor

    def residual_variance_method1(self) -> np.ndarray:
        """
        returns the residual variance of the output channels.

        This is the method that was originally in RME_RR.m.  It seems more correct
        than the one in RME, but also has more computational overhead.
        """
        res = self.Yc - self.Y_hat  # initial estimate of error variance
        residual_variance = np.sum(np.abs(res * np.conj(res)), axis=0) / self.n_data
        return residual_variance

    def residual_variance_method2(self) -> np.ndarray:
        """
        Returns the residual variance of the output channels (error variances).

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
        QHY2 = (
            np.linalg.norm(self.QHYc, axis=0) ** 2
        )  # note this is the same as QQHYc=Y_hat
        residual_variance = (Y2 - QHY2) / self.n_data

        try:
            assert (residual_variance > 0).all()
        except AssertionError:
            logger.warning("WARNING - Negative error variances observed")
            logger.warning(residual_variance)
            logger.warning(
                "Setting residual_variance to zero - Negative values observed"
            )
            residual_variance *= 0

        return residual_variance

    def update_y_cleaned_via_huber_weights(self) -> None:
        """
        Updates the values of self.Yc and self.expectation_psi_prime

        Parameters
        ----------
        residual_variance : numpy array
            1D array, the same length as the number of output channels
            see self.residual_variance() method for its calculation
        Y_hat : numpy array
            The predicted data, usually from QQHY


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
        self.update_QHYc()  # note the QH is different in RME_RR vs RME
        return

    def initial_estimate(self) -> None:
        """
        Make first estimate of TF (b), Y_hat, and residual_variance
        """
        self.qr_decomposition()
        self.update_b()
        self.update_y_hat()
        self.update_residual_variance()

    def apply_huber_regression(self) -> None:
        """
            This is the 'convergence loop' from RME, RME_RR

        TODO: Consider not setting iter_control.number_of_iterations
         - Instead, Initialize a new iter_control object
        """
        converged = self.iter_control.max_iterations_reached
        if self.iter_control.number_of_iterations:
            self.iter_control.reset_number_of_iterations()
        while not converged:
            b0 = self.b
            self.iter_control.increment_iteration_number()
            self.update_y_cleaned_via_huber_weights()
            self.update_b()
            self.update_y_hat()
            self.update_residual_variance(correction_factor=self.correction_factor)
            converged = self.iter_control.converged(self.b, b0)
        return

    def apply_redecending_influence_function(self) -> None:
        """
        Performs one or two iterations with re-descending influence curve cleaned data
        """
        if self.iter_control.max_number_of_redescending_iterations:
            self.iter_control.reset_number_of_redescending_iterations()  # reset per channel
            while self.iter_control.continue_redescending:
                self.iter_control.increment_redescending_iteration_number()
                self.update_y_cleaned_via_redescend_weights()
                self.update_b()
                self.update_y_hat()
                self.update_residual_variance()
            # crude estimate of expectation of psi
            # accounts for redescending influence curve
            self.expectation_psi_prime = 2 * self.expectation_psi_prime - 1
        return

    def estimate(self) -> None:
        """
        Executes the regression

        Development Notes:
        Here is a comment from the matlab codes:
        "need to look at how we should compute adjusted residual cov to make
         consistent with tranmt"
        See issue#69 aurora github repo addresses this
        """
        if self.is_underdetermined:
            self.solve_underdetermined()
            return

        self.initial_estimate()
        self.apply_huber_regression()
        self.apply_redecending_influence_function()

        if self.iter_control.return_covariance:
            self.compute_inverse_signal_covariance()
            self.compute_noise_covariance()
            self.compute_squared_coherence()
        return

    def update_y_cleaned_via_redescend_weights(self) -> None:
        """
        Updates estimate for self.Yc as a match-filtered sum of Y and Y_hat.

        Note: It is not unheard of to observe RuntimeWarning: overflow encountered in exp in the calculation of t.
        This can happen when large residuals are present.  In that case, t goes to -inf, and w goes to zero,
        -- the desired behaviour.  When this happens an "invalid value" will also occur  in the calculation
        of t, but this does not propagate into self.expectation_psi_prime.

        Matlab documentation:
        function[YC, E_psiPrime] = RedescendWt(Y, YP, sig, u0):
        inputs are data(Y) and predicted(YP), estimated error variances (for each
        column) and Huber parameter u0.  Allows for multiple columns of data

        Parameters
        ----------
        Y_hat: numpy array
            An estimate of the output data Y obtained by
            self.Q @ self.QHY
            or
            self.Q @ self.QHYc
        residual_variance

        """
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

    def compute_squared_coherence(self) -> None:
        """
        Updates the array self.R2

        Here is taken the ratio of the energy in the residuals with the energy in the cleaned data.
        This metric can be interpreted as how much of the signal (Y) is "explained" by the regression.

        Development Notes:
        The matlab code (TRME_RR) claimed:
            %  R2 is squared coherence (top row is using raw data, bottom
            %    cleaned, with crude correction for amount of down-weighted data)

        TODO: There seem to be other valid metrics for this sort of quantity.  In particular, we may want to
         consider SSY (the sum of squares of the observed data) over SSR.

        TODO: consider renaming self.R2.  That name invokes the idea of the squared residuals.  That is not what
         is being stored in self.R2.  This is more like a CMRR.

        res: Residuals, the original data minus the predicted data.
        SSR : Sum of squares of the residuals, per channel

        """
        res = self.Y - self.Y_hat
        SSR = np.einsum(
            "ij,ji->i", res.conj().T, res
        )  # takes the diagonal of the matrix product
        SSR = np.real(SSR).T
        Yc2 = np.abs(self.Yc) ** 2
        SSYC = np.sum(Yc2, axis=0)
        R2 = 1 - SSR / SSYC
        R2[R2 < 0] = 0

        self.R2 = xr.DataArray(
            R2,
            dims=[
                "output_channel",
            ],
            coords={
                "output_channel": self.output_channel_names,
            },
        )

        if self.iter_control.verbosity > 1:
            msg = f"squared coherence {list(self.R2.coords['output_channel'].values)}  {R2}"
            logger.info(msg)
        return

    def compute_noise_covariance(self) -> None:
        """
        Computes the noise covariance (covariance of the residuals)

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
        inv_psi_prime2 = np.diag(1.0 / (self.expectation_psi_prime**2))
        cov_nn = inv_psi_prime2 @ SSR_clean / self.degrees_of_freedom

        self.cov_nn = xr.DataArray(
            cov_nn,
            dims=["output_channel_1", "output_channel_2"],
            coords={
                "output_channel_1": self.output_channel_names,
                "output_channel_2": self.output_channel_names,
            },
        )
        return
