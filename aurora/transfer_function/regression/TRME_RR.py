"""
follows Gary's TRME.m in
iris_mt_scratch/egbert_codes-20210121T193218Z-001/egbert_codes/matlabPrototype_10-13-20/TF/classes

    % 2009 Gary Egbert , Maxim Smirnov
    % Oregon State University

    %  (Complex) Robust remote reference estimator, for arbitray number of
    %     input channels (i.e., columns of design matrix).
    %    X gives the local input, Z is the reference
    %       (matrices of the same size)
    %  As for TRME the model  is Y = X*b, and multiple columns
    %    of Y (predicted or output variables) are allowed (estimates
    %    of b for each column are computed separately)
    %  Allows multiple columns of Y, but estimates b for each column separately
    %    no missing data is allowed for the basic RR class
    %
    %   S and N are estimated signal and noise covariance
    %    matrices, which together can be used to compute
    %    error covariance for the matrix of regression coefficients b
    %  R2 is squared coherence (top row is using raw data, bottom
    %    cleaned, with crude correction for amount of downweighted data)

    %  Parameters that control regression M-estimates are defined in ITER

"""
import numpy as np
import xarray as xr

from aurora.transfer_function.regression.base import RegressionEstimator


class TRME_RR(RegressionEstimator):
    def __init__(self, **kwargs):
        """
        %   class constructor for TRME_RR objects
        %    Robust remote reference estimator, for arbitray number of
        %     input channels (i.e., columns of design matrix).
        %    X gives the local input, Z is the reference
        %       (matrices of the same size)
        %  As for regrM.m the model  is Y = X*b, and multiple columns
        %    of Y (predicted or output variables) are allowed (estimates
        %    of b for each column are computed separately)
        %
        %   Usage: obj = TRME_RR(X,Z,Y,iter);

            needs X, Y, Z, iter
            :param kwargs:
        """
        super(TRME_RR, self).__init__(**kwargs)
        self._Z = kwargs.get("Z", None)
        self.Z = self._Z.to_array().data.T
        self.expectation_psi_prime = np.ones(self.n_channels_out)
        self.sigma_squared = np.zeros(self.n_channels_out)
        self.check_for_nan()
        self.check_number_of_observations_xy_consistent()
        self.check_for_enough_data_for_rr_estimate()
        self.check_reference_data_shape()

    def check_for_nan(self):
        cond1 = np.isnan(self.X).any()
        cond2 = np.isnan(self.Y).any()
        cond3 = np.isnan(self.Z).any()
        nans_present = cond1 or cond2 or cond3
        if nans_present:
            print("Missing data not allowed for TRME_RR class")
            raise Exception

    def check_for_enough_data_for_rr_estimate(self):
        if self.n_param > self.n_data:
            error_msg = "not enough data for RR estimate:"
            print(f"{error_msg} # param = {self.n_param} # data = {self.n_data}")
            raise Exception

    def check_reference_data_shape(self):
        if self.Z.shape != self.X.shape:
            print("sizes of local and remote do not agree in RR estimation routine")
            raise Exception

    # <COMMON RME METHODS>
    @property
    def r0(self):
        return self.iter_control.r0

    def apply_huber_weights(self, sigma, YP):
        """

        Parameters
        ----------
        sigma : numpy array
            1D array, the same length as the number of output channels
            see self.sigma() method for its calculation
        YP : numpy array
            The predicted data, usually from QQHY

        Returns
        -------
        Updates the values of self.Yc and self.expectation_psi_prime

        """
        """
        function [YC,E_psiPrime] = HuberWt(Y,YP,sig,r0)

        inputs are data (Y) and predicted (YP), estiamted
        error variances (for each column) and Huber parameter r0
        allows for multiple columns of data


        """
        for k in range(self.n_channels_out):
            r0s = self.r0 * np.sqrt(sigma[k])
            residuals = np.abs(self.Y[:, k] - YP[:, k])
            w = np.minimum(r0s / residuals, 1.0)
            self.Yc[:, k] = w * self.Y[:, k] + (1 - w) * YP[:, k]
            self.expectation_psi_prime[k] = 1.0 * np.sum(w == 1) / self.n_data
        return

    # </COMMON RME METHODS>

    def estimate(self):
        """
        %   function that does the actual remote reference estimate
        %
        %   Usage:  [b]  = Estimate(obj);
        %    (Object has all outputs; estimate of coefficients is also returned
        %              as function output)


        %   note that ITER is a handle object, so mods to ITER properties are
        %   already made also to obj.ITER!

        :return:
        """
        Q, R = self.qr_decomposition(self.Z)
        # initial LS RR estimate b0, error variances sigma
        # [Q, ~] = qr(obj.Z, 0);#0 means "economy" -- same as np.lnalg default
        QHX = self.Q.conj().T @ self.X
        QHY = self.Q.conj().T @ self.Y
        # We need to get the properties of QHX, QXY to trace the flow of the
        # solution in matlab using mldivide
        b0 = np.linalg.solve(QHX, QHY)  # b0 = QTX\QTY;
        # predicted data
        Yhat = self.X @ b0
        # intial estimate of error variance
        res = self.Y - Yhat
        sigma = np.sum(res * np.conj(res), axis=0) / self.n_data
        cfac = self.iter_control.correction_factor
        if self.iter_control.max_number_of_iterations > 0:
            converged = False
            cfac = 1.0 / (1.0 - np.exp(-self.iter_control.r0))

        else:
            converged = True
            # not needed - its defined in the init
            # self.expectation_psi_prime = np.ones(self.n_channels_out)
            Yhat = self.X @ b0
            self.b = b0
            self.Yc = self.Y

        # <CONVERGENCE STUFF>
        self.iter_control.number_of_iterations = 0

        while not converged:
            self.iter_control.number_of_iterations += 1
            # cleaned data
            self.apply_huber_weights(sigma, Yhat)
            # TRME_RR
            # updated error variance estimates, computed using cleaned data
            QHY = Q.conj().T @ self.Yc
            self.b = np.linalg.solve(QHX, QHY)  # self.b = QTX\QTY
            Yhat = self.X @ self.b
            res = self.Yc - Yhat
            sigma = cfac * np.sum(res * np.conj(res), axis=0) / self.n_data
            converged = self.iter_control.converged(self.b, b0)
            b0 = self.b
        # </CONVERGENCE STUFF>

        # <REDESCENDING STUFF>
        while self.iter_control.continue_redescending:
            # one iteration with redescending influence curve cleaned data
            self.iter_control.number_of_redescending_iterations += 1
            # [obj.Yc, E_psiPrime] = RedescendWt(obj.Y, Yhat, sigma, ITER.u0) # #TRME_RR
            self.redescend(Yhat, sigma)  # update cleaned data, and expectation
            # updated error variance estimates, computed using cleaned data
            QHYc = self.QH @ self.Yc
            self.b = np.linalg.solve(QHX, QHYc)  # QHX\QHYc
            Yhat = self.X * self.b
            res = self.Yc - Yhat  # res_clean!

        # crude estimate of expectation of psi accounts for redescending influence curve
        self.expectation_psi_prime = 2 * self.expectation_psi_prime - 1
        # </REDESCENDING STUFF>

        # <Covariance and Coherence>
        self.compute_inverse_signal_covariance()

        # Below is a comment from the matlab codes:
        # "need to look at how we should compute adjusted residual cov to make
        # consistent with tranmt"
        self.compute_noise_covariance(Yhat)
        self.compute_squared_coherence(Yhat)

        # </Covariance and Coherence>

    def compute_inverse_signal_covariance(self):
        """
        Matlab code was :
        Cov_SS = (self.ZH @self.X) \ (self.XH @ self.X) / (self.XH @self.Z)
        I broke the above line into B/A where
        B = (self.ZH @self.X) \ (self.XH @ self.X), and A = (self.XH @self.Z)
        Then follow matlab cookbok, B/A for matrices = (A'\B')
        """
        ZH = self.Z.conj().T
        XH = self.X.conj().T
        B = np.linalg.solve(ZH @ self.X, XH @ self.X)
        A = XH @ self.Z
        cov_ss_inv = np.linalg.solve(A.conj().T, B.conj().T).conj().T

        self.cov_ss_inv = xr.DataArray(
            cov_ss_inv,
            dims=["input_channel_1", "input_channel_2"],
            coords={
                "input_channel_1": list(self._X.data_vars),
                "input_channel_2": list(self._X.data_vars),
            },
        )
        return

    def compute_noise_covariance(self, Yhat):
        """
        res_clean: The cleaned data minus the predicted data. The residuals
        SSR_clean: Sum of squares of the residuals.  Diagonal is real
        Parameters
        ----------
        YP

        Returns
        -------

        """
        res_clean = self.Yc - Yhat
        SSR_clean = np.conj(res_clean.conj().T @ res_clean)
        degrees_of_freedom = self.n_data - self.n_param
        inv_psi_prime2 = np.diag(1.0 / (self.expectation_psi_prime ** 2))
        cov_nn = inv_psi_prime2 @ SSR_clean / degrees_of_freedom

        self.cov_nn = xr.DataArray(
            cov_nn,
            dims=["output_channel_1", "output_channel_2"],
            coords={
                "output_channel_1": list(self._Y.data_vars),
                "output_channel_2": list(self._Y.data_vars),
            },
        )
        return

    def compute_squared_coherence(self, Y_hat):
        """
        TODO: Compare this method with compute_squared_coherence in TRME.  I think
        they are identical, in which case we can merge them, and maybe even put into
        the regression base class.

        TODO: Also, RegressionEstimator may be better as a more abstract base class and
        we can have a QRRegressionEstimator() class between base and {TRME, TRME_RR}

        res: Residuals: The original data minus the predicted data.
        SSR : Sum of squares of the residuals.  Diagonal is real
        Parameters
        ----------
        YP

        Returns
        -------

        """
        res = self.Y - Y_hat
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
