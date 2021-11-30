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

from aurora.transfer_function.regression.m_estimator import MEstimator


class TRME_RR(MEstimator):
    def __init__(self, **kwargs):
        """
        Robust remote reference estimator.  Z is the reference station data,
        and is the same size as X (see regression.base.RegressionEstimator(
        """
        super(TRME_RR, self).__init__(**kwargs)
        self._Z = kwargs.get("Z", None)
        self.Z = self._Z.to_array().data.T
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
        if self.is_underdetermined:
            error_msg = "not enough data for RR estimate:"
            error_msg = f"{error_msg} n_channels_in = {self.n_channels_in}"
            error_msg = f"{error_msg} N_data = {self.n_data}"
            print(f"{error_msg}")
            raise Exception

    def check_reference_data_shape(self):
        if self.Z.shape != self.X.shape:
            print("sizes of local and remote do not agree in RR estimation routine")
            raise Exception

    def initial_estimate(self):
        pass
    
    def update_y_hat(self, b):
        return self.X @ b
        

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
        # <INITIAL ESTIMATE>
        Q, R = self.qr_decomposition(self.Z)
        QHX = self.Q.conj().T @ self.X
        QHY = self.Q.conj().T @ self.Y
        # We need to get the properties of QHX, QXY to trace the flow of the
        # solution in matlab using mldivide; b0 = QTX\QTY;
        # also, below, what happens if we set self.b=np.linalg.solve(QHX, QHY)?
        #could do that in TRME and in TRME_RR
        b0 = np.linalg.solve(QHX, QHY)
        Y_hat = self.update_y_hat(b0)
        res = self.Y - Y_hat  # intial estimate of error variance
        residual_variance = np.sum(res * np.conj(res), axis=0) / self.n_data
        # </INITIAL ESTIMATE>
        
        if self.iter_control.max_number_of_iterations > 0:
            converged = False
        else:
            converged = True
            Y_hat = self.update_y_hat(b0)
            self.b = b0

        # <CONVERGENCE STUFF>
        self.iter_control.number_of_iterations = 0

        while not converged:
            self.iter_control.number_of_iterations += 1
            self.update_y_cleaned_via_huber_weights(residual_variance, Y_hat)
            # TRME_RR
            # updated error variance estimates, computed using cleaned data
            QHYc = self.QH @ self.Yc
            self.b = np.linalg.solve(QHX, QHYc)  # self.b = QTX\QTY
            Y_hat = self.update_y_hat(self.b)
            res = self.Yc - Y_hat
            squared_residuals = np.real(res * np.conj(res))
            mean_ssq_residuals = np.sum(squared_residuals, axis=0) / self.n_data
            residual_variance = self.correction_factor * mean_ssq_residuals
            converged = self.iter_control.converged(self.b, b0)
            b0 = self.b
        # </CONVERGENCE STUFF>

        # <REDESCENDING STUFF>
        if self.iter_control.max_number_of_redescending_iterations:
            self.iter_control.number_of_redescending_iterations = 0 #reset per channel
            while self.iter_control.continue_redescending:
                # one iteration with redescending influence curve cleaned data
                self.iter_control.number_of_redescending_iterations += 1
                self.update_y_cleaned_via_redescend_weights(Y_hat, residual_variance)
                # updated error variance estimates, computed using cleaned data
                self.update_QHYc()
#                QHYc = self.QH @ self.Yc
                self.b = np.linalg.solve(QHX, self.QHYc)  # QHX\QHYc
                Y_hat = self.update_y_hat(self.b)
                #res = self.Yc - Y_hat  # res_clean!

            # crude estimate of expectation of psi accounts for redescending influence curve
            self.expectation_psi_prime = 2 * self.expectation_psi_prime - 1
        # </REDESCENDING STUFF>

        # <Covariance and Coherence>
        self.compute_inverse_signal_covariance()

        # Below is a comment from the matlab codes:
        # "need to look at how we should compute adjusted residual cov to make
        # consistent with tranmt"
        self.compute_noise_covariance(Y_hat)
        self.compute_squared_coherence(Y_hat)

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

