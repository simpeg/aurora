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
        self.qr_input = "Z"
        self._QHX = None
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
        """
        Make first estimate of TF (b), Y_hat, and residual_variance
        """
        pass
    
    def update_y_hat(self):
        self._Y_hat = self.X @ self.b 
        return

    def update_b(self):
        """matlab was: b = QTX\QTY"""
        self.b = np.linalg.solve(self.QHX, self.QHYc)
        return

    @property
    def QHX(self):
        if self._QHX is None:
            self._QHX = self.QH @ self.X
        return self._QHX


    def update_residual_variance(self, correction_factor=1):
        self._residual_variance = self.residual_variance_method1()
        self._residual_variance *= correction_factor
        return self._residual_variance

    def estimate(self):
        """
        function that does the actual remote reference estimate
        """
        #self.initial_estimate()
        self.qr_decomposition()
        self.update_b()
        self.update_y_hat()
        self.update_residual_variance()

        # <CONVERGENCE STUFF>
        converged = self.iter_control.max_number_of_iterations <= 0
        self.iter_control.number_of_iterations = 0
        while not converged:
            b0 = self.b
            self.iter_control.number_of_iterations += 1
            self.update_y_cleaned_via_huber_weights()
            self.update_b()
            self.update_y_hat()
            self.update_residual_variance(correction_factor=self.correction_factor)
            converged = self.iter_control.converged(self.b, b0)
        # </CONVERGENCE STUFF>

        # <REDESCENDING STUFF>
        if self.iter_control.max_number_of_redescending_iterations:
            self.iter_control.number_of_redescending_iterations = 0 #reset per channel
            while self.iter_control.continue_redescending:
                # one iteration with redescending influence curve cleaned data
                self.iter_control.number_of_redescending_iterations += 1
                self.update_y_cleaned_via_redescend_weights()
                self.update_b()
                self.update_y_hat()
                self.update_residual_variance()
            # crude estimate of expectation of psi accounts for redescending influence curve
            self.expectation_psi_prime = 2 * self.expectation_psi_prime - 1
        # </REDESCENDING STUFF>

        # <Covariance and Coherence>
        self.compute_inverse_signal_covariance()
        # Below is a comment from the matlab codes:
        # "need to look at how we should compute adjusted residual cov to make
        # consistent with tranmt"
        self.compute_noise_covariance()
        self.compute_squared_coherence()
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

