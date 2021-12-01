"""
follows Gary's TRME.m in
iris_mt_scratch/egbert_codes-20210121T193218Z-001/egbert_codes/matlabPrototype_10-13-20/TF/classes

% 2009 Gary Egbert , Maxim Smirnov
% Oregon State University

%
%  (Complex) regression-M estimate for the model  Y = X*b
%
%  Allows multiple columns of Y, but estimates b for each column separately
%
%   S and N are estimated signal and noise covariance
%    matrices, which together can be used to compute
%    error covariance for the matrix of regression coefficients b
%  R2 is squared coherence (top row is using raw data, bottom
%    cleaned, with crude correction for amount of downweighted data)

%  Parameters that control regression M-estimates are defined in ITER

TODO: Consider making a QR-Estimator class between RegressionEstimator and RMEEstimator

# <QR-decomposition Notes>
The QR-decomposition is employed on the matrix of input variables X.
X = Q R where Q is unitary/orthogonal and R upper triangular.
Since X is [n_data x n_channels_in] Q is [n_data x n_data].

Wikipedia has a nice description of the QR factorization:
https://en.wikipedia.org/wiki/QR_decomposition
On a high level, the point of the QR decomposition is to transform the data
into a domain where the inversion is done with a triangular matrix.

The symbol QH will denote the conjugate transpose of the matrix Q.

We employ here the "economical" form of the QR decompostion, so that Q is not square,
and not in fact unitary.  This is because its inverse is not defined (as it isn't
square). Q does however obey: Q.H @ Q = I.

Really Q = [Q1 | Q2] where Q1 has as many columns as there are input variables
and Q2 is a matrix of zeros.

QQH is the projection matrix, or hat matrix equivalent to X(XHX)^-1XH.
https://math.stackexchange.com/questions/3360485/projection-matrix-expressed-as-qr-identity

The quantity QHY floating around is a convenience matrix that makes computing the
predicted data less numerically expensive.  The use of QHY is not so much physically
meaningful as it is a trick to compute more efficiently the predicted data QQHY.

The predicted data Y_hat = QQHY (since QQH is proj mtx)
but, when computing the sums of squares of Y_hat, such as we do in the error
variances calculation, we can compute  QHY2 = np.linalg.norm(QHY, axis=0) ** 2
instead of QQHY2 = Y_hat2 = np.linalg.norm(Y_hat, axis=0) ** 2 since
Y_hat.H @ Y_hat = QQHY.H @ QQHY = QHY.H @ Q.H @ Q @ QHY = QHY.H @ QHY.

The predicted data has to lie in span of the columns in the design matrix X.
The predicted data has to be a linear combination of the columns of Y.
Q is an orthoganal basis for the columns of X.
The norms of QQHY and QHY are the same

    < MATLAB Documentation >
    https://www.mathworks.com/help/matlab/ref/qr.html

Matlab's reference to the "economy" rerpresentation is what Trefethen and Bau
call the "reduced QR factorization".  Golub & Van Loan (1996, §5.2) call Q1R1
the thin QR factorization of A;
    < /MATLAB Documentation >

There are several discussions online about the differences in
numpy, scipy, sklearn, skcuda etc.
https://mail.python.org/pipermail/numpy-discussion/2012-November/064485.html
We will default to using numpy for now.
Note that numpy's default is to use the "reduced" form of Q, R.  R is
upper-right triangular.
# </QR-decomposition Notes>


This is cute:
https://stackoverflow.com/questions/26932461/conjugate-transpose-operator-h-in-numpy

The Matlab mldivide flowchart can be found here:
https://stackoverflow.com/questions/18553210/how-to-implement-matlabs-mldivide-a-k-a-the-backslash-operator
And the matlab documentation here
http://matlab.izmiran.ru/help/techdoc/ref/mldivide.html
"""
import numpy as np
import xarray as xr

from scipy.linalg import solve_triangular

from aurora.transfer_function.regression.m_estimator import MEstimator


class TRME(MEstimator):
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
        super(TRME, self).__init__(**kwargs)
        self.qr_input = "X"


    def initial_least_squares_estimate(self):
        pass
    
    def update_y_hat(self):
        """?rename as update_predicted_data?"""
        self._Y_hat = self.Q @ self.QHYc
        return

    def update_residual_variance(self, correction_factor=1):
        self._residual_variance = self.residual_variance_method2()
        self._residual_variance *= correction_factor
        return self._residual_variance

    def update_b(self):
        """
        matlab was: b = R\QTY;
        Returns
        -------

        """
        self.b = solve_triangular(self.R, self.QHYc)

        
    def estimate(self):
        """
        function that does the actual regression - M estimate

        Usage: [b] = Estimate(obj);
        (Object has all outputs; estimate of coefficients is also returned
        as function output)

        # note that ITER is a handle object, so mods to ITER properties are
        # already made also to obj.ITER!
        Returns
        -------

        """
        if self.is_underdetermined:
            self.b = self.solve_underdetermined()
            return

        self.initial_estimate()

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
                self.iter_control.number_of_redescending_iterations += 1
                #self.update_y_hat()
                self.update_y_cleaned_via_redescend_weights()
                self.update_y_hat()
                self.update_b()
                self.update_residual_variance()
            # crude estimate of expectation of psi accounts for redescending influence curve
            self.expectation_psi_prime = 2 * self.expectation_psi_prime - 1
        # </REDESCENDING STUFF>

        if self.iter_control.return_covariance:
            self.compute_inverse_signal_covariance()
            self.compute_noise_covariance()
            self.compute_squared_coherence()

        return self.b


    def compute_inverse_signal_covariance(self):
        """
        Note that because X = QR, we have
        X.H @ X = (QR).H @ QR = R.H Q.H Q R = R.H @ R
        i.e. computing R.H @ R below is just computing the signal covariance matrix of X
        Returns
        -------

        """
        cov_ss_inv = np.linalg.inv(self.R.conj().T @ self.R)

        self.cov_ss_inv = xr.DataArray(
            cov_ss_inv,
            dims=["input_channel_1", "input_channel_2"],
            coords={
                "input_channel_1": list(self._X.data_vars),
                "input_channel_2": list(self._X.data_vars),
            },
        )
        return
