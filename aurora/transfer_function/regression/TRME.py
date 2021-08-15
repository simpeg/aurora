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



TODO: set Q, R to be variables asssociated with this class (actually put Q, R inside
RegressionEstimator())
TODO: Move TRME to RME-Regression-Estimator
TODO: Consider making a QR-Estimator class between RegressionEstimator and RMEEstimator

# <QR-decomposition Notes>
The QR-decomposition is employed on the matrix of independent variables.
X = Q R where Q is unitary/orthogonal and R upper triangular.
Since X is [n_data x n_channels_in] Q is [n_data x n_data].  Wikipedia has a
nice description of the QR factorization:
https://en.wikipedia.org/wiki/QR_decomposition
On a high level, the point of the QR decomposition is to transform the data
into a domain where the inversion is done with a triangular matrix.

I use the symbol QH to denote the conjugate transpose.

Note that we employ here the "economical" form of the QR decompostion,
so that Q is not square, and not in fact unitary.  This is because its inverse is not
defined (as it isn't square). Q does however obey: Q.H @ Q = I.

Really Q = [Q1 | Q2] where Q1 has as many columns as there are input variables
and Q2 is a matrix of zeros.

QQH is the projection matrix, or hat matrix equivalent to X(XHX)^-1XH.
https://math.stackexchange.com/questions/3360485/projection-matrix-expressed-as-qr-identity

The quantity QHY floating around is a convenience matrix that makes computing the
predicted data less numerically expensive.  The use of QHY is not so much physically
meaningful as it is a trick to compute more efficiently the predicted data QQHY.

Y_predicted = QQHY (since QQH is proj mtx)
but, when computing the sums of squares of Y_predicted, such as we do in the error
variances calculation, we can compute  QHY2 = np.linalg.norm(QHY, axis=0) ** 2
instead of QQHY2 = YP2 = np.linalg.norm(YP, axis=0) ** 2 since
YP.H @ YP = QQHY.H @ QQHY = QHY.H @ Q.H @ Q @ QHY = QHY.H @ QHY.
The predicted data has to lie in span of the columns in the design matrix X.
The predicted data has to be a linear combination of the columns of Y.
Q is an orthoganal basis for the columns of X.
So the predicted data is Q*QH*Y
Write it out by hand and you'll see it.
The norms of QQHY and QHY are the same

    < MATLAB Documentation >
[Q,R] = qr(A) performs a QR decomposition on m-by-n matrix A such that A = Q*R.
The factor R is an m-by-n upper-triangular matrix, and the factor Q is an
m-by-m orthogonal matrix.
[___] = qr(A,0) produces an economy-size decomposition using any of the
previous output argument combinations. The size of the outputs depends on the
size of m-by-n matrix A:

If m > n, then qr computes only the first n columns of Q and the first n rows of R.

If m <= n, then the economy-size decomposition is the same as the regular decomposition.

If you specify a third output with the economy-size decomposition, then it is
returned as a permutation vector such that A(:,P) = Q*R.
    < /MATLAB Documentation >

Matlab's reference to the "economy" rerpresentation is what Trefethen and Bau
call the "reduced QR factorization".  Golub & Van Loan (1996, §5.2) call Q1R1
the thin QR factorization of A;

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
from scipy.linalg import solve_triangular
import xarray as xr

from aurora.transfer_function.regression.base import RegressionEstimator


class TRME(RegressionEstimator):
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
        self.expectation_psi_prime = np.ones(self.n_channels_out)
        self.sigma_squared = np.zeros(self.n_channels_out)

    @property
    def r0(self):
        return self.iter_control.r0

    @property
    def u0(self):
        return self.iter_control.u0

    @property
    def correction_factor(self):
        # MOVE THIS METHOD INTO AN RME-Specific CONFIG
        return self.iter_control.correction_factor

    def sigma(self, QHY, Y_or_Yc, correction_factor=1.0):
        """
        These are the error variances.
        TODO: Move this method to the base class, or a QR decorator.
        Computes the squared norms difference of the output channels from the
        "output channels inner-product with QQH"

        ToDo: Rename this to sigma_squared, or residual_variance rather than sigma.
        It is a variance.  Especially in the context or the redecnd using
        it's sqrt to normalize the residual amplitudes

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
        sigma : numpy array
            One entry per output channel.

        """
        Y2 = np.linalg.norm(Y_or_Yc, axis=0) ** 2  # variance?
        QHY2 = np.linalg.norm(QHY, axis=0) ** 2
        sigma = correction_factor * (Y2 - QHY2) / self.n_data

        try:
            assert (sigma > 0).all()
        except AssertionError:
            print("WARNING - Negative error variances observed")
            print(sigma)
            print("Setting sigma to zero - Negative sigma_squared observed")
            sigma *= 0
            # raise Exception
        return sigma

    def apply_huber_weights(self, sigma, YP):
        """
        Updates the values of self.Yc and self.expectation_psi_prime

        Parameters
        ----------
        sigma : numpy array
            1D array, the same length as the number of output channels
            see self.sigma() method for its calculation
        YP : numpy array
            The predicted data, usually from QQHY

        Returns
        -------

        function [YC,E_psiPrime] = HuberWt(Y,YP,sig,r0)

        inputs are data (Y) and predicted (YP), estiamted
        error variances (for each column) and Huber parameter r0
        allows for multiple columns of data
        """
        # Y_cleaned = np.zeros(self.Y.shape, dtype=np.complex128)
        for k in range(self.n_channels_out):
            r0s = self.r0 * np.sqrt(sigma[k])
            residuals = np.abs(self.Y[:, k] - YP[:, k])
            w = np.minimum(r0s / residuals, 1.0)
            self.Yc[:, k] = w * self.Y[:, k] + (1 - w) * YP[:, k]
            self.expectation_psi_prime[k] = 1.0 * np.sum(w == 1) / self.n_data
        return

    def update_predicted_data(self):
        pass

    def redescend(
        self,
        Y_predicted,
        sigma,
    ):
        """
        % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
        function[YC, E_psiPrime] = RedescendWt(Y, YP, sig, u0)

        % inputs
        are
        data(Y) and predicted(YP), estiamted
        % error
        variances(
        for each column) and Huber parameter u0
        % allows
        for multiple columns of data
        """
        # Y_cleaned = np.zeros(self.Y.shape, dtype=np.complex128)
        for k in range(self.n_channels_out):

            r = np.abs(self.Y[:, k] - Y_predicted[:, k]) / np.sqrt(sigma[k])
            t = -np.exp(self.u0 * (r - self.u0))
            w = np.exp(t)

            # cleaned data
            self.Yc[:, k] = w * self.Y[:, k] + (1 - w) * Y_predicted[:, k]

            # computation of E(psi')
            t = self.u0 * (t * r)
            t = w * (1 + t)
            self.expectation_psi_prime[k] = np.sum(t[t > 0]) / self.n_data
        return

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
            b0 = self.solve_underdetermined()
            return b0

        # <INITIAL ESTIMATE>
        self.qr_decomposition(self.X)
        b0 = solve_triangular(self.R, self.QHY)

        if self.iter_control.max_number_of_iterations > 0:
            converged = False
        else:
            converged = True
            self.expectation_psi_prime = np.ones(self.n_channels_out)  # defualt
            YP = self.Q @ self.QHY
            # YP needed only in case we want covariance and no huber and no redescend
            self.b = b0
            self.Yc = self.Y

        sigma = self.sigma(self.QHY, self.Y)
        self.iter_control.number_of_iterations = 0
        # </INITIAL ESTIMATE>

        while not converged:
            self.iter_control.number_of_iterations += 1
            if self.iter_control.number_of_iterations == 1:
                YP = self.Q @ self.QHY  # predicted data, initial estimate
            else:
                YP = self.Q @ self.QHYc
            self.apply_huber_weights(sigma, YP)
            self.update_QHYc()
            # QHYc = self.QH @ self.Yc
            self.b = solve_triangular(self.R, self.QHYc)  # self.b = R\QTY;

            # update error variance estimates, computed using cleaned data
            sigma = self.sigma(
                self.QHYc, self.Yc, correction_factor=self.correction_factor
            )
            converged = self.iter_control.converged(self.b, b0)
            b0 = self.b

        if self.iter_control.max_number_of_redescending_iterations:
            print(b0)
            # self.iter_control.number_of_redescending_iterations = 0;
            while self.iter_control.continue_redescending:
                self.iter_control._number_of_redescending_iterations += 1
                # add setter here
                YP = self.Q @ self.QHYc  # predict from cleaned data
                self.redescend(YP, sigma)  # update cleaned data, and expectation
                # updated error variance estimates, computed using cleaned data
                self.update_QHYc()  # QHYc = self.QH @ self.Yc
                self.b = solve_triangular(self.R, self.QHYc)
                sigma = self.sigma(self.QHYc, self.Yc)
            # crude estimate of expectation of psi ... accounting for
            # redescending influence curve
            self.expectation_psi_prime = 2 * self.expectation_psi_prime - 1

        if self.iter_control.return_covariance:
            self.compute_inverse_signal_covariance()
            self.compute_noise_covariance(YP)
            self.compute_squared_coherence(YP)

        return self.b

    def compute_noise_covariance(self, YP):
        """
        res_clean: The cleaned data minus the predicted data. The residuals
        SSR_clean: Sum of squares of the residuals.  Diagonal is real
        Parameters
        ----------
        YP

        Returns
        -------

        """
        res_clean = self.Yc - YP
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

    def compute_squared_coherence(self, YP):
        """
        res: Residuals: The original data minus the predicted data.
        #SSR : Sum of squares of the residuals.  Diagonal is real
        Parameters
        ----------
        YP

        Returns
        -------

        """
        res = self.Y - YP
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
        # array([ 0.97713185,  0.97552176,  0.97480946])
