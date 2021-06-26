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

initialization:
obj = TRME(X=X, Y=Y, iter_control=iter)

TODO: set Q, R to be variables asssociated with this class (actually put Q,
R inside RegressionEstimator())
TODO: Move TRME to RME-Regression-Estimator

THe QR-decomposition is employed on the matrix of independent variables.
X = Q R where Q is unitary/orthogonal and R upper triangular.
Since X is [n_data x n_channels_in] Q is [n_data x n_data].  Wikipedia has a
nice description of the QR factorization:
https://en.wikipedia.org/wiki/QR_decomposition
On a high level, the point of the QR decomposition is to transform the data
into a domain where the inversion is done with a triangular matrix.

Note that we employ here the "economical" form of the QR decompostion,
so that Q is not square, and not in fact unitary.

Really Q = [Q1 | Q2] where Q1 has as many columns as there are input variables
and Q2 is a matrix of zeros.  In this case QQH is the projection matrix,
or hat matrix equivalent to X(XHX)^-1XH.

The use of QHY is not so much physically meaningful as it is a trick to
compute more efficiently QQHY.  ? Really are we doing fewer calculations?



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

There are sevearl discussions online about the differences in
numpy, scipy, sklearn, skcuda etc.
https://mail.python.org/pipermail/numpy-discussion/2012-November/064485.html
We will default to using numpy for now.
Note that numpy's default is to use the "reduced" form of Q, R.  R is
upper-right triangular.

This is cute:
https://stackoverflow.com/questions/26932461/conjugate-transpose-operator-h-in-numpy

THe Matlab mldivide flowchart can be found here:
https://stackoverflow.com/questions/18553210/how-to-implement-matlabs-mldivide-a-k-a-the-backslash-operator
And the matlab documentation here
http://matlab.izmiran.ru/help/techdoc/ref/mldivide.html
"""
import numpy as np
from scipy.linalg import solve_triangular

from aurora.transfer_function.TRegression import RegressionEstimator

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
        #MOVE THIS METHOD INTO AN RME-Specific CONFIG
        return self.iter_control.correction_factor


    def sigma(self, QHY, Y_or_Yc, correction_factor=1.0):
        """
        These are the error variances.
        TODO: Move this method to the base class
        Computes the squared norms difference of the output channels from the
        "output channels inner-product with Q"

        This seems like it is more like sigma^2 than sigma.  i.e. it is a
        variance.  Especially in the context or the redecnd using it's sqrt to
        normalize the residual amplitudes

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

        #<PUT THIS SOMEWHERE RELEVANT>
        Y2 is the norm sqaured of the data, and
        QQHY is the projected (and predicted) data...
        The predicted data has to lie in span of the columns in the
        design matrix X.
        The predicted data has to be a linear combination of the
        columns of b.
        Q is an orthoganal basis for the columns of X
        So the predicted data is Q*QH*X
        Write it out by hand and you'll see it.
        The norms of QQHY  and the length of QHY are the same
        #</PUT THIS SOMEWHERE RELEVANT>

        Returns
        -------
        sigma : numpy array

        """
        Y2 = np.linalg.norm(Y_or_Yc, axis=0)**2 #variance?
        QHY2 = np.linalg.norm(QHY, axis=0)**2
        sigma = correction_factor * (Y2 - QHY2) / self.n_data;
        return sigma

    def solve_overdetermined(self):
        """
        Overdetermined problem...use svd to invert, return
        NOTE: the solution IS non - unique... and by itself RME is not setup
        to do anything sensible to resolve the non - uniqueness(no prior info
        is passed!).  This is stop-gap, to prevent errors when using RME as
        part of some other estimation scheme!

        We basically never get here and when we do we dont trust the results
        https://docs.scipy.org/doc/numpy-1.9.2/reference/generated/numpy.linalg.svd.html
        https://www.mathworks.com/help/matlab/ref/double.svd.html
        Returns
        -------

        """
        print("STILL NEEDS TO BE TRANSLATED")
        #Return None /nan and flag it is a valid solution
        #for aurora 2021 Sept.
        U, s, V = np.linalg.svd(self.X, full_matrices=False)
        #[u, s, v] = svd(self.X, 'econ');
        sInv = 1. / diag(s);
        self.b = v * diag(sInv) * u.T * self.Y;
        if self.iter_control.return_covariance:
            self.noise_covariance = np.zeros(self.n_channels_out,
                                             self.n_channels_out);
            self.inverse_signal_covariance = np.zeros(self.n_param,
                                                      self.n_param);

        return self.b

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
        #Y_cleaned = np.zeros(self.Y.shape, dtype=np.complex128)
        for k in range(self.n_channels_out):
            r0s = self.r0 * np.sqrt(sigma[k])
            residuals = np.abs(self.Y[:, k] - YP[:, k])
            w = np.minimum(r0s/residuals, 1.0)
            self.Yc[:, k] = w * self.Y[:, k] + (1 - w) * YP[:, k]
            self.expectation_psi_prime[k] = 1.0 * np.sum(w == 1) / self.n_data;
        return

    def qr_decomposition(self, X, sanity_check=False):
        [Q, R] = np.linalg.qr(X)
        if sanity_check:
            if np.isclose(np.matmul(Q, R) - self.X, 0).all():
                pass
            else:
                print("Failed QR decompostion sanity check")
                raise Exception
        return Q, R

    def update_predicted_data(self):
        pass

    def redescend(self, Y_predicted, sigma, ):
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
        #Y_cleaned = np.zeros(self.Y.shape, dtype=np.complex128)
        for k in range(self.n_channels_out):

            r = np.abs(Y[:, k] - Y_predicted[:, k]) / np.sqrt(sigma[k])
            t = -np.exp(self.u0 * (r - u0))
            w = np.exp(t)

            # cleaned data
            self.Yc[:, k] = w * Y[:, k] + (1 - w) * Y_predicted[:, k]

            # computation of E(psi')
            t = u0 * (t * r)
            t = w * (1 + t)
            self.expectation_psi_prime[k] = np.sum(t[t>0]) / self.n_data
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
        if self.is_overdetermined:
            b0 = self.solve_overdetermined()
            return b0

        #<INITIAL ESTIMATE>
        Q, R = self.qr_decomposition(self.X)
        QH = np.conjugate(np.transpose(Q))
        # initial LS estimate b0, error variances sigma
        QHY = np.matmul(QH, self.Y)
        b0 = solve_triangular(R, QHY)
        # </INITIAL ESTIMATE>

        if self.iter_control.max_number_of_iterations > 0:
            converged = False;
        else:
            converged = True
            self.expectation_psi_prime = np.ones(self.n_channels_out) #let
            # this be defualt
            YP = np.matmul(Q, QHY);#not sure we need this?  only in the case
            # that we want the covariance and do no huber and no redescend
            self.b = b0;
            self.Yc = self.Y;

        sigma = self.sigma(QHY, self.Y)
        self.iter_control.number_of_iterations = 0;

        while not converged:
            self.iter_control.number_of_iterations += 1
            YP = np.matmul(Q, QHY) # predicted data,
            self.apply_huber_weights(sigma, YP)
            QHYc = np.matmul(QH, self.Yc)
            self.b = solve_triangular(R, QHYc) #self.b = R\QTY;

            #update error variance estimates, computed using cleaned data
            sigma = self.sigma(QHYc, self.Yc,
                               correction_factor=self.correction_factor)
            converged = self.iter_control.converged(self.b, b0);
            b0 = self.b;

        if self.iter_control.max_number_of_redescending_iterations:
            print(b0)
            #self.iter_control.number_of_redescending_iterations = 0;
            while self.iter_control.continue_redescending:
                self.iter_control._number_of_redescending_iterations += 1
                #add setter here
                YP = np.matmul(Q, QHYc) # predict from cleaned data
                self.redescend(YP, sigma) #update cleaned data, and expectation
                # updated error variance estimates, computed using cleaned data
                QHYc = np.matmul(QH, self.Yc)
                self.b = solve_triangular(R, QHYc)
                sigma = self.sigma(QHYc, self.Yc)
            # crude estimate of expectation of psi ... accounting for
            # redescending influence curve
            self.expectation_psi_prime = 2 * self.expectation_psi_prime - 1

        result = self.b;

        if self.iter_control.return_covariance:
            # compute error covariance matrices
            self.inverse_signal_covariance= np.linalg.inv(np.matmul(R.conj().T,R));

            res_clean = self.Yc - YP;
            SSR_clean = np.conj(res_clean.conj().T @ res_clean);
            res = self.Y-YP;
            SSR = np.conj(np.matmul(res.conj().T, res));
            Yc2 = np.abs(self.Yc)**2
            SSYC = np.sum(Yc2, axis=0);
            inv_psi_prime2 = np.diag(1. / (self.expectation_psi_prime**2))
            degrees_of_freedom = self.n_data-self.n_param
            self.noise_covariance = inv_psi_prime2 @ SSR_clean / degrees_of_freedom
            self.R2 = 1-np.diag(np.real(SSR)).T / SSYC
            self.R2[self.R2 < 0] = 0
        return self.b
            
            
