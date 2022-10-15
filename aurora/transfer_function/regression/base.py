"""
follows Gary's TRegression.m in
iris_mt_scratch/egbert_codes-20210121T193218Z-001/egbert_codes/matlabPrototype_10-13-20/TF/classes


There are some high-level decisions to make about usage of xarray.Dataset,
xarray.DataArray and numpy arrays.  For now I am going to cast X,Y to numpy
arrays to follow Gary's codes more easily since his are matlab arrays.
"""
import numpy as np
import xarray as xr
from aurora.transfer_function.regression.iter_control import IterControl


class RegressionEstimator(object):
    """
    Abstract base class for solving Y = X*b + epsilon for b, complex-valued

    Many of the robust transfer estimation methods we will use repeat the
    model of solving Y = X*b +epsilon for b.  X is variously called the "input",
    "predictor", "explanatory", "confounding", "independent" "exogenous", variable(s)
    or the "design matrix", "model matrix" or "regressor matrix".
    Y are variously called the the "output", "predicted", "outcome",
    "response", "endogenous", "regressand", or "dependent" variable.  I will try to
    use input and output.

    When we "regress Y on X", we use the values of variable X to predict
    those of Y.

    Typically operates on single "frequency_band".
    Allows multiple columns of Y, but estimates b for each column separately.


    Estimated signal and noise covariance matrices can be used to compute error
    together to compute covariance for the matrix of regression coefficients b

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.



    Attributes
    ----------
    _X : xarray.Dataset
        X.data is numpy array (normally 2-dimensional)
        These are the input variables.  Like the matlab codes each observation
        corresponds to a row and each parameter (channel) is a column.
    X :  numpy array (normally 2-dimensional)
        This is a "pure array" representation of _X used to emulate Gary
        Egbert's matlab codes. It may or may not be deprecated.
    _Y : xarray.Dataset
        These are the output variables, aranged same as X above.
    Y : numpy array (normally 2-dimensional)
        This is a "pure array" representation of _X used to emulate Gary
        Egbert's matlab codes. It may or may not be deprecated.
    b : numpy array (normally 2-dimensional)
        Matrix of regression coefficients, i.e. the solution to the regression
        problem.  In our context they are the "Transfer Function"
        The dimension of b, to match X and Y is [n_input_ch, n_output_ch]
        When we are solving "channel-by-channel", b is usually [2,1].

    inverse_signal_covariance: numpy array (n_input_ch, n_input_ch)
        This was Cov_SS in Gary's matlab codes.  It is basically inv(X.H @ X)
        Reference needed
    noise_covariance : numpy array (n_output_ch, n_output_ch)
        This was Cov_NN in Gary's matlab codes
        Reference needed
    squared_coherence: numpy array (n_output_ch)
        This was R2 in Gary's matlab codes
        Formula?  Reference?
        Squared coherence (top row is using raw data, bottom cleaned, with crude
        correction for amount of downweighted data)
    Yc : numpy array (normally 2-dimensional)
        A "cleaned" version of Y the output variables.
    iter_control : transfer_function.iter_control.IterControl()
        is a structure which controls the robust scheme
        Fields: r0, RG.nITmax, tol (rdcndwt ... not coded yet)
        On return also contains number of iterations
    """

    def __init__(self, **kwargs):
        """

        Parameters

        ----------
        kwargs
        """
        self._X = kwargs.get("X", None)
        self._Y = kwargs.get("Y", None)
        self.b = None
        self.cov_nn = None
        self.cov_ss_inv = None
        self.squared_coherence = None
        self.iter_control = kwargs.get("iter_control", IterControl())

        self.X = self._X.to_array().data.T
        self.Y = self._Y.to_array().data.T
        self.Yc = np.zeros(self.Y.shape, dtype=np.complex128)
        self.check_number_of_observations_xy_consistent()
        self.R2 = None
        self.qr_input = "X"
        self._Q = None
        self._R = None
        self._QH = None  # conjugate transpose of Q (Hermitian operator)
        self._QHY = None  #
        self._QHYc = None

    @property
    def inverse_signal_covariance(self):
        return self.cov_ss_inv

    @property
    def noise_covariance(self):
        return self.cov_nn

    def b_to_xarray(self):
        xra = xr.DataArray(
            np.transpose(self.b),
            dims=["output_channel", "input_channel"],
            coords={
                "output_channel": list(self._Y.data_vars),
                "input_channel": list(self._X.data_vars),
            },
        )
        return xra

    def solve_underdetermined(self):
        """
        20210806
        This method was originally in TRME.m, but it does not depend in
        general on using RME method so I am putting it in the base class.

        We basically never get here and when we do we dont trust the results
        https://docs.scipy.org/doc/numpy-1.9.2/reference/generated/numpy.linalg.svd.html
        https://www.mathworks.com/help/matlab/ref/double.svd.html

        <ORIGINAL MATLAB>
            <COMMENT>
        Overdetermined problem...use svd to invert, return
        NOTE: the solution IS non - unique... and by itself RME is not setup
        to do anything sensible to resolve the non - uniqueness(no prior info
        is passed!).  This is stop-gap, to prevent errors when using RME as
        part of some other estimation scheme!
            </COMMENT>
            <CODE>
        [u,s,v] = svd(obj.X,'econ');
        sInv = 1./diag(s);
        obj.b = v*diag(sInv)*u'*obj.Y;
        if ITER.returnCovariance
           obj.Cov_NN = zeros(K,K);
           obj.Cov_SS = zeros(nParam,nParam);
        end
        result = obj.b
            </CODE>
        </ORIGINAL MATLAB>


        -------

        """
        U, s, V = np.linalg.svd(self.X, full_matrices=False)
        S_inv = np.diag(1.0 / s)
        self.b = (V.T @ S_inv @ U.T) * self.Y
        if self.iter_control.return_covariance:
            print("Warning covariances are not xarray, may break things downstream")
            self.cov_nn = np.zeros((self.n_channels_out, self.n_channels_out))
            self.cov_ss_inv = np.zeros((self.n_channels_in, self.n_channels_in))

        return

    def check_number_of_observations_xy_consistent(self):
        if self.Y.shape[0] != self.X.shape[0]:
            print(
                f"Design matrix (X) has {self.X.shape[0]} rows but data (Y) "
                f"has {self.Y.shape[0]}"
            )
            raise Exception

    @property
    def n_data(self):
        """
        or return self.Y.shape[0], any reason to choose one or the other?
        See Also Issue#7 in aurora github: Masked arrays stuff will go here
        Returns
        -------

        """
        return self.X.shape[0]

    @property
    def n_channels_in(self):
        return self.X.shape[1]

    @property
    def n_channels_out(self):
        """number of output variables"""
        return self.Y.shape[1]

    @property
    def degrees_of_freedom(self):
        return self.n_data - self.n_channels_in

    @property
    def is_underdetermined(self):
        return self.degrees_of_freedom < 0

    def mask_input_channels(self):
        """
        ADD NAN MANAGEMENT HERE
        Returns
        -------

        """
        pass

    def qr_decomposition(self, X=None, sanity_check=False):
        """
        performs QR decomposition on input matrix X.  If X is not provided as a kwarg
        then check the value of self.qr_input
        Parameters
        ----------
        X: numpy array
            In TRME this is the Input channels X
            In TRME_RR this is the RR channels Z
        sanity_check: boolean
            check QR decomposition is working correctly.  Can probably be deprecated.

        Returns
        -------

        """
        if X is None:
            if self.qr_input == "X":
                X = self.X
            elif self.qr_input == "Z":
                X = self.Z
            else:
                print("Matrix to perform QR decompostion not specified")
                raise Exception

        Q, R = np.linalg.qr(X)
        self._Q = Q
        self._R = R
        if sanity_check:
            if np.isclose(np.matmul(Q, R) - X, 0).all():
                pass
            else:
                print("Failed QR decompostion sanity check")
                raise Exception
        return Q, R

    @property
    def Q(self):
        return self._Q

    @property
    def R(self):
        return self._R

    @property
    def QH(self):
        if self._QH is None:
            self._QH = self.Q.conj().T
        return self._QH

    @property
    def QHY(self):
        if self._QHY is None:
            self._QHY = self.QH @ self.Y
        return self._QHY

    # @property
    # def QHYc(self):
    #     if self._QHYc is None:
    #         self.update_QHYc()
    #     return self._QHYc
    #
    # def update_QHYc(self):
    #     self._QHYc = self.QH @ self.Yc

    def estimate_ols(self, mode="solve"):
        """

        Parameters
        ----------
        mode : str
            "qr", "brute_force", "solve"

        Returns
        -------
        b : numpy array
            Normally the impedance tensor Z

        Solve Y = Xb

        Brute Force tends to be less stable because we actually compute the
        inverse matrix.  It is not recommended, its just here for completeness.
        X'Y=X'Xb
        (X'X)^-1 X'Y = b

        -------

        """
        if mode.lower() == "qr":
            from scipy.linalg import solve_triangular

            self.qr_decomposition(self.X)
            b = solve_triangular(self.R, self.QH @ self.Y)
        else:
            X = self.X
            Y = self.Y
            XH = np.conj(X.T)
            XHX = XH @ X
            XHY = XH @ Y
            if mode.lower() == "brute_force":
                XHX_inv = np.linalg.inv(XHX)
                b = np.matmul(XHX_inv, XHY)
            elif mode.lower() == "solve":
                b = np.linalg.solve(XHX, XHY)
            else:
                print(f"mode {mode} not recognized")
                raise Exception
        self.b = b
        return b

    def estimate(self):
        Z = self.estimate_ols(mode="qr")
        return Z
