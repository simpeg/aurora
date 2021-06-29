"""
follows Gary's TRegression.m in
iris_mt_scratch/egbert_codes-20210121T193218Z-001/egbert_codes/matlabPrototype_10-13-20/TF/classes


There are some high-level decisions to make about usage of xarray.Dataset,
xarray.DataArray and numpy arrays.  For now I am going to cast X,Y to numpy
arrays to follow Gary's codes more easily since his are matlab arrays.
"""
import numpy as np
import xarray as xr
from aurora.transfer_function.iter_control import IterControl

class RegressionEstimator(object):
    """
    Abstract base class for solving Y = X*b + epsilon for b, complex-valued

    Many of the robust transfer estimation methods we will use repeat the
    model of solving Y = X*b +epsilon for b.  X is variously called the "input",
    "predictor", "explanatory" or "confounding" or "independent" variable(s).
    Y are variously called the the "output", "predicted", "outcome",
    "response",  or "dependent" variable.  I will try to use independent and
    dependent.

    When we "regress Y on X", we use the values of variable X to predict
    those of Y.

    Typically operates on single "frequency_band"s one-at-a-time/
    Allows multiple columns of Y, but estimates b for each column separately
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
    _X : xarray.Dataset or xarray.DataArray ... still trying to decide
        numpy array (normally 2-dimensional)
        These are the independent variables.  In the matlab codes each
        observation was a row and each parameter (channel) is a column
    X :  numpy array (normally 2-dimensional)
        This is a "pure array" representation of _X used to emulate Gary
        Egbert's matlab codes. It may or may not be deprecated.
    _Y : xarray.Dataset or xarray.DataArray
        These are the dependent variables.
    Y : numpy array (normally 2-dimensional)
        These are the dependent variables.  In the matlab codes each
        observation was a row and each parameter (channel) is a column
    b : numpy array (normally 2-dimensional)
        Matrix of regression coefficients, i.e. the solution to the regression
        problem.  In our context they are the "Transfer Function"
    inverse_signal_covariance: numpy array (????-Dimensional)
        This was Cov_SS in Gary's matlab codes
        Formula? Reference?
    noise_covariance : numpy array (????-Dimensional)
        This was Cov_NN in Gary's matlab codes
        Formula?  Reference?
    squared_coherence: numpy array (????-Dimensional)
        This was R2 in Gary's matlab codes
        Formula?  Reference?
        Squared coherence (top row is using raw data, bottom cleaned, with crude
        correction for amount of downweighted data)
    Yc : numpy array (normally 2-dimensional)
        A "cleaned" version of Y the dependent variables.
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
        self.inverse_signal_covariance = None
        self.noise_covariance = None
        self.squared_coherence = None
        self.iter_control = kwargs.get("iter_control", IterControl())

        self.X = self.cast_data_to_2d_for_regression(self._X)
        self.Y = self.cast_data_to_2d_for_regression(self._Y)
        self.Yc = np.zeros(self.Y.shape, dtype=np.complex128)
        self.check_number_of_observations_xy_consistent()
        self.R2 = None

        self._Q = None
        self._R = None
        self._QH = None #conjugate transpose of Q (Hermitian operator)

    def cast_data_to_2d_for_regression(self, XY):
        """
        When the data are "harvested" from frequency bands they have a
        typical STFT structure, which means one axis is time (the time of the
        window that was FFT-ed) and the other is frequency.  However we make
        no distinction between Fourier coefficients (or bins) within a band,
        so we need to gather all the FCs for each channel into a 1D array.
        This performs that reshaping (ravelling) operation

        It is not important how we unravel the FCs but it is important that
        we use the same scheme for X and Y.

        Parameters
        ----------
        XY: either X or Y of the regression nomenclature.  Should be an 
        xarray.Dataset already splitted on channel

        Returns
        -------
        output_array: numpy array of two dimensions (observations, channel)

        """
        if isinstance(XY, xr.DataArray):
            XY = XY.to_dataset("channel")
        n_channels = len(XY)
        n_frequency = len(XY.frequency)
        try:
            n_segments = len(XY.time)
        except TypeError:
            n_segments = 1
            #overdetermined problem
        n_fc_per_channel = n_frequency * n_segments

        output_array = np.full((n_fc_per_channel, n_channels),
                               np.nan+1.j*np.nan, dtype=np.complex128)

        channel_keys = list(XY.keys())
        for i_ch, key in enumerate(channel_keys):
            output_array[:, i_ch] = XY[key].data.ravel()
        return output_array
    


    def check_number_of_observations_xy_consistent(self):
        if self.Y.shape[0] != self.X.shape[0]:
            print(f"Design matrix (X) has {self.X.shape[0]} rows but data (Y) "
                  f"has {self.Y.shape[0]}")
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

    #REPLACE WITH n_channels_in
    @property
    def n_param(self):
        return self.X.shape[1]

    @property
    def n_channels_out(self):
        """ number of dependent variables"""
        return self.Y.shape[1]

    @property
    def is_overdetermined(self):
        return self.n_param > self.n_data

    # ADD NAN MANAGEMENT HERE


    # <MAY SUPERCLASS REGRESSION ESTIMATOR LEAVING THIS AN ABSTRACT BASE CLASS>
    def qr_decomposition(self, X, sanity_check=False):
        Q, R = np.linalg.qr(X)
        self._Q = Q
        self._R = R
        if sanity_check:
            if np.isclose(np.matmul(Q, R) - self.X, 0).all():
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
            self._QH = Q.conj().T
        return self._QH

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
        if mode.lower()=="qr":
            from scipy.linalg import solve_triangular
            self.qr_decomposition(self.X)
            b = solve_triangular(self.R, self.QH @ self.Y)
        else:
            X = self.X
            Y = self.Y
            XH = np.conj(X.T)
            XHX = XH @ X
            XHY = XH @ Y
            if mode.lower()=="brute_force":
                XHX_inv = np.linalg.inv(XHX)
                b = np.matmul(XHX_inv, XHY)
            elif mode.lower()=="solve":
                b = np.linalg.solve(XHX, XHY)
            else:
                print(f"mode {mode} not recognized")
                raise Exception
        self.b = b
        return b


    def estimate(self):
        print("this method is not defined for the abstract base class")
        print("But we put OLS in here for dev")
        Z = self.estimate_ols(mode="qr")
        return Z
    # </MAY SUPERCLASS REGRESSION ESTIMATOR LEAVING THIS AN ABSTRACT BASE CLASS>



