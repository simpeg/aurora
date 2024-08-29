"""

This module contains the base class for regression functions.
 It follows Gary Egbert's EMTF Matlab code TRegression.m in
 which can be found in
iris_mt_scratch/egbert_codes-20210121T193218Z-001/egbert_codes/matlabPrototype_10-13-20/TF/classes

This class originally used numpy arrays to make adapting the Matlab easier, but
experimental support for xarray is now added (2024).

"""
import numpy as np
import xarray as xr

from aurora.transfer_function.regression.iter_control import IterControl
from loguru import logger
from typing import Optional, Union


class RegressionEstimator(object):
    """
    Abstract base class for solving Y = X*b + epsilon for b, complex-valued

    Many of the robust transfer estimation methods we will use repeat the
    model of solving Y = X*b +epsilon for b.  X is variously called the "input",
    "predictor", "explanatory", "confounding", "independent" "exogenous", variable(s)
    or the "design matrix", "model matrix" or "regressor matrix".
    Y are variously called the "output", "predicted", "outcome",
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
    _X : Union[xr.Dataset, xr.DataArray, np.ndarray]
        The underlying dataset is assumed to be if shape nCH x nObs (normally 2-dimensional)
        These are the input variables.  In the matlab codes each observation
        corresponds to a row and each parameter (channel) is a column.
        These data are transposed before regression
    X : numpy array (normally 2-dimensional)
        This is a "pure array" representation of _X used to emulate Gary
        Egbert's matlab codes. It may or may not be deprecated.
    _Y : xarray.Dataset
        These are the output variables, arranged same as X above.
    Y : numpy array (normally 2-dimensional)
        This is a "pure array" representation of _Y used to emulate Gary
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
        is a structure which controls the robust scheme iteration.
        On return also contains number of iterations.

    Kwargs:
    input_channel_names: list
        List of strings for channel names.

    """

    def __init__(
        self,
        X: Union[xr.Dataset, xr.DataArray, np.ndarray],
        Y: Union[xr.Dataset, xr.DataArray, np.ndarray],
        iter_control: IterControl = IterControl(),
        input_channel_names: Optional[Union[list, None]] = None,
        output_channel_names: Optional[Union[list, None]] = None,
        **kwargs,  # n
    ):
        """
        Constructor

        Parameters
        ----------
        X: Union[xr.Dataset, xr.DataArray, np.ndarray]
            The input channels for regression
        Y: Union[xr.Dataset, xr.DataArray, np.ndarray]
            The output channels for regression
        iter_control: IterControl
            Contains parameters controlling the regression, e.g. convergence criteria, thresholds, etc.
        input_channel_names: Optional[Union[list, None]]
            If X is np.ndarray, this allows associating channel names to X's columns
        output_channel_names: Optional[Union[list, None]]
            If Y is np.ndarray, this allows associating channel names to Y's columns

        ----------
        kwargs
        """
        self._X = X
        self._Y = Y
        self.iter_control = iter_control
        self.b = None
        self.cov_nn = None
        self.cov_ss_inv = None
        self.squared_coherence = None
        self._set_up_regression_variables()
        self.R2 = None
        self.qr_input = "X"
        self._Q = None
        self._R = None
        self._QH = None  # conjugate transpose of Q (Hermitian operator)
        self._QHY = None  #
        self._QHYc = None
        self._n_channels_out = None
        self._input_channel_names = input_channel_names
        self._output_channel_names = output_channel_names

    def _set_up_regression_variables(self) -> None:
        """
        Initialize arrays needed for regression and cast any xarray to numpy

        Development Notes:

        When xr.Datasets are X, Y we cast to array (num channels x num observations) and then transpose them
        When xr.DataArrays are X, Y extract the array -- but how do we know whether or not to transpose?
        - it would be nice to have a helper function that applies the logic of getting the data from the
         xarray and transposing or not appropriately.
        - for now we assume that the input data are organized so that input arrays are (n_ch x n_observations).
        This assumption is OK for xr.Dataset where the datavars are the MT components ("hx", "hy", etc)

        """
        self.X = _input_to_numpy_array(self._X)
        self.Y = _input_to_numpy_array(self._Y)
        self.Yc = np.zeros(self.Y.shape, dtype=np.complex128)
        self._check_number_of_observations_xy_consistent()

    @property
    def input_channel_names(self) -> list:
        """returns the list of channel names associated with X"""
        if self._input_channel_names is None:
            self._input_channel_names = _get_channel_names(self._X, label="IN")
        return self._input_channel_names

    @property
    def output_channel_names(self) -> list:
        """returns the list of channel names associated with Y"""
        if self._output_channel_names is None:
            self._output_channel_names = _get_channel_names(self._Y, label="OUT")
        return self._output_channel_names

    @property
    def inverse_signal_covariance(self) -> np.ndarray:
        """Returns the inverse signal covariance matrix of the input channels as xarray"""
        return self.cov_ss_inv

    @property
    def noise_covariance(self) -> xr.DataArray:
        """Returns the noise covariance matrix of the output channels  as xarray"""
        return self.cov_nn

    def b_to_xarray(self) -> xr.DataArray:
        """Wraps the TF results as an xarray"""
        xra = xr.DataArray(
            np.transpose(self.b),
            dims=["output_channel", "input_channel"],
            coords={
                "output_channel": self.output_channel_names,
                "input_channel": self.input_channel_names,
            },
        )
        return xra

    def solve_underdetermined(self) -> None:
        """
        Solves the regression problem if it is under-determined -- Not Stable

        Development Notes:
        20210806
        This method was originally in TRME.m, but it does not depend in
        general on using RME method so putting it in the base class.

        We basically never get here and when we do, we don't trust the results
        https://docs.scipy.org/doc/numpy-1.9.2/reference/generated/numpy.linalg.svd.html
        https://www.mathworks.com/help/matlab/ref/double.svd.html

        Note that the svd outputs are different in matlab and numpy
        https://stackoverflow.com/questions/50930899/svd-command-in-python-v-s-matlab
        "The SVD of a matrix can be written as

        A = U S V^H

        Where the ^H signifies the conjugate transpose. Matlab's svd command returns U, S and V,
        while numpy.linalg.svd returns U, the diagonal of S, and V^H.
        Thus, to get the same S and V as in Matlab you need to reconstruct the S and also get the V:


        ORIGINAL MATLAB

        % Overdetermined problem...use svd to invert, return
        % NOTE: the solution IS non - unique... and by itself RME is not setup
        % to do anything sensible to resolve the non - uniqueness(no prior info
        % is passed!).  This is stop-gap, to prevent errors when using RME as
        % part of some other estimation scheme!
        [u,s,v] = svd(obj.X,'econ');
        sInv = 1./diag(s);
        obj.b = v*diag(sInv)*u'*obj.Y;
        if ITER.returnCovariance
           obj.Cov_NN = zeros(K,K);
           obj.Cov_SS = zeros(nParam,nParam);
        end
        result = obj.b


        -------

        """
        U, s, V = np.linalg.svd(self.X, full_matrices=False)
        S_inv = np.diag(1.0 / s)
        self.b = (V.T.conj() @ S_inv @ U.T.conj()) * self.Y
        if self.iter_control.return_covariance:
            logger.warning(
                "Warning covariances are not xarray, may break things downstream"
            )
            self.cov_nn = np.zeros((self.n_channels_out, self.n_channels_out))
            self.cov_ss_inv = np.zeros((self.n_channels_in, self.n_channels_in))

        return

    def _check_number_of_observations_xy_consistent(self) -> None:
        """Raises an exception if the X, Y data have different number of observations"""
        if self.Y.shape[0] != self.X.shape[0]:
            logger.error(
                f"Design matrix (X) has {self.X.shape[0]} rows but data (Y) "
                f"has {self.Y.shape[0]}"
            )
            raise Exception

    @property
    def n_data(self) -> int:
        """
        Return the number of multivariate observations in the regression dataset.

        Development Notes:
         This may need to be modified if we use masked arrays

        Returns
        -------
        int:
            The number of rows in the input data vector
        """
        return self.X.shape[0]

    @property
    def n_channels_in(self) -> int:
        """
        Returns the number of input channels.

        Returns
        -------
        int:
            The number of channels of input data (columns of X)
        """
        return self.X.shape[1]

    @property
    def n_channels_out(self) -> int:
        """
        number of output variables

        Returns
        int
            number of output variables (Assumed to be num columns of a 2D array)
        """
        if self._n_channels_out is None:
            self._n_channels_out = self.Y.shape[1]
        return self._n_channels_out

    @property
    def degrees_of_freedom(self) -> int:
        """
        gets the number of degress of freedom in the dataset.
        Returns
        int
            The total number of multivariate observations minus the number of input channels.
        """
        return self.n_data - self.n_channels_in

    @property
    def is_underdetermined(self) -> bool:
        """
        Check if the regression problem is under-determined
        Returns
        -------
        bool
            True if regression problem is under-determined, otherwise False
        """
        return self.degrees_of_freedom < 0

    # TODO Add support for masked arrays
    # def mask_input_channels(self):
    #     """
    #     ADD NAN MANAGEMENT HERE
    #     Returns
    #     -------
    #
    #     """
    #     pass

    def qr_decomposition(
        self, X: Optional[Union[np.ndarray, None]] = None, sanity_check: bool = False
    ) -> tuple:
        """
        performs QR decomposition on input matrix X.

        If X is not provided as an optional argument then check the value of self.qr_input

        Parameters
        ----------
        X: numpy array
            In RME this is the Input channels X
            In RME_RR this is the RR channels Z
        sanity_check: boolean
            check QR decomposition is working correctly.  Set to True for debugging.
            Can probably be deprecated.

        Returns
        -------
        Q, R: tuple
            The matrices Q and R from the QR-decomposition.
            X = Q R where Q is unitary/orthogonal and R upper triangular.
        """
        if X is None:
            if self.qr_input == "X":
                X = self.X
            elif self.qr_input == "Z":
                X = self.Z
            else:
                msg = f"Matrix to perform QR decomposition not specified by {self.qr_input}"
                logger.error("msg")
                raise ValueError(msg)

        Q, R = np.linalg.qr(X)
        self._Q = Q
        self._R = R
        if sanity_check:
            if np.isclose(np.matmul(Q, R) - X, 0).all():
                pass
            else:
                logger.error("Failed QR decomposition sanity check")
                raise Exception
        return Q, R

    @property
    def Q(self) -> np.ndarray:
        """Returns the Q matrix from the QR decomposition"""
        return self._Q

    @property
    def R(self) -> np.ndarray:
        """Returns the R matrix from the QR decomposition"""
        return self._R

    @property
    def QH(self) -> np.ndarray:
        """Returns the conjugate transpose of the Q matrix from the QR decomposition"""
        if self._QH is None:
            self._QH = self.Q.conj().T
        return self._QH

    @property
    def QHY(self) -> np.ndarray:
        """
        Returns the QH @ Y

        QHY is a convenience matrix that makes computing the predicted data (QQHY) more efficient.

        """
        if self._QHY is None:
            self._QHY = self.QH @ self.Y
        return self._QHY

    def estimate_ols(self, mode: Optional[str] = "solve") -> np.ndarray:
        """
        Solve Y = Xb with ordinary least squares, not robust regression.

        Development Notes:
        Brute Force tends to be less stable because we actually compute the
        inverse matrix.  It is not recommended, its just here for completeness.
        X'Y=X'Xb
        (X'X)^-1 X'Y = b

        Parameters
        ----------
        mode : str
            must be one of ["qr", "brute_force", "solve"]

        Returns
        -------
        b : numpy array
            Normally the impedance tensor Z

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
                logger.error(f"mode {mode} not recognized")
                raise Exception
        self.b = b
        return b

    def estimate(self) -> np.ndarray:
        """
        Executes the regression

        Returns
        -------
        b: np.ndarray
            The regression solution to Y = Xb
        """
        b = self.estimate_ols(mode="qr")
        return b


def _input_to_numpy_array(X: Union[xr.Dataset, xr.DataArray, np.ndarray]) -> np.ndarray:
    """
    Casts data input to regression as numpy array, with channels as column vectors.

    Currently, we store array channels row-wise (as num_channels x num_observations),
    but the way regression is set up the variables should be column vectors (num_observations x num_channels)

    This is a place where we could distill the logic for which dimension is which.


    Parameters
    ----------
    X: Union[xr.Dataset, xr.DataArray]
        Data to be used in regression in xarray form

    Returns
    -------
    output: np.ndarray
        Data to be used in regression as a numpy array

    """
    if isinstance(X, xr.Dataset):
        output = X.to_array().data.T
    elif isinstance(X, xr.DataArray):
        output = X.data.T
        if len(output.shape) == 1:
            output = np.atleast_2d(output).T  # cast to 2D if 1D
    elif isinstance(X, np.ndarray):
        msg = "np.ndarray input is assumed to be nCH x nObs -- transposing"
        logger.debug(msg)
        output = X.T
    else:
        msg = f"input argument of type {type(X)} not supported -- try an xarray"
        raise NotImplementedError(msg)

    return output


def _get_channel_names(
    X: Union[xr.Dataset, xr.DataArray, np.ndarray], label: Optional[str] = ""
) -> list:
    """
    Returns list of channel names.

    If X is a numpy array, names will be created. These are needed by TRME.estimate() to return xarrays.

    Parameters
    ----------
    X: Union[xr.Dataset, xr.DataArray, np.ndarray]
        If X is xarray just return the labels
        If X is numpy array, make the names up.  numpy array assumed to contain data from each channel
        in a separate row, i.e. (n_ch x n_observations) shaped array.
    label: Optional[str]
        This gets prepended onto incrementing integers for channel labels.
        For example, this could be "input", "output", or a station name.
        Used to keep the indexing a 2D xarray unique.

    Returns
    -------
    channel_names: list
        The names of the channels for the input array X
    """
    if isinstance(X, xr.Dataset):
        channel_names = list(X.data_vars)
    elif isinstance(X, xr.DataArray):
        # Beware hard coded assumption of "variable"
        try:
            channel_names = list(X.coords["variable"].values)
        except TypeError:  # This happens when xarray has only one channel
            channel_names = [
                X.coords["variable"].values.item(),
            ]
    else:
        # numpy array doesn't have input channel_names predefined
        channel_names = np.arange(
            X.shape[0]
        )  # note its 0, not 1 here because we are talking to _X
        channel_names = [f"{label}{x}" for x in channel_names]
    return channel_names
