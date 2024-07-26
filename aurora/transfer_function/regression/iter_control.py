"""
This module contains a class that holds parameters to control the iterations of
robust regression including convergence criteria and thresholds.

Based on Gary's IterControl.m in
iris_mt_scratch/egbert_codes-20210121T193218Z-001/egbert_codes/matlabPrototype_10-13-20/TF/classes
"""
from loguru import logger
import numpy as np

from aurora.transfer_function.regression.helper_functions import rme_beta


class IterControl(object):
    """
    Notes:
         Here is a class to hold variables that are used to control the regression
         - currently this is used for variations on RME (Robust M-Estimator)
         - TODO: in the original matlab code there was a class property called `epsilon`, but
         it was unused; There was no documentation about it's purpose, except possibly that
         the abstract base class solved Y = X*b + epsilon for b, complex-valued.
         Perhaps this was intended as an intrinsic tolerated noise level.  The value of
         epsilon was set to 1000.
         -  TODO The return covariance boolean just initializes arrays of zeros.  Needs to be
         made functional or removed
    """

    def __init__(
        self,
        max_number_of_iterations: int = 10,
        max_number_of_redescending_iterations: int = 2,
        r0: float = 1.4,
        u0: float = 2.8,
        tolerance: float = 0.0005,
        verbosity: int = 0,
    ) -> None:
        """
        Constructor

        Parameters
        ----------
        max_number_of_iterations: int
            Set to zero for OLS, otherwise, this is how many times the RME
            will refine the estimate.
        max_number_of_redescending_iterations : int
            1 or 2 is fine at most.  If set to zero we ignore the redescend code block.
        r0: float
            Effectively infinty for OLS, this controls the point at which residuals
            transition from being penalized by a squared vs a linear function.  The
            units are standard deviations of the residual distribution.  Normally
            1.4, 1.5 or thereabouts
        u0: float
            u0 is a parameter for the redescending portion of the robust regression.
            It is controlled by a double exponential formula (REFERENCE NEEDED).  It
            makes for severe downweighting about u0.  The function is continuous
            "math friendly" (all derivates exist etc) so you can prove theorems about it
            etc.
        tolerance : float
            minimum fractional change in any term in b.  Any smaller change
            than this will trigger convergence
        verbosity: int
            Allows setting of custom logging messages in development.

        Development notes:
        There was originally a parameter called epsilon (float) but it was not used.

        """
        self._number_of_iterations = 0
        self._number_of_redescending_iterations = 0

        self.tolerance = tolerance
        self.max_number_of_iterations = max_number_of_iterations
        self.max_number_of_redescending_iterations = (
            max_number_of_redescending_iterations
        )

        # regression-M estimator params
        self.r0 = r0
        self.u0 = u0

        self.verbosity = verbosity
        # Additional properties
        # used sometimes to control one or another of the iterative algorithms
        # These were translated from the matlab code and may be moved in future
        self.return_covariance = True  # TODO: add functionality
        self.save_cleaned = False  # TODO: add functionality
        self.robust_diagonalize = False  # TODO: add functionality

    @property
    def number_of_iterations(self) -> int:
        return self._number_of_iterations

    # @number_of_iterations.setter
    # def number_of_iterations(self, value) -> int:
    #     self._number_of_iterations = value

    def reset_number_of_iterations(self) -> int:
        self._number_of_iterations = 0

    def increment_iteration_number(self):
        self._number_of_iterations += 1

    @property
    def number_of_redescending_iterations(self) -> int:
        return self._number_of_redescending_iterations

    def reset_number_of_redescending_iterations(self):
        self._number_of_redescending_iterations = 0

    def increment_redescending_iteration_number(self):
        self._number_of_redescending_iterations += 1

    @property
    def max_iterations_reached(self) -> bool:
        """
        Return True of the number of iterations carried out is greater or equal the
        maximum number of iterations set in the processing config
        """
        return self.number_of_iterations >= self.max_number_of_iterations

    def converged(self, b, b0):
        """
        Parameters
        ----------
        b : complex-valued numpy array
            the most recent regression estimate
        b0 : complex-valued numpy array
            The previous regression estimate
        verbose: bool
            Set to True for debugging

        Returns
        -------
        converged: bool
            True of the regression has terminated, False otherwise

        Notes:
        The variable maximum_change finds the maximum amplitude component of the vector
        1-b/b0.  Looking at the formula, one might want to cast this instead as
        1 - abs(b/b0), however, that will be insensitive to phase changes in b,
        which is complex valued.  The way it is coded np.max(np.abs(1 - b / b0)) is
        correct as it stands.

        """

        maximum_change = np.max(np.abs(1 - b / b0))
        tolerance_cond = maximum_change <= self.tolerance
        iteration_cond = self.max_iterations_reached
        if tolerance_cond or iteration_cond:
            converged = True
            if self.verbosity > 0:
                msg_start = "Converged due to"
                msg_end = (
                    f"{self.number_of_iterations} of "
                    f"{self.max_number_of_iterations} iterations"
                )
                if tolerance_cond:
                    msg = f"{msg_start} MaxChange < Tolerance after {msg_end}"
                elif iteration_cond:
                    msg = f"{msg_start} maximum number_of_iterations {msg_end}"
                logger.info(msg)
        else:
            converged = False

        return converged

    @property
    def continue_redescending(self):
        maxxed_out = (
            self.number_of_redescending_iterations
            >= self.max_number_of_redescending_iterations
        )
        if maxxed_out:
            return False
        else:
            return True

    @property
    def correction_factor(self):
        """
        Returns correction factor for residual variances.

        Detailed notes on usage in
        transfer_function.regression.helper_functions.rme_beta

        TODO: This is an RME specific property.  Suggest move r0, u0 and this method
         into an RME-config class.

        Returns
        -------
        correction_factor : float
            correction factor used for scaling the residual error_variance
        """
        correction_factor = 1.0 / rme_beta(self.r0)
        return correction_factor
