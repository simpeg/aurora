"""
follows Gary's IterControl.m in
iris_mt_scratch/egbert_codes-20210121T193218Z-001/egbert_codes/matlabPrototype_10-13-20/TF/classes
"""
import numpy as np


class IterControl(object):
    """ """

    def __init__(
        self,
        max_number_of_iterations=10,
        max_number_of_redescending_iterations=2,
        **kwargs,
    ):
        """

        Parameters
        ----------
        max_number_of_iterations: int
            Set to zero for OLS, otherwise, this is how many times the RME
            will refine the estimate.
        max_number_of_redescending_iterations : int
            1 or 2 is fine at most.  If set to zero we ignore the redescend code block.
        tolerance : float
            minimum fractional change in any term in b.  Any smaller change
            than this will trigger convergence
        epsilon : float
            NOT USED: REMOVE
        kwargs

        Class Variables:
        <Specific to Egbert's Robust Regression>
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
        </Specific to Egbert's Robust Regression>
        """
        self.number_of_iterations = 0
        self.number_of_redescending_iterations = 0

        self.tolerance = 0.005
        self.epsilon = 1000
        self.max_number_of_iterations = max_number_of_iterations
        self.max_number_of_redescending_iterations = (
            max_number_of_redescending_iterations
        )

        # <regression-M estimator params>
        self.r0 = 1.5
        self.u0 = 2.8
        # </regression-M estimator params>

        # <Additional properties>
        # used sometimes to control one or another of the iterative algorithms
        # These were translated from the matlab code and may be moved in future
        self.return_covariance = True
        self.save_cleaned = False
        self.robust_diagonalize = False
        # </Additional properties>

    def converged(self, b, b0):
        """
        Parameters
        ----------
        b : complex-valued numpy array
            the most recent regression estimate
        b0 : complex-valued numpy array
            The previous regression estimate

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

        converged = False
        maximum_change = np.max(np.abs(1 - b / b0))
        tolerance_cond = maximum_change <= self.tolerance
        iteration_cond = self.number_of_iterations >= self.max_number_of_iterations
        if tolerance_cond or iteration_cond:
            converged = True
            if tolerance_cond:
                print(
                    f"Converged Due to MaxChange < Tolerance after "
                    f" {self.number_of_iterations} of "
                    f" {self.max_number_of_iterations} iterations"
                )
            elif iteration_cond:
                print(
                    f"Converged Due to maximum number_of_iterations "
                    f" {self.max_number_of_iterations}"
                )
        else:
            converged = False

        return converged

    @property
    def continue_redescending(self):
        maxxed_out = (
            self.number_of_redescending_iterations
            <= self.max_number_of_redescending_iterations
        )
        if maxxed_out:
            return False
        else:
            return True

    @property
    def correction_factor(self):
        """
        TODO: This is an RME specific property.  Suggest move r0, u0 and this method
        into an RME-config class.
        TODO: Note that IterControl itself should probably be factored.
        A base class can be responsible for iteration_watcher and convergence checks
        etc.  But u0, and r0 are specific to the Robust methods.

        In the regression esimtate you downweight things with large errors, but
        you need to define what's large.  You estimate the standard devation
        (sigma) of the errors from the residuals BUT with this cleaned data
        approach (Yc) sigma is smaller than it should be, you need to
        compensate for this by using a correction_factor. It's basically the
        expectation, if the data really were Gaussian, and you estimated from
        the corrected data. This is how much too small the estimate would be.

        If you change the penalty functional you may need a pencil, paper and
        some calculus.  The relationship between the corrected-data-residuals
        and the gaussin residauls could change if you change the penalty

        Returns
        -------
        cfac : float
            correction factor used when
        """
        cfac = 1.0 / (2 * (1.0 - (1.0 + self.r0) * np.exp(-self.r0)))
        # cfac = 1. / (1. - np.exp(-self.r0));
        return cfac
