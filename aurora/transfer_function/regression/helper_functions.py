import numpy as np


def rme_beta(r0):
    """
    This is an RME specific property.
    It represents a bias in the calculation of residual_variance
    which we correct for in TRME and TRME_RR.

    The implemented formula is an approximation.  This is approximately equal to 1/beta
    where beta is defined by Equation A3 in Egbert & Booker 1986.

    for r0=1.5  the correction factor is ~1.28 and beta ~0.78

    Some notes from a discussion with Gary:
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

    Parameters
    ----------
    r0 : float
        The number of standard errors at which the RME method transitions from
        quadratic weighting to linear

    Returns
    -------
    beta : float
        correction factor = 1/beta
    """
    beta = 1.0 - np.exp(-r0)
    return beta


def solve_single_time_window(Y, X, R=None):
    """
    Cast problem Y = Xb into scipy.linalg.solve form which solves: a @ x = b
    - This function is used for testing vectorized, direct solver.


    Parameters
    ----------
    Y: numpy.ndarray
        The "output" of regression problem.  For testing this often has shape (2,).
    X: numpy.ndarray
        The "input" of regression problem.  For testing this often has shape (2,2).
    R: numpy.ndarray or None
        Remote reference channels (optional)

    Returns
    -------
    z: numpy.ndarray
        The TF estimate -- Usually two complex numbers, (Zxx, Zxy), or (Zyx, Zyy)

    """
    if R is None:
        xH = X.conjugate().transpose()
    else:
        xH = R.conjugate().transpose()
    a = xH @ X
    b = xH @ Y
    z = np.linalg.solve(a, b)
    return z
