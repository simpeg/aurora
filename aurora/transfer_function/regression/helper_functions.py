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


def simple_solve_tf(Y, X, R=None):
    """
    Cast problem Y = Xb into scipy.linalg.solve form which solves: a @ x = b
    - Note that the "b" in the two equations is different.
      - The EMTF regression problem (and many regression problems in general) use Y=Xb
      - Y, X are known and b is the solution
      - scipy.linalg.solve form which solves: a @ x = b
      - Here a, b are known and x is the solution.
    - This function is used for testing vectorized, direct solver.


    Parameters
    ----------
    Y: numpy.ndarray
        The "output" of regression problem.  For testing this often has shape (N,).
        Y is normally a vector of Electric field FCs
    X: numpy.ndarray
        The "input" of regression problem.  For testing this often has shape (N,2).
        X is normally an array of Magnetic field FCs (two-component)
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


def direct_solve_tf(Y, X, R=None):
    """
    Cast problem Y = Xb

    Parameters
    ----------
    Y: numpy.ndarray
        The "output" of regression problem.  For testing this often has shape (N,).
        Y is normally a vector of Electric field FCs
    X: numpy.ndarray
        The "input" of regression problem.  For testing this often has shape (N,2).
        X is normally an array of Magnetic field FCs (two-component)
    R: numpy.ndarray or None
        Remote reference channels (optional)

    Returns
    -------
    z: numpy.ndarray
        The TF estimate -- Usually two complex numbers, (Zxx, Zxy), or (Zyx, Zyy)
    Parameters
    ----------
    Y
    X
    R

    Returns
    -------

    """
    if R is None:
        xH = X.conjugate().transpose()
    else:
        xH = R.conjugate().transpose()

    # multiply both sides by conjugate transpose
    xHx = xH @ X
    xHY = xH @ Y

    # Invert the square matrix
    inv = np.linalg.inv(xHx)

    # get solution
    b = inv @ xHY
    return b
