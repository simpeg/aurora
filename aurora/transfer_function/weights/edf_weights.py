"""
    This module contains a class for computing so-called "Effective Degrees of Freedom" weights.

Development notes:
The code here is based on the function Edfwts.m from egbert_codes-
20210121T193218Z-001/egbert_codes/matlabPrototype_10-13-20/TF/functions/Edfwts.m

"""

import numpy as np
import xarray as xr
from loguru import logger
from typing import Optional, Union
import aurora.transfer_function.weights.edf_weights


class EffectiveDegreesOfFreedom(object):
    def __init__(
        self,
        edf_l1: Optional[float] = 20.0,
        alpha: Optional[float] = 0.5,
        c1: Optional[float] = 2.0,
        c2: Optional[float] = 10.0,
        p3: Optional[float] = 5.0,
        n_data: Optional[int] = 0,
    ):
        """
        Constructor.

        Parameters
        ----------
        n_data : int
            Number of "numeric" observations in the input and remote reference data.
            Any time-frequency index having nan value in any channel is subtracted
            from the total number of observations before computation of p1, p2
        edf_l1 : float
             the effective number of degrees of freedom at which a datapoint should be
              removed from the total energy calculation in the iterative loop
        alpha : float
            The exponent on n_data in formulae for thresholds p1 and p2
        c1 : float
            ad hoc parameter for scaling the p1 threshold
        c2 : float
            ad hoc parameter for scaling the p2 threshold
        p3 : float
            parameter controlling threshold at which the ratio of a data weighting
            associated with remote and input channel is unacceptable.
        """

        self.edf_l1 = edf_l1
        self.alpha = alpha
        self.c1 = c1
        self.c2 = c2
        self.p3 = p3
        self.n_data = n_data

    @property
    def p1(self) -> float:
        """
        Threshold applied to edf.  All edf below  this value
        are set to weight=0
        """
        return self.c1 * (self.n_data**self.alpha)

    @property
    def p2(self) -> float:
        """
        Threshold applied to edf.  All edf above th  this value
        are set to weight=0
        """
        return self.c2 * (self.n_data**self.alpha)

    def compute_weights(self, X: np.ndarray, use: np.ndarray) -> np.ndarray:
        """
        Compute the EDF Weights

        Development Notes:
        The data covariance matrix s and its inverse h are iteratively recomputed using
        fewer and fewer observations. However, the edf is also computed at every
        iteration but doesn't seem to use any fewer observations.
        Thus the edf weights change as use drops, even for indices that were
        previously computed ... TODO: Could that be an error?

        Discussing this with Gary:
        "... because you are down-weighting (omitting) more and more highpower events
        the total signal is going down.  The signal power goes down with every call
        to this method"
        ...
        "The "HAT" Matrix, where the diagonals of this matrix are really big it means an
        individual data point is controlling its own prediction, and the estimate.
        If the problem was balanced, each data point would contribute equally to the
        estimate.  Each data point should contribute 1/N to each parameter. When one
        data point is large and the others are tiny, then it may be contributing a
        lot, say 1/2 rather than 1/n.
        edf is like the diagonal of the Hat matrix (in the single station case)
        How much does the data point contribute to the prediction of itself.
        If there are n data points contributing equally, each datapoint
        should contribute ~1/n to its prediction
        - Note: H = inv(S) in general has equal H[0,1] = H[1,0];  2x2 matrices with
        matching off-diagonal terms have inverses with the same property.

        Parameters
        ----------
        X: np.ndarray
            The data to for which to determine weights.
        use : np.ndarray
            popolated with booleans

        Returns
        -------
        edf:np.ndarray
            The weights values.

        """
        S = X[:, use] @ np.conj(X[:, use]).T  # covariance matrix, 2x2
        S /= sum(use)  # normalize by the number of datapoints
        H = np.linalg.inv(S)  # inverse covariance matrix

        xx_term = np.real(X[0, :] * np.conj(X[0, :]) * H[0, 0])
        yy_term = np.real(X[1, :] * np.conj(X[1, :]) * H[1, 1])
        xy_term = 2 * np.real(np.conj(X[1, :]) * X[0, :] * H[1, 0])  # real or abs?
        edf = xx_term + yy_term + xy_term
        return edf


def effective_degrees_of_freedom_weights(
    X: xr.Dataset,
    R: Union[xr.Dataset, None],
    edf_obj=None,
) -> np.ndarray:
    """
    Computes the effective degrees of freedom weights. Emulates edfwts ("effective dof") from tranmt.
    - Based on Edfwts.m matlab code from
    iris_mt_scratch/egbert_codes-20210121T193218Z-001/egbert_codes/matlabPrototype_10-13-20/TF/functions/

    Flow:
    0. Initialize weights vector (of 1's) the length of the "observation" axis
    1. Remove any nan in X,R
    2. compute the weights on the reduced (no-nan) arrays of X and R
    3. Overwrite the weights vector for the non-nan entries.
    4. Return weights and broadcast-multiply against data to apply.

    Development Notes:
    Note about the while loop: variable "use" never changes length, it only
    flips its bit.  The while loop exits when n_valid_observations == sum(use)
    i.e.the effective dof are all below threshold Estimate dof.  Then we "use"
    only points whose dof are smaller than the threshold.  Then we recompute dof.
    This time the covariance matrix diagonals are smaller, there is less energy
    in the time series for the S, H calculation.

    TODO:
     This follows the matlab code by using a boolean index vector.  An xarray
     implementation which uses the observation dimension of X, R was started but never
     finished.  Here are the breadcrumbs for the xarray method:
     0. Create weights:
     import xarray as xr
     X = X.assign(weights=lambda x: X.frequency*0+1.0)
     weights = X["weights"]
     X = X[["hx","hy"]]
     1. Then to drop nan:
     from aurora.time_series.xarray_helpers import handle_nan
     X, Y, RR = handle_nan(XX, None, R, drop_dim="observation")
     2. Same as below
     3. Overwrite: Got stuck here.
     Accessing the weights was done by:
     weights.sel(observation=X.observation)
     Assignment should be something like weights.loc[x.observation]
     May need cast as dataarray for the assignment step.

    TODO:
     Why is the Remote reference weighting not done with a while loop?  Maybe
     an oversight in the matlab codes.

    TODO:
     Since zero-weights assigned to Nan, could probably remove the keep_indices
     methods and simply assign nan or zero weights to those data up front. Since the
     "use" boolean selects data before computation are performed on X, R we should
     never get nans in the computed edfs, but should simplify the code somewhat.

    TODO:
     Add tests for data having Nan values

    Parameters
    ----------
    X : xr.Dataset
        The input channels for regression.  Usually horizontal magnetics hx, hy
    R : xr.Dataset or None
        The remote reference channels, usually hx, hy.  Can be None if single-station
        processing
    edf_obj : aurora.transfer_function.weights.edf_weights.EffectiveDegreesOfFreedom
        Object with parameters for Gary's adhoc edfwts method.

    Returns
    -------
    weights : numpy.ndarray
        Weights for reducing leverage points.

    """

    num_channels = len(X.data_vars)
    if num_channels != 2:
        logger.error("edfwts only works for 2 input channels")
        raise Exception
    X = X.to_array(dim="channel")
    if R is not None:
        R = R.to_array(dim="channel")

    n_observations_initial = len(X.observation)
    weights = np.ones(n_observations_initial)

    # reduce the data to only valid (non-nan) observations
    if R is not None:
        keep_x_indices = ~np.isnan(X.data).any(axis=0)
        keep_r_indices = ~np.isnan(R.data).any(axis=0)
        keep_indices = keep_r_indices & keep_x_indices
        RR = R.data[:, keep_indices]
    else:
        keep_indices = ~np.isnan(X.data).any(axis=0)
    n_observations_numeric = keep_indices.sum()
    XX = X.data[:, keep_indices]

    if edf_obj is None:
        edf_obj = EffectiveDegreesOfFreedom(n_data=n_observations_numeric)

    # %    determine initial robust B-field cross-power matrix; this just uses
    # %    edf_l1 -- cut off for estimating robust local magnetic covariance
    nOmit = n_observations_numeric
    use = np.ones(n_observations_numeric, dtype=bool)
    # initialize use as a boolean of True, as large as XX
    n_valid_observations = n_observations_numeric

    while nOmit > 0:
        eff_deg_of_freedom = edf_obj.compute_weights(XX, use)
        use = eff_deg_of_freedom <= edf_obj.edf_l1  # update "use" boolean selector
        nOmit = n_valid_observations - sum(use)
        n_valid_observations = sum(use)

    wt = np.ones(n_observations_numeric)
    wt[eff_deg_of_freedom > edf_obj.p2] = 0
    cond = (eff_deg_of_freedom <= edf_obj.p2) & (eff_deg_of_freedom > edf_obj.p1)
    wt[cond] = np.sqrt(edf_obj.p1 / eff_deg_of_freedom[cond])

    if R is not None:
        # now find additional segments with crazy remotes
        wtRef = np.ones(n_observations_numeric)
        ref_use = np.ones(n_observations_numeric, dtype=bool)
        edf_ref = edf_obj.compute_weights(RR, ref_use)

        wtRef[edf_ref > edf_obj.p2] = 0
        cond = (edf_ref <= edf_obj.p2) & (edf_ref > edf_obj.p1)
        wtRef[cond] = np.sqrt(edf_obj.p1 / edf_ref[cond])

        # are either of the weights very different
        cond1 = wtRef / wt > edf_obj.p3
        cond2 = wt / wtRef > edf_obj.p3
        differentAmp = cond1 | cond2
        wt = wt * wtRef
        wt[differentAmp] = 0

    weights[keep_indices] = wt
    weights[weights == 0] *= np.nan
    return weights
