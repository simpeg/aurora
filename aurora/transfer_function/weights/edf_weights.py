"""
from egbert_codes-20210121T193218Z-001/egbert_codes/matlabPrototype_10-13-20/TF
/functions/Edfwts.m
"""

import numpy as np


class EffectiveDegreesOfFreedom(object):
    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        kwargs
        n_observations : integer
        Notes number of "numeric" observations in the input and remote reference
        data.  Any time-frequency index having nan value in any channel is subtracted
        from the total number of observations before computation of p1
        """

        self.edfl1 = kwargs.get("edf_l1", 20.0)
        self.alpha = kwargs.get("alpha", 0.5)
        self.c1 = kwargs.get("c1", 2.0)
        self.c2 = kwargs.get("c2", 10.0)
        self.p3 = kwargs.get("p3", 5)
        self.n_data = kwargs.get("n_data", 0)

    @property
    def p1(self):
        """
        Adhoc parameter - needs documenation describing it
        """
        return self.c1 * (self.n_data ** self.alpha)

    @property
    def p2(self):
        """
        Adhoc parameter - needs documenation describing it
        """
        return self.c2 * (self.n_data ** self.alpha)

    def compute_weights(self, X, use):
        """
        The data covariance matrix s and its inverse h are iteratively recomputed using
        fewer and fewer observations. However, the edf is also computed at every
        iteration but doesn't seem to use any fewer observations.
        Thus the edf weights change as use drops, EVEN FOR INDICES that were
        previously computed... this seems like it could be an error.

        Let's assume its correct for now and program it.  We need a boolean (use) to
        select the data.

        Parameters
        ----------
        X
        use : boolean array

        Returns
        -------
        The "HAT" Matrix, where the diagonals of this matrix are really big it means an
        individual data point is controlling its own prediction, and the estimate.
        If the problem was balanced, each data point would contribute equally to the
        estimate.  Each data point shoudl contribute 1/N to each parameter. When one
        data point is large and the others are tiny, then it may be contributing a
        lot, say 1/2 rather than 1/n.

        edf is like the diagonal of the Hat matrix (in the single station case)
        How much does the data point contribute to the prediction of itself.
        If there are n data points contributing equally, each datapoint
        #should contribute ~1/n to its prediction

        Note: If using h[2,1] feels unbalanced because it feels like there should be an
        h(1,2) term in there,  the inverse of s does in general have h(2,1) = h(1,2).
         ...An interesting property of covariance matrices., or at a minimum, 2x2
         matirces with matching off-diagonal terms have an inverse with the same
        property it seems.


        """
        s = X[:, use] @ np.conj(X[:, use]).T  # covariance matrix, 2x2
        s /= sum(use)
        # covariance matrix, 2x2
        h = np.linalg.inv(s)
        # invert the 2x2
        xx_term = np.real(X[0, :] * np.conj(X[0, :]) * h[0, 0])
        yy_term = np.real(X[1, :] * np.conj(X[1, :]) * h[1, 1])
        xy_term = 2 * np.real(np.conj(X[1, :]) * X[0, :] * h[1, 0])
        edf = xx_term + yy_term + xy_term
        return edf


def effective_degrees_of_freedom_weights(X, R, edf_obj=None, test=True):
    """
    Emulates edfwts ("effective dof") from tranmt.  Based on Edfwts.m matlab code from
    iris_mt_scratch/egbert_codes-20210121T193218Z-001/egbert_codes/matlabPrototype_10-13-20/TF/functions/

    Flow:
    0. Initialize weights vector (of 1's) the length of the "observation" axis
    1. Remove any nan in X,R
    2. compute the weights on the reduced (no-nan) arrays of X and R
    3. Overwrite the weights vector for the non-nan entries.
    4 Return the weights vector.  We then broadcast multiply that array against the
    data, to apply the weights.

    This follows the matlab code by using a boolean index vector.  An xarray
    implementaiton which uses the observation dimension of X, R was started but never
    finished.  Here are the breadcrumbs for that method:
    0. Create weights:
    import xarray as xr
    X = X.assign(weights=lambda x: X.frequency*0+1.0)
    weights = X["weights"]
    X = X[["hx","hy"]]
    1. Then to drop nan:
    from aurora.time_series.xarray_helpers import handle_nan
    X, Y, RR = handle_nan(XX, None, R, drop_dim="observation")
    2. Same as below
    3. Overwrite: Not sure, it wasn't clear how to assign them. Possibly the syntax
    with multi-index was confusing me when trying to assign.
    Accessing the weights was done by:
    weights.sel(observation=X.observation)
    But the assignment syntax I'm not sure about.  Could also be that I need to cast
    as dataarray for the assignement step.

    Things to review:
    -Why is the Remote reference weighting not done with a while
    loop like the input channels are??
    - since we assign zero-weights to Nan, we could probably remove the keep_indices
    methods and simply assign nan or zero weights to those data up front. Since the
    "use" boolean selects data before computiation are performed on X, R we should
    never get nans in the computed edfs, but should simplify the code somewhat.

    Parameters
    ----------
    X : xr.Dataset
        The input channels for regression.  Usually horizontal magnetics hx, hy
    R : xr.Dataset or None
        The remote reference channels, usually hx, hy.  Can be None if single-station
        processing
    EDFparam : EffectiveDegreesOfFreedom
        Object with parameters for Gary's adhoc edfwts method.

    Returns
     weights : numpy array
        Weights for reducing leverage points.
    -------

    """
    num_channels = len(X.data_vars)
    if num_channels != 2:
        print("edfwts only works for 2 input channels")
        raise Exception
    X = X.to_array(dim="channel")
    if R is not None:
        R = R.to_array(dim="channel")

    if test:
        pass
        # add_nan - this syntax only works for dataarray
        # X[0,10] *= 2e1
        # X[0,1] *= np.nan
        # # X[0,-3] *= np.nan
        # # try:
        # #     X[0,-13] *= 1e13#np.nan
        # # except IndexError:
        # #     pass
        # # if R is not None:
        # #     R[1,2:11] *= np.nan

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

    # %    determine intial robust B-field cross-power matrix; this just uses
    # %    edfl1 -- cut off for estimating robust local magnetic covariance
    nOmit = n_observations_numeric
    use = np.ones(n_observations_numeric, dtype=bool)
    # initialize use as a boolean of True, as large as XX
    n_valid_observations = n_observations_numeric
    # use never changes length, it only flips its bit
    while nOmit > 0:
        eff_deg_of_freedom = edf_obj.compute_weights(XX, use)
        use = eff_deg_of_freedom <= edf_obj.edfl1  # update "use" boolean selector
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
