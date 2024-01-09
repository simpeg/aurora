"""
There are several ways to go about this.  One reasonably straightforward one is:

Take all the Fourier coefficients right before they go into regression (right before or
after we apply the effective degrees of freedom weights).  Data are strictly 2D at
this point, you have each channel of data corresponding to a column, and each row
to an observation.
Following Jones and Jodicke, lets take the coherence of Ex and Hy
and that of Ey and Hx.  This yields one single number.

Then, iteratively remove each observation in turn and count whether that partial
coherence is bigger or smaller than the baseline and by how much

Deflection downward means naughty data and deflection upward means well-behaved data.

This leads to a coherence score for each observation.  We can boot out all data that
 say move the estimate down (worse than average)
"""

import numpy as np
from loguru import logger


def drop_column(x, i_col):
    """

    Parameters
    ----------
    x
    i_drop

    Returns
    -------

    """
    return np.hstack([x[:, 0:i_col], x[:, i_col + 1 : :]])


def coherence_from_fc_series(c_xy, c_xx, c_yy):
    n_obs = len(c_xy)
    num = np.sum(c_xy) ** 2
    den = np.sum(c_xx * c_yy)
    coh = (num / den) / n_obs
    return coh


def jackknife_coherence_weights(x, y, keep_fraction=0.95):  # 975):#0.98
    """
    Note 1: Extremely high coherence can be due to noise
    Consider ways to pre-filter those events before this is called.
    - That may need a min_fraction_to_keep=0.2-0.8, which guards against
    generally poor data being completely discarded.
    - Consider variations on coh-sorting threshold_min, and threshold_max
    near perfect coherency can be expected when coherent noise is present
    - 95% seems extreme for a threshold, maybe not yielding enogh
    observations for robust statistics.


    Parameters
    ----------
    x: X["hx"].data
    y: Y["ey"].data
    keep_fraction : value in (0,1) to set what to keep / reject.
    A value of 0.1 would keep 10% of the estimates, a val

    Returns
    -------
    W: Weights (currently set to be 0 or 1)
    """
    # Initialize a weight vector the length = num_observations
    n_obs = len(x)
    clip_fraction = 1 - keep_fraction
    n_clip = int(clip_fraction * n_obs)
    logger.info(f"removing worst {n_clip} of {n_obs} via jackknife coherence")

    partial_coh = np.zeros(n_obs)
    W = np.zeros(n_obs)  # for example

    c_xy = np.abs(x * np.conj(y))
    c_xx = np.real(x * np.conj(x))
    c_yy = np.real(y * np.conj(y))

    ccc = np.vstack([c_xy, c_xx, c_yy])

    for i in range(n_obs):
        partial_series = drop_column(ccc, i)
        partial_coh[i] = coherence_from_fc_series(
            partial_series[0, :],
            partial_series[1, :],
            partial_series[2, :],
        )

    worst_to_best = np.argsort(partial_coh)
    keepers = worst_to_best[n_clip:]
    W[keepers] = 1
    return W


def coherence_weights_jj84(X, Y, RR, coh_type="local"):
    """
    This method loosely follows Jones and Jodicke 1984, at least inasfaras it uses a jackknife process.
    However, the termination criteria for when to stop rejecting is given by a simple fraction of data to keep/reject
    This is not what is done in JJ84. They use an SNR maximizing method, that could be added here.

    The variables remote_threshold and local_threshold DO NOT refer to thresholds on the value of coherence!
    Rather, the threshold refers to the fraction of the estimates to keep (1 - fraction to reject)

    2022-09-09: This method is not yet supported.  It needs to be made tolerant of channel_nomenclature.  A not
    unreasonable solution to this would be to make a decorator:
    def use_channel_nomenclature():
        Loops over input xarrays and maps channel names from current nomenclature
        to hexy, and then, when it is done executing, maps the labels back.
    However, it would cloud a lot of the math and readability if this was done all the time.  Suggest decorating
    process_transfer_function


    Parameters
    ----------
    X
    Y
    RR
    coh_type: "local" or "remote"

    Returns
    -------

    """
    # these should be params in the config
    remote_keep_fraction = 0.8
    local_keep_fraction = 0.95

    # redundant - these should already be dropped
    X = X.dropna(dim="observation")
    Y = Y.dropna(dim="observation")
    if RR is not None:
        RR = RR.dropna(dim="observation")

    null_indices = X["hx"].isnull()
    finite_indices = ~null_indices
    W = np.zeros(len(X.observation))

    x = X["hx"]
    if coh_type == "local":
        y = Y["ey"]
        keep_fraction = local_keep_fraction
    elif coh_type == "remote":
        y = RR["hx"]
        keep_fraction = remote_keep_fraction

    W1 = jackknife_coherence_weights(x, y, keep_fraction=keep_fraction)

    W[finite_indices] = W1

    W[W == 0] = np.nan
    X["hx"].data *= W
    Y["ey"].data *= W

    # x = X["hy"].data
    # y = Y["ex"].data
    W = np.zeros(len(finite_indices))
    x = X["hy"]
    if coh_type == "local":
        y = Y["ex"]
        keep_fraction = local_keep_fraction
    elif coh_type == "remote":
        y = RR["hy"]
        keep_fraction = remote_keep_fraction
    W2 = jackknife_coherence_weights(x, y, keep_fraction=keep_fraction)
    W[finite_indices] = W2
    W[W == 0] = np.nan
    X["hy"].data *= W
    Y["ex"].data *= W
    if RR is not None:
        RR *= W
    return X, Y, RR
