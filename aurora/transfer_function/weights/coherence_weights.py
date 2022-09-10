"""
There are several ways to go about this.  One reasonably straightforward one is:

Take all the Fourier coefficients right before the go into regression (right before or
after we apply the effective degrees of freedom weights).  Data are strictly 2D at
this point, you have each channel of data corresponding to a column, and each row
to an observation.
Following Jones and Jodicke, lets take the coherence of Ex and Hy
and that of Ey and Hx.  This yields one single number.

Then, iteratively remove each observation in turn and count whether that partial
cohernece is bigger or smaller than the baseline and by how much

Deflection downward means naughty data and defelction  upward means well-behaved data.

This leads to a coherence score for each obervation.  We can boot out all data that
 say move the estimate down (worse than average)
"""

import numpy as np


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


def coherence_weights_v00(x, y, threshold=0.95):  # 975):#0.98
    """

    Parameters
    ----------
    x: X["hx"].data
    y: Y["ey"].data
    threshold : value in [0,1] to set what to keep / reject

    Returns
    -------
    W: Weights (currently set to be 0 or 1)
    """
    # Initialize a weight vector the length = num_observations
    n_obs = len(x)
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
    # coh0 = coherence_from_fc_series(c_xy, c_xx, c_yy)
    # relative_coherence = partial_coh / coh0
    # sorted_partials = partial_coh[worst_to_best]
    clip_point = int(threshold * n_obs)
    keepers = worst_to_best[clip_point:]
    W[keepers] = 1
    return W


def compute_coherence_weights(X, Y, RR, coh_type="local"):
    """
    2022-09-09: This method is not yet supported.  It needs to be made
    tolerant of channel_nomenclature.

    Parameters
    ----------
    X
    Y
    RR
    coh_type: "local" or "remote"

    Returns
    -------

    """
    remote_threshold = 0.8
    local_threshold = 0.95

    X = X.dropna(dim="observation")
    Y = Y.dropna(dim="observation")
    if RR is not None:
        RR = RR.dropna(dim="observation")

    # < INSERT COHERENCE SORTING HERE>    y_type = "remote"
    null_indices = X["hx"].isnull()  # not robust -- hail mary
    finite_indices = ~null_indices
    W = np.zeros(len(X.observation))

    x = X["hx"]  # .dropna(dim="observation").data

    if coh_type == "local":
        y = Y["ey"]  # .dropna(dim="observation").data
        threshold = local_threshold
    elif coh_type == "remote":
        y = RR["hx"]  # .dropna(dim="observation").data
        threshold = remote_threshold

    W1 = coherence_weights_v00(x, y, threshold=threshold)

    W[finite_indices] = W1

    W[W == 0] = np.nan
    X["hx"].data *= W
    Y["ey"].data *= W

    # x = X["hy"].data
    # y = Y["ex"].data
    W = np.zeros(len(finite_indices))
    x = X["hy"]  # .dropna(dim="observation").data
    if coh_type == "local":
        y = Y["ex"]  # .dropna(dim="observation").data
        threshold = local_threshold
    elif coh_type == "remote":
        y = RR["hy"]  # .dropna(dim="observation").data
        threshold = remote_threshold
    W2 = coherence_weights_v00(x, y)
    W[finite_indices] = W2
    W[W == 0] = np.nan
    X["hy"].data *= W
    Y["ex"].data *= W
    # W = W*W2
    # X *= W
    # Y *= W
    if RR is not None:
        RR *= W
    return X, Y, RR
