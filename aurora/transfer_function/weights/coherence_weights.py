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
    # n_obs = len(c_xy)
    num = np.sum(c_xy) ** 2
    den = np.sum(c_xx) * np.sum(c_yy)
    coh = num / den
    return coh


def jackknife_coherence_weights(
    x, y, lower_quantile_cutoff=0.3, upper_quantile_cutoff=0.95
):  # 975):#0.98
    """

    Note 1: Extremely high coherence can be due to noise
    Consider ways to pre-filter those events before this is called.
    - That may need a min_fraction_to_keep=0.2-0.8, which guards against
    generally poor data being completely discarded.
    - Consider variations on coh-sorting threshold_min, and threshold_max
    near perfect coherency can be expected when coherent noise is present
    - 95% seems extreme for a threshold, maybe not yielding enogh
    observations for robust statistics.

    Note2: 2024-09-10: vectorization
    Instead of repeatedly dropping 1 value from Cxy, Cxx, Cyy, we could try the following:
    Make a "Jackknife" matrix J that is all ones but zero on the diagonal
    Then we can just take the inner product of the crosspower with the nth row of J to
    get the sum of all cross-powers but the nth... or multiply the crosspower vector J
    to get the partial sums of all crosspowers except the nth element.
    First of all, we will follow the convention of Bendat & Perisol (1980):
    "The two-sided spectral density function between two random processes is defined
    using X*Y and _not_ XY*.


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
    n_obs = len(x)
    n_clip_low = int(lower_quantile_cutoff * n_obs)
    n_clip_high = int(upper_quantile_cutoff * n_obs)
    logger.info(f"removing worst {n_clip_low} of {n_obs} via jackknife coherence")
    logger.info(
        f"removing 'best' {n_obs - n_clip_high} of {n_obs} via jackknife coherence"
    )

    # Initialize a weight vector the length = num_observations
    partial_coh = np.zeros(n_obs)
    W = np.zeros(n_obs)  # for example

    c_xy = np.abs(x.conj() * y)
    c_xx = np.real(x.conj() * x)
    c_yy = np.real(y.conj() * y)

    # "Jackknife" matrix; all ones but zero on the diagonal
    J = np.ones((n_obs, n_obs)) - np.eye(n_obs)
    ccc = np.vstack([c_xy, c_xx, c_yy])

    for i in range(n_obs):
        partial_series = drop_column(ccc, i)
        partial_coh[i] = coherence_from_fc_series(
            partial_series[0, :],
            partial_series[1, :],
            partial_series[2, :],
        )

    p_xx = J @ c_xx.data
    p_xy = J @ c_xy.data
    p_yy = J @ c_yy.data
    pc = p_xy**2 / (p_xx * p_yy)
    assert np.isclose(pc, partial_coh).all()
    worst_to_best = np.argsort(partial_coh)
    keepers = worst_to_best[n_clip_low:n_clip_high]
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
    local_lower_quantile_cutoff = 0.5
    local_upper_quantile_cutoff = 0.9
    remote_lower_quantile_cutoff = 0.3
    remote_upper_quantile_cutoff = 0.99

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
        lower_quantile_cutoff = local_lower_quantile_cutoff
        upper_quantile_cutoff = local_upper_quantile_cutoff
    elif coh_type == "remote":
        y = RR["hx"]
        lower_quantile_cutoff = remote_lower_quantile_cutoff
        upper_quantile_cutoff = remote_upper_quantile_cutoff

    W1 = jackknife_coherence_weights(
        x,
        y,
        lower_quantile_cutoff=lower_quantile_cutoff,
        upper_quantile_cutoff=upper_quantile_cutoff,
    )

    W[finite_indices] = W1

    W[W == 0] = np.nan
    X["hx"].data *= W
    Y["ey"].data *= W

    W = np.zeros(len(finite_indices))
    x = X["hy"]
    if coh_type == "local":
        y = Y["ex"]
        lower_quantile_cutoff = local_lower_quantile_cutoff
        upper_quantile_cutoff = local_upper_quantile_cutoff
    elif coh_type == "remote":
        y = RR["hy"]
        lower_quantile_cutoff = remote_lower_quantile_cutoff
        upper_quantile_cutoff = remote_upper_quantile_cutoff
    W2 = jackknife_coherence_weights(
        x,
        y,
        lower_quantile_cutoff=lower_quantile_cutoff,
        upper_quantile_cutoff=upper_quantile_cutoff,
    )
    W[finite_indices] = W2
    W[W == 0] = np.nan
    X["hy"].data *= W
    Y["ex"].data *= W
    if RR is not None:
        RR *= W
    return X, Y, RR


# def compute_multiple_coherence_weights(band, local_stft_obj, remote_stft_obj):
#     print(band)
#     print(band)
#     pass
