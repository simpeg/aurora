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

from aurora.time_series.frequency_band_helpers import Spectrogram
from collections import namedtuple
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
    Follow the convention of Bendat & Perisol (1980):
    "The two-sided spectral density function between two random processes is defined
    using X*Y and _not_ XY*.


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

    # Note3: Note 2 was not a terrible idea, it is much, much faster than for-looping.
    However, when the data are large, the JAckknife matrix gets unwieldly, (encountered memory errors).
    here is maybe a clever algebra trick to get around that.

    We need the partial sums of the
    c_xy = np.abs(x.conj() * y)
    c_xx = np.real(x.conj() * x)
    c_yy = np.real(y.conj() * y)
    But these are special partial sums, being the total sum except one value.
    So, that means each "jackknifed" partail sum will differ from the total sum
    by the difference of one estimate.
    In fact, if we consider all N partial sums (say of C_xy), the i_th partial sum
    is the total sum, minus the ith element of C_xy..
    So, start by computing the sum of C_xy, C_xx, and C_yy one time. This yields three
    scalar values.
    Now replicate the scalar values:



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

    c_xy = np.abs(x.conj() * y)
    c_xx = np.real(x.conj() * x)
    c_yy = np.real(y.conj() * y)

    # "Jackknife" matrix; all ones but zero on the dia may need some memory checks and a lazy approach with hdf5
    # J = np.ones((n_obs, n_obs)) - np.eye(n_obs)
    J = np.ones((n_obs, n_obs), dtype=np.int8) - np.eye(n_obs, dtype=np.int8)
    p_xx = J @ c_xx.data
    p_xy = J @ c_xy.data
    p_yy = J @ c_yy.data
    p_xy**2 / (p_xx * p_yy)
    partial_coh = p_xy**2 / (p_xx * p_yy)

    worst_to_best = np.argsort(partial_coh)
    keepers = worst_to_best[n_clip_low:n_clip_high]
    # Initialize a weight vector the length = num_observations
    W = np.zeros(n_obs)  # for example
    W[keepers] = 1
    return W


# def compute_multiple_coherence_weights(band, local_stft_obj, remote_stft_obj):
#     print(band)
#     print(band)
#     pass


def estimate_mulitple_coherence(X, Y, Z):
    """
    TODO: add option for how TF is computed from cross-powers ala Sims and ala Vozoff
    Parameters
    ----------
    X
    Y
    Z

    Returns
    -------

    """
    pass
    # # Estimate Gxy, Gxx, Gyy (Bendat and Piersol, Eqn 3.46, 3.47) - factors of 2/T drop out when we take ratios
    # xHy = (np.real(X.data.conj() * Y.data)).sum(axis=1)
    # xHx = (np.real(X.data.conj() * X.data)).sum(axis=1)
    # yHy = (np.real(Y.data.conj() * Y.data)).sum(axis=1)
    #
    # coherence = np.abs(xHy) / np.sqrt(xHx * yHy)
    # return coherence


def estimate_simple_coherence(X, Y):
    # Estimate Gxy, Gxx, Gyy (Bendat and Piersol, Eqn 3.46, 3.47) - factors of 2/T drop out when we take ratios
    xHy = (np.real(X.data.conj() * Y.data)).sum(axis=1)
    xHx = (np.real(X.data.conj() * X.data)).sum(axis=1)
    yHy = (np.real(Y.data.conj() * Y.data)).sum(axis=1)

    coherence = np.abs(xHy) / np.sqrt(xHx * yHy)
    return coherence


def estimate_jackknife_coherence(X, Y):
    # Estimate Gxy, Gxx, Gyy (Bendat and Piersol, Eqn 3.46, 3.47) - factors of 2/T drop out when we take ratios
    xHy = (np.real(X.data.conj() * Y.data)).sum(axis=1)
    xHx = (np.real(X.data.conj() * X.data)).sum(axis=1)
    yHy = (np.real(Y.data.conj() * Y.data)).sum(axis=1)

    # Compute the sum over all time windows
    Sxy = np.sum(xHy)
    Sxx = np.sum(xHx)
    Syy = np.sum(yHy)

    # Replicate the sum on each time window
    Jxy = Sxy * np.ones(X.time.shape[0])
    Jxx = Sxx * np.ones(X.time.shape[0])
    Jyy = Syy * np.ones(X.time.shape[0])

    # Diminish the sums by the cross- or auto- power at that time window (Jackknife)
    Jxy -= xHy
    Jxx -= xHx
    Jyy -= yHy

    # Estimate jackknife coherence
    jackknife_coherence = np.abs(Jxy) / np.sqrt(Jxx * Jyy)

    print("the largest partial coherence is due to removing the worst segment")
    worst = np.argmax(jackknife_coherence)
    worst_coh = np.abs(xHy[worst]) / np.sqrt(xHx[worst] * yHy[worst])
    print(f"Which has value {worst_coh}")
    print("the smallest partial coherence is due to removing the best segment")
    best = np.argmin(jackknife_coherence)
    best_coh = np.abs(xHy[best]) / np.sqrt(xHx[best] * yHy[best])
    print(f"Which has value {best_coh}")

    return jackknife_coherence


def coherence_weights_jj84(
    frequency_band,
    local_stft_obj,
    remote_stft_obj,
    coh_types=[("local", "ex"), ("local", "ey"), ("remote", "hx"), ("remote", "hy")],
    widening_rule="min3",
):
    """
         This method loosely follows Jones and Jodicke 1984, at least inasfaras it uses a jackknife process.
    However, the termination criteria for when to stop rejecting is given by a simple fraction of data to keep/reject
    This is not what is done in JJ84. They use an SNR maximizing method, that could be added here.

    The thresholds here are NOT values of coherence, they are quantiles to reject.

    This method is not yet supported.  It needs to be made tolerant of channel_nomenclature.  A not
    unreasonable solution to this would be to make a decorator:
    def use_channel_nomenclature():
        Loops over input xarrays and maps channel names from current nomenclature
        to hexy, and then, when it is done executing, maps the labels back.
    However, it would cloud a lot of the math and readability if this was done all the time.  Suggest decorating
    process_transfer_function

    This is a form of "simple coherence", i.e. it just uses two channels at a time.
    Thus it can take the form of C_xy, C_xx, C_yy

    FLOW:
    1. Define rejection quantiles
    2. Define conjugate channel pairs (should be imported from elsewhere, as they are reusable)
    3. Define the band -- this needs to be factored out (need to have multiple harmonics in a band)
    4. Extract the FCs for band
    5. Loop over channels pairs
        Standard pairings are:
        "local_ex": X,Y --> local["ex"], local["hy"]
        "local_ey": X,Y --> local["ey"], local["hx"]
        "remote_hx": X,Y --> local["hx"], local["hx"]
        "remote_hy": X,Y --> local["hy"], local["hy"]
    6. compute weights
    7. update cumulative weights
    Parameters
    ----------
    frequency_band
    local_stft_obj
    remote_stft_object
    coh_type
    widening_rule

    Returns
    -------
    cumulative_weights: arraylike

    """

    # Define rejection quantiles: these should be params in the config
    quantile_cutoffs = {}
    quantile_cutoffs["local"] = {}
    quantile_cutoffs["local"]["lower"] = 0.2
    quantile_cutoffs["local"]["upper"] = 0.98
    quantile_cutoffs["remote"] = {}
    quantile_cutoffs["remote"]["lower"] = 0.1
    quantile_cutoffs["remote"]["upper"] = 0.99

    # define conjugate channel pairs
    Channel = namedtuple("Channel", ["local_or_remote", "component"])
    paired_channels = {}
    paired_channels["local"] = {}
    paired_channels["local"]["ex"] = Channel("local", "hy")
    paired_channels["local"]["ey"] = Channel("local", "hx")
    paired_channels["remote"] = {}
    paired_channels["remote"]["hx"] = Channel("local", "hx")
    paired_channels["remote"]["hy"] = Channel("local", "hy")

    # initialize a dict to hold the weights
    weights = {}
    for (local_or_remote, component) in coh_types:
        if local_or_remote not in weights.keys():
            weights[local_or_remote] = {}
        weights[local_or_remote][component] = None

    # Define the band
    band = frequency_band.copy()
    logger.info(
        f"Processing band {band.center_period:.6f}s  ({1. / band.center_period:.6f}Hz)"
    )
    local_stft = Spectrogram(local_stft_obj)
    remote_stft = Spectrogram(remote_stft_obj)
    if local_stft.num_harmonics_in_band(band) == 1:
        logger.warning("Cant evaluate coherence with only 1 harmonic")
        logger.info(f"Widening band according to {widening_rule} rule")
        if widening_rule == "min3":
            band.frequency_min -= local_stft.df
            band.frequency_max += local_stft.df
        else:
            msg = f"Widening rule {widening_rule} not recognized"
            logger.error(msg)
            raise NotImplementedError(msg)

    # Extract the FCs for band
    band_datasets = {}
    band_datasets["local"] = local_stft.extract_band(
        band,
    )
    band_datasets["remote"] = remote_stft.extract_band(band)
    n_obs = band_datasets["local"].time.shape[0]

    cumulative_weights = np.ones(n_obs)

    # Define the channel pair
    for (local_or_remote, component) in coh_types:

        ch1 = Channel(local_or_remote, component)
        ch2 = paired_channels[local_or_remote][component]
        print(f"ch1: {ch1}; ch2: {ch2}")

        X = band_datasets[ch1.local_or_remote][ch1.component]
        Y = band_datasets[ch2.local_or_remote][ch2.component]

        jackknife_coherence = estimate_jackknife_coherence(X, Y)

        # Sanity checking
        # from matplotlib import pyplot as plt
        # plt.hist(pc, 10000)
        # plt.plot(pc)

        # rank the windows -- careful,
        # the "lowest" partial coherence is due to removing the "highest" coherence window.
        # so invert the list [::-1]
        worst_to_best = np.argsort(jackknife_coherence)[::-1]

        # Find cutoff indices
        lower_quantile_cutoff = quantile_cutoffs[local_or_remote]["lower"]
        n_clip_low = int(lower_quantile_cutoff * n_obs)
        upper_quantile_cutoff = quantile_cutoffs[local_or_remote]["upper"]
        n_clip_high = int(upper_quantile_cutoff * n_obs)

        # Assign weights
        logger.info(f"removing lowest {n_clip_low} of {n_obs} via jackknife coherence")
        logger.info(
            f"removing highest {n_obs - n_clip_high} of {n_obs} via jackknife coherence"
        )
        keepers = worst_to_best[n_clip_low:n_clip_high]

        # Uncomment for sanity check
        simple_coherence = estimate_simple_coherence(X, Y)
        print("the largest partial coherence is due to removing the worst segment")
        worst_ndx = worst_to_best[0]
        print(f"Which has value worst_coh {simple_coherence[worst_ndx]}")
        print("the last low-coh segment to remove has coherence: ")
        last_low_ndx = worst_to_best[n_clip_low]
        print(f"Which has value last_bad {simple_coherence[last_low_ndx]}")

        print("the smallest partial coherence is due to removing the best segment")
        best_ndx = worst_to_best[-1]
        print(f"Which has value best_coh {simple_coherence[best_ndx]}")

        print("the first high-coh segment to remove has coherence: ")
        first_high_ndx = worst_to_best[n_obs - n_clip_high]
        print(f"Which has value last_high {simple_coherence[first_high_ndx]}")

        # Initialize a weight vector the length = num_observations
        w = np.zeros(n_obs)
        w[keepers] = 1
        weights[local_or_remote][component] = w
        cumulative_weights *= w
    print(f"cumulative_weights shape {cumulative_weights.shape}")
    return cumulative_weights
