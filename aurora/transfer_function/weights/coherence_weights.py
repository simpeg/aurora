# """
# There are several ways to go about this.  One reasonably straightforward one is:
#
# Take all the Fourier coefficients right before they go into regression (right before or
# after we apply the effective degrees of freedom weights).  Data are strictly 2D at
# this point, you have each channel of data corresponding to a column, and each row
# to an observation.
# Following Jones and Jodicke, lets take the coherence of Ex and Hy
# and that of Ey and Hx.  This yields one single number.
#
# Then, iteratively remove each observation in turn and count whether that partial
# coherence is bigger or smaller than the baseline and by how much
#
# Deflection downward means naughty data and deflection upward means well-behaved data.
#
# This leads to a coherence score for each observation.  We can boot out all data that
#  say move the estimate down (worse than average)
# """
#
# import numpy as np
#
# from aurora.time_series.frequency_band_helpers import adjust_band_for_coherence_sorting
# from aurora.time_series.spectrogram import Spectrogram
# from aurora.transfer_function.weights.spectral_features import (
#     estimate_multiple_coherence,
# )
# from aurora.transfer_function.weights.spectral_features import (
#     estimate_time_series_of_impedances,
# )
# from collections import namedtuple
# from loguru import logger
#
# # convenience class used in coherence calculations
# # not to be confused with Channel classes in mt_metdata and mth5
# CoherenceChannel = namedtuple("Channel", ["local_or_remote", "component"])
#
#
# def simple_coherence_channel_pairs():
#     # define conjugate channel pairs
#     paired_channels = {}
#     paired_channels["local"] = {}
#     paired_channels["local"]["ex"] = CoherenceChannel("local", "hy")
#     paired_channels["local"]["ey"] = CoherenceChannel("local", "hx")
#     paired_channels["remote"] = {}
#     paired_channels["remote"]["hx"] = CoherenceChannel("local", "hx")
#     paired_channels["remote"]["hy"] = CoherenceChannel("local", "hy")
#     return paired_channels
#
#
# def coherence_from_fc_series(c_xy, c_xx, c_yy):
#     # n_obs = len(c_xy)
#     num = np.sum(c_xy) ** 2
#     den = np.sum(c_xx) * np.sum(c_yy)
#     coh = num / den
#     return coh
#
#
# def jackknife_coherence_weights(
#     x, y, lower_quantile_cutoff=0.3, upper_quantile_cutoff=0.95
# ):  # 975):#0.98
#     """
#     Follow the convention of Bendat & Perisol (1980):
#     "The two-sided spectral density function between two random processes is defined
#     using X*Y and _not_ XY*.
#
#
#     Note 1: Extremely high coherence can be due to noise
#     Consider ways to pre-filter those events before this is called.
#     - That may need a min_fraction_to_keep=0.2-0.8, which guards against
#     generally poor data being completely discarded.
#     - Consider variations on coh-sorting threshold_min, and threshold_max
#     near perfect coherency can be expected when coherent noise is present
#     - 95% seems extreme for a threshold, maybe not yielding enogh
#     observations for robust statistics.
#
#     Note2: 2024-09-10: vectorization
#     Instead of repeatedly dropping 1 value from Cxy, Cxx, Cyy, we could try the following:
#     Make a "Jackknife" matrix J that is all ones but zero on the diagonal
#     Then we can just take the inner product of the crosspower with the nth row of J to
#     get the sum of all cross-powers but the nth... or multiply the crosspower vector J
#     to get the partial sums of all crosspowers except the nth element.
#
#     # Note3: Note 2 was not a terrible idea, it is much, much faster than for-looping.
#     However, when the data are large, the JAckknife matrix gets unwieldly, (encountered memory errors).
#     here is maybe a clever algebra trick to get around that.
#
#     We need the partial sums of the
#     c_xy = np.abs(x.conj() * y)
#     c_xx = np.real(x.conj() * x)
#     c_yy = np.real(y.conj() * y)
#     But these are special partial sums, being the total sum except one value.
#     So, that means each "jackknifed" partail sum will differ from the total sum
#     by the difference of one estimate.
#     In fact, if we consider all N partial sums (say of C_xy), the i_th partial sum
#     is the total sum, minus the ith element of C_xy..
#     So, start by computing the sum of C_xy, C_xx, and C_yy one time. This yields three
#     scalar values.
#     Now replicate the scalar values:
#
#
#
#     Parameters
#     ----------
#     x: X["hx"].data
#     y: Y["ey"].data
#     keep_fraction : value in (0,1) to set what to keep / reject.
#     A value of 0.1 would keep 10% of the estimates, a val
#
#     Returns
#     -------
#     W: Weights (currently set to be 0 or 1)
#     """
#
#     n_obs = len(x)
#     n_clip_low = int(lower_quantile_cutoff * n_obs)
#     n_clip_high = int(upper_quantile_cutoff * n_obs)
#     logger.info(f"removing worst {n_clip_low} of {n_obs} via jackknife coherence")
#     logger.info(
#         f"removing 'best' {n_obs - n_clip_high} of {n_obs} via jackknife coherence"
#     )
#
#     c_xy = np.abs(x.conj() * y)
#     c_xx = np.real(x.conj() * x)
#     c_yy = np.real(y.conj() * y)
#
#     # "Jackknife" matrix; all ones but zero on the dia may need some memory checks and a lazy approach with hdf5
#     # J = np.ones((n_obs, n_obs)) - np.eye(n_obs)
#     J = np.ones((n_obs, n_obs), dtype=np.int8) - np.eye(n_obs, dtype=np.int8)
#     p_xx = J @ c_xx.data
#     p_xy = J @ c_xy.data
#     p_yy = J @ c_yy.data
#     p_xy**2 / (p_xx * p_yy)
#     partial_coh = p_xy**2 / (p_xx * p_yy)
#
#     worst_to_best = np.argsort(partial_coh)
#     keepers = worst_to_best[n_clip_low:n_clip_high]
#     # Initialize a weight vector the length = num_observations
#     W = np.zeros(n_obs)  # for example
#     W[keepers] = 1
#     return W
#
#
# # def compute_multiple_coherence_weights(band, local_stft_obj, remote_stft_obj):
# #     print(band)
# #     print(band)
# #     pass
#
#
# def multiple_coherence_weights(
#     frequency_band,
#     local_stft_obj,
#     remote_stft_obj,
#     local_or_remote="remote",
#     components=("ex", "ey", "hz"),
#     rule="min3",
# ):
#     """
#         Estimate the multiple coherence for each time window independently
#         For each time index in the spectrograms we will solve the following three equations
#         Ex = Zxx Hx + Zxy Hy    (Equation 1a)
#         Ey = Zyx Hx + Zyy Hy    (Equation 1b)
#         Hz = Tzx Hx + Tzy Hy    (Equation 1c)
#         where (at each time step) we can think of Ex, Ey, Hx, Hy, Hz as column vectors having M frequencies
#         i.e. they are M x 1 matrices.
#
#         Taking the Equation 1a (Ex = Zxx Hx + Zxy Hy) as an example, we have:
#         [Ex1 = [ Hx1 Hy1 ] [ Zxx
#          Ex2 = [ Hx2 Hy2 ]   Zyx]
#           .
#           .
#           .
#          ExM]= [ HxM HyM ]  (Equation #2)
#         an Mx1 = Mx2 * 2X1
#
#         We can also write this as E = HZ,  (Equation 3)
#         The standard solution is to multiply on the left by H.T, (conjugate transpose if complex valued).
#         Let the symbol R denote the conjugate transpose of H, then R is a 2 x M array
#         Yielding
#         R E = R H Z     (Equation 4)
#         RH is square, 2x2 and almost always invertable.  We can solve this using numpy.linalg.solve,
#         but we need to solve once for every time-window, and that can get very slow, so we want to vectorize
#         the solution over all time windows.
#
#         **ASIDE**: For an OLS solution R is the conjugate transpose of H, BUT any matrix of the same
#         shape whose columns are independent should work (some fine points in the math should be checked)
#         but the point is that we can use any combination of available channels as the rows
#         of R.  It could be Ex & Ey, Hx & Hy, or Remote Hx & Remote Hy, which are the usual, preferred soln,
#         (hence the selection of the variable R).
#
#         Here it is convenient to work in cross-spectral powers -- which can be computed for each
#         at time window individually using vectorial xarray e.g. rxex = (rx*ex).sum(axis=freq)
#
#         Note that the left hand side of Equation 4 is just a 2x1 array, having entries:
#         [<rx,ex>
#          <ry,ex>]
#         where <a,b> denotes the cross power a.H,b.  I.e. it is the Inner product of the FCs of ex
#         with the conjugate transpose vectors hx and hy.
#         Similarly, the RHS of Equation 4 is just a 2x2 matrix times Z (a 2x1)
#         this 2x2 matrix has entries:
#         [<rx,hx>, <ry,hx>
#          <rx,hy>, <ry,hy>]
#
#         We can rewrite Equation 4 as:
#         R E = R H Z
#         =
#         [<rx,ex>]  = [<rx,hx>, <rx,hy]   [Zxx]
#         [<ry,ex>]    [<ry,hx>, <ry,hy>]   [Zyx]   (Equation 5)
#
#         Thus we have N equations:
#
#         [<rx1,ex1>]  = [<rx1,hx1>, <ry1,hx1>]   [Zxx1]   (Equation 6a)
#         [<ry1,ex1>]    [<rx1,hy1>, <ry1,hy1>]   [Zyx1]
#
#         [<rx2,ex2>]  = [<rx2,hx2>, <ry2,hx2>]   [Zxx2]   (Equation 6b)
#         [<ry2,ex2>]    [<rx2,hy2>, <ry2,hy2>]   [Zyx2]
#         ...
#         ...
#         [<rxN,exN>]  = [<rxN,hxN>, <ryN,hxN>]   [ZxxN]   (Equation 6c)
#         [<ryN,exN>]    [<rxN,hyN>, <ryN,hyN>]   [ZyxN]
#
#
#
#         Which can be merged ... combining two of them look like this:
#
#         [<rx1,ex1>]  = [<rx1,hx1>, <ry1,hx1>,     0          0    ]   [Zxx1]
#         [<ry1,ex1>]    [<rx1,hy1>, <ry1,hy1>,     0          0    ]   [Zyx1]
#         [<rx2,ex2>]    [   0     ,    0     , <rx2,hx2>, <ry2,hx2>]   [Zxx2]
#         [<ry2,ex2>]    [   0     ,    0     , <rx2,hy2>, <ry2,hy2>]   [Zxy2]    (Equation 7)
#
#
#         2N X 1   = 2N x 2N  2N x1
#         In general  combining N of them leaves us with a 2N x 2N (relatively sparse) matrix to invert.
#         This matrix can easily have tens of thousands of entries per row/col so inverting (and even forming)
#         the large matrix is impractical.  I have NOT checked if np.solve can do this efficiently, with the
#         sparse matrix library.
#
#         Because each equation is only associated with inverting a 2x2, we can use the formula
#         for the inverse of a 2x2
#
#         For a single matrix we have, for example Equation 5:
#         R E = R H Z
#         =
#         [<rx,ex>]  = [<rx,hx>, <rx,hy]   [Zxx]
#         [<ry,ex>]    [<ry,hx>, <ry,hy>]   [Zyx]
#
#         Now instead of np.solving the 2x2, we can instead use the explicit solution:
#         A= [a b] ^-1 =  # 1/det(A)   [ d -b]   where det(A)  = 1/(ad-bc)
#            [c d]                     [-c  a]
#
#         This requires a bit of bookkeeping but its fast and relatively light on memory especailly as compared
#         to forming the large square matrix problem suggested by (Equation 7) but with N large.
#
#
#
#
#
#     Parameters
#     ----------
#     frequency_band
#     local_stft_obj
#     remote_stft_obj
#     local_or_remote
#     components
#     rule
#
#     Returns
#     -------
#
#     """
#
#     def multiple_coherence_channel_sets(local_or_remote):
#         """tells what channels to use for H.H in multiple coherence"""
#         if local_or_remote == "local":
#             return (CoherenceChannel("local", "hx"), CoherenceChannel("local", "hy"))
#         elif local_or_remote == "remote":
#             return (CoherenceChannel("remote", "hx"), CoherenceChannel("remote", "hy"))
#
#     # cutoff_type = "threshold"
#     cutoffs = {}
#     cutoffs["local"] = {}
#     cutoffs["local"]["lower"] = 0.5
#     cutoffs["local"]["upper"] = 1.01
#     cutoffs["remote"] = {}
#     cutoffs["remote"]["lower"] = 0.5
#     cutoffs["remote"]["upper"] = 1.01
#
#     # Widen the band if needed
#     local_stft = Spectrogram(local_stft_obj)
#     band = adjust_band_for_coherence_sorting(frequency_band, local_stft, rule=rule)
#     if remote_stft_obj is not None:
#         remote_stft = Spectrogram(remote_stft_obj)
#
#     # Extract the FCs for band
#     local_dataset = local_stft.extract_band(band)
#     if remote_stft_obj is not None:
#         remote_dataset = remote_stft.extract_band(band, channels=["hx", "hy"])
#         remote_dataset = remote_dataset.rename({"hx": "rx", "hy": "ry"})
#         band_dataset = local_dataset.merge(remote_dataset)
#     else:
#         band_dataset = local_dataset
#
#     n_obs = band_dataset.time_axis.shape[0]
#     import time
#
#     t0 = time.time()
#     component = "ex"
#     Zxx, Zxy = estimate_time_series_of_impedances(
#         band_dataset, output_ch=component, use_remote=False
#     )
#     print(f"Time elapsed to estimate Z for each time window: {time.time() - t0}")
#     multiple_coh = estimate_multiple_coherence(local_dataset, component, Zxx, Zxy)
#
#     assert len(multiple_coh) == n_obs
#
#     # initialize a dict to hold the weights
#     # in this case there will be only two sets of weights, one set
#     # from the ex equation and another from the ey
#     # and one from hz I guess also if we compute Txy, Tyy
#     weights = {}
#     for component in components:
#         weights[component] = np.ones(n_obs)
#     print("Now use the multiple_coh variable to compute weights and return them")
#     return np.ones(n_obs)
#
#
# def estimate_simple_coherence(X, Y):
#     """
#     X, Y have same time dimension values
#     X,Y, should have same frequency dimension values
#     Frequency dimension should not be of length 1
#     TODO: consider binding X, Y as a single xarray
#
#     Parameters
#     ----------
#     X:xarray.core.dataarray.DataArray
#        Spectrogram: 2D array of Fourier coefficients with dims ('time', 'frequency')
#     Y: xarray.core.dataarray.DataArray
#        Spectrogram: 2D array of Fourier coefficients with dims ('time', 'frequency')
#     X, Y are have same time dimension
#
#     Returns
#     -------
#     coherence: numpy.ndarray
#         univariate time series with one entry per "time" entry in inputs X, Y.
#         Values correspond to simple coherence in the frequency
#     """
#     # Estimate Gxy, Gxx, Gyy (Bendat and Piersol, Eqn 3.46, 3.47) - factors of 2/T drop out when we take ratios
#     xHy = ((X.data.conj() * Y.data)).sum(axis=1)
#     xHx = (np.real(X.data.conj() * X.data)).sum(axis=1)
#     yHy = (np.real(Y.data.conj() * Y.data)).sum(axis=1)
#
#     coherence = np.abs(xHy) / np.sqrt(xHx * yHy)
#     return coherence
#
#
# def estimate_jackknife_coherence(X, Y, ttl=""):
#     """
#     A reasonably efficient way to estimate the coherence between X,Y
#     where the estimate is made by using all but one value in X, Y.
#
#      N.B. This method seeks to identify the ensembles that most influence the sum of all coherences
#      The "worst" value is not necessarily the ensemble with the lowest coherence ... it may also matter the
#      amplitude of the FCs within.
#       We are not just removing the lowest or highest coherence - that is an amplitude independent
#     # measure... we are removing
#
#     Parameters
#     ----------
#     X
#     Y
#
#     Returns
#     -------
#
#     """
#     # Estimate Gxy, Gxx, Gyy (Bendat and Piersol, Eqn 3.46, 3.47) - factors of 2/T drop out when we take ratios
#     xHy = ((X.data.conj() * Y.data)).sum(axis=1)
#     xHx = (np.real(X.data.conj() * X.data)).sum(axis=1)
#     yHy = (np.real(Y.data.conj() * Y.data)).sum(axis=1)
#
#     # Compute the sum over all time windows
#     Sxy = np.sum(xHy)
#     Sxx = np.sum(xHx)
#     Syy = np.sum(yHy)
#
#     # Replicate the sum on each time window
#     Jxy = Sxy * np.ones(X.time.shape[0])
#     Jxx = Sxx * np.ones(X.time.shape[0])
#     Jyy = Syy * np.ones(X.time.shape[0])
#
#     # Diminish the sums by the cross- or auto- power at that time window (Jackknife)
#     Jxy -= xHy
#     Jxx -= xHx
#     Jyy -= yHy
#
#     # Estimate jackknife coherence
#     jackknife_coherence = np.abs(Jxy) / np.sqrt(Jxx * Jyy)
#
#     # Sanity check stuffs:
#     largest_partial_coh_ndx = np.argmax(jackknife_coherence)
#     print(
#         f"Largest partial coherence is due to removing segment {largest_partial_coh_ndx}"
#     )
#     worst = largest_partial_coh_ndx
#     worst_coh = np.abs(xHy[worst]) / np.sqrt(xHx[worst] * yHy[worst])
#     print(f"Which has simple coherence value {worst_coh} -- expected to be low")
#     print("the smallest partial coherence is due to removing the best segment")
#     best = np.argmin(jackknife_coherence)
#     best_coh = np.abs(xHy[best]) / np.sqrt(xHx[best] * yHy[best])
#     print(f"Which has value {best_coh}")
#
#     sc = np.abs(xHy) / np.sqrt(xHx * yHy)
#     import matplotlib.pyplot as plt
#
#     plt.hist(sc, 1000)
#     plt.xlabel("Simple Coherence")
#     plt.ylabel("# Occurrences")
#     if ttl:
#         plt.title(ttl)
#         plt.show()
#     return jackknife_coherence
#
#
# def simple_coherence_weights(
#     frequency_band,
#     local_stft_obj,
#     remote_stft_obj,
#     coh_types=[("local", "ex"), ("local", "ey"), ("remote", "hx"), ("remote", "hy")],
#     cutoffs={},
#     rule="min3",
# ):
#     """
#     TODO: consider making ("local", "ex") apply weights only to local ex and it's paired channel,
#     ditto for ("local", "ey"), ("remote", "hx"), ("remote", "hy")
#     - This would require a bit of wrangling and make this a "channel_weights" method
#     as opposed to a global "segment" weights method.
#
#     Also, Note that when solving the "ex" equation, we should apply the weights for:
#     [local ex, local hy], [local hx, remote hx], [local hy, remote hy],
#     When cutoffs correspond to fairly large amounts of data getting ejected,
#     it is possible that keeping say 20% in each of the three cases, may leave us with no observations
#     So the cumulative weights for ex, hx, hy need to be evaluated together,
#     as do the cumulative weights for ey, hx, hy
#
#
#     Parameters
#     ----------
#     frequency_band
#     local_stft_obj
#     remote_stft_obj
#     coh_types
#     cutoffs
#     rule
#
#     Returns
#     -------
#
#     """
#     # Define rejection criteria: these should be params in the config
#     #    cutoff_type = "threshold"
#     cutoffs = {}
#     cutoffs["local"] = {}
#     cutoffs["local"]["lower"] = 0.3
#     cutoffs["local"]["upper"] = 0.999
#     cutoffs["remote"] = {}
#     cutoffs["remote"]["lower"] = 0.3
#     cutoffs["remote"]["upper"] = 0.999
#     max_reject_fraction = 0.8
#
#     # define conjugate channel pairs
#     paired_channels = simple_coherence_channel_pairs()
#
#     # Widen the band if needed
#     local_stft = Spectrogram(local_stft_obj)
#     remote_stft = Spectrogram(remote_stft_obj)
#     band = adjust_band_for_coherence_sorting(frequency_band, local_stft, rule=rule)
#     # band = frequency_band
#
#     # Extract the FCs for band
#     band_datasets = {}
#     band_datasets["local"] = local_stft.extract_band(band)
#     band_datasets["remote"] = remote_stft.extract_band(band)
#     n_obs = band_datasets["local"].time.shape[0]
#
#     # initialize a dict to hold the weights
#     weights = {}
#     for (local_or_remote, component) in coh_types:
#         if local_or_remote not in weights.keys():
#             weights[local_or_remote] = {}
#         weights[local_or_remote][component] = np.ones(n_obs)
#     cumulative_weights = np.ones(n_obs)
#
#     # Define the channel pair
#     for (local_or_remote, component) in coh_types:
#         ch1 = CoherenceChannel(local_or_remote, component)
#         ch2 = paired_channels[local_or_remote][component]
#         msg = (
#             f"ch1: {ch1.local_or_remote} {ch1.component}; "
#             f"ch2: {ch2.local_or_remote} {ch2.component}; "
#             f"{band.center_frequency:.3f}Hz"
#         )
#         logger.info(f"\n{msg}")
#
#         X = band_datasets[ch1.local_or_remote][ch1.component]
#         Y = band_datasets[ch2.local_or_remote][ch2.component]
#         W = weights[local_or_remote][component]
#         simple_coherence = estimate_simple_coherence(X, Y)
#
#         # rank the windows from low to high
#         # worst_to_best = np.argsort(simple_coherence)
#         # sorted_simple_coherence = simple_coherence[worst_to_best]
#
#         # Apply lower weights
#         threshold_lower = cutoffs[local_or_remote]["lower"]
#         zero_weights = simple_coherence < threshold_lower
#         n_reject = zero_weights.sum()
#         reject_fraction = 1.0 * n_reject / n_obs
#         if reject_fraction > max_reject_fraction:
#             msg = f"\nRejection fraction {reject_fraction} greater than max allowed {max_reject_fraction}\n"
#             msg += f" -- corresponding to {n_reject}/{n_obs} observations\n"
#             msg += f" -- using {max_reject_fraction} quantile instead\n"
#             logger.warning(msg)
#             zero_weights = simple_coherence < np.quantile(simple_coherence, 0.8)
#             n_reject = zero_weights.sum()
#
#         msg = f"Rejecting {n_reject} of {n_obs} spectrogram samples based on simple coherence"
#         msg = f"{msg} between {ch1.local_or_remote} {ch1.component} & "
#         msg = f"{msg} {ch1.local_or_remote} {ch1.component},  < threshold={threshold_lower}"
#         logger.info(msg)
#         W[zero_weights] = 0.0
#
#         # Apply upper weights
#         threshold_upper = cutoffs[local_or_remote]["upper"]
#         zero_weights = simple_coherence > threshold_upper
#         W[zero_weights] = 0.0
#         n_reject = zero_weights.sum()
#         msg = f"Rejecting {n_reject} of {n_obs} spectrogram samples based on simple coherence"
#         msg = f"{msg} between {ch1.local_or_remote} {ch1.component} & "
#         msg = f"{msg} {ch1.local_or_remote} {ch1.component},  > threshold={threshold_upper}"
#         logger.info(msg)
#
#         cumulative_weights *= W
#
#     return cumulative_weights
#
#
# def coherence_weights_jj84(
#     frequency_band,
#     local_stft_obj,
#     remote_stft_obj,
#     coh_types=[("local", "ex"), ("local", "ey"), ("remote", "hx"), ("remote", "hy")],
#     widening_rule="min3",
# ):
#     """
#          This method loosely follows Jones and Jodicke 1984, at least inasfaras it uses a jackknife process.
#     However, the termination criteria for when to stop rejecting is given by a simple fraction of data to keep/reject
#     This is not what is done in JJ84. They use an SNR maximizing method, that could be added here.
#
#     Also, the way I have it now, the jackknifes are estimated once and then rejection is done
#     en masse on the jackknifes ... JJ84 actually suggest recomputing the Jackknifes after each
#     rejection of a single pair.
#
#     TODO: Review this somewhat non-intuitive fact that the FFT-time-window whose removal results in the
#     largest partial coherence (the one that hurts the group most) does not appear to correspond in general
#     to the FFT-time-window that has the lowest simple coherence.  This may be due to the fact that
#     said window itself is highly coherent, but the signal polarization is counter to the average.  Mix in
#     a possibly large amplitude and you can have "worst time windows" being highly coherent ...
#
#     How exactly to mix this method into the flow in general is not 100% clear, but the above observation
#      emphasizes the fact that in the presence of coherent noise, simple and multiple coherence can
#      lead the coherence sorting down an incorrect path.
#
#
#     The thresholds here are NOT values of coherence, they are quantiles to reject.
#
#     This method is not yet supported.  It needs to be made tolerant of channel_nomenclature.  A not
#     unreasonable solution to this would be to make a decorator:
#     def use_channel_nomenclature():
#         Loops over input xarrays and maps channel names from current nomenclature
#         to hexy, and then, when it is done executing, maps the labels back.
#     However, it would cloud a lot of the math and readability if this was done all the time.  Suggest decorating
#     process_transfer_function
#
#     This is a form of "simple coherence", i.e. it just uses two channels at a time.
#     Thus it can take the form of C_xy, C_xx, C_yy
#
#     FLOW:
#     1. Define rejection quantiles
#     2. Define conjugate channel pairs (should be imported from elsewhere, as they are reusable)
#     3. Define the band -- this needs to be factored out (need to have multiple harmonics in a band)
#     4. Extract the FCs for band
#     5. Loop over channels pairs
#         Standard pairings are:
#         "local_ex": X,Y --> local["ex"], local["hy"]
#         "local_ey": X,Y --> local["ey"], local["hx"]
#         "remote_hx": X,Y --> local["hx"], local["hx"]
#         "remote_hy": X,Y --> local["hy"], local["hy"]
#     6. compute weights
#     7. update cumulative weights
#     Parameters
#     ----------
#     frequency_band
#     local_stft_obj
#     remote_stft_object
#     coh_type
#     widening_rule
#
#     Returns
#     -------
#     cumulative_weights: arraylike
#
#     """
#
#     # Define rejection quantiles: these should be params in the config
#     quantile_cutoffs = {}
#     quantile_cutoffs["local"] = {}
#     quantile_cutoffs["local"]["lower"] = 0.2
#     quantile_cutoffs["local"]["upper"] = 0.98
#     quantile_cutoffs["remote"] = {}
#     quantile_cutoffs["remote"]["lower"] = 0.1
#     quantile_cutoffs["remote"]["upper"] = 0.99
#
#     # define conjugate channel pairs
#     paired_channels = simple_coherence_channel_pairs()
#
#     # initialize a dict to hold the weights
#     weights = {}
#     for (local_or_remote, component) in coh_types:
#         if local_or_remote not in weights.keys():
#             weights[local_or_remote] = {}
#         weights[local_or_remote][component] = None
#
#     # Widen the band if needed
#     local_stft = Spectrogram(local_stft_obj)
#     remote_stft = Spectrogram(remote_stft_obj)
#     band = adjust_band_for_coherence_sorting(frequency_band, local_stft, rule="min3")
#     # band = frequency_band
#     # Extract the FCs for band
#     band_datasets = {}
#     band_datasets["local"] = local_stft.extract_band(band)
#     band_datasets["remote"] = remote_stft.extract_band(band)
#     n_obs = band_datasets["local"].time.shape[0]
#
#     cumulative_weights = np.ones(n_obs)
#
#     # Define the channel pair
#     for (local_or_remote, component) in coh_types:
#
#         ch1 = CoherenceChannel(local_or_remote, component)
#         ch2 = paired_channels[local_or_remote][component]
#         msg = (
#             f"ch1: {ch1.local_or_remote} {ch1.component}; "
#             f"ch2: {ch2.local_or_remote} {ch2.component}; "
#             f"{band.center_frequency:.3f}Hz"
#         )
#         logger.info(f"\n{msg}")
#
#         X = band_datasets[ch1.local_or_remote][ch1.component]
#         Y = band_datasets[ch2.local_or_remote][ch2.component]
#
#         jackknife_coherence = estimate_jackknife_coherence(X, Y, ttl=msg)
#
#         # Sanity checking
#         # from matplotlib import pyplot as plt
#         # plt.hist(pc, 10000)
#         # plt.plot(pc)
#
#         # rank the windows -- careful,
#         # the "lowest" partial coherence is due to removing the "highest" coherence window.
#         # so invert the list [::-1]
#         worst_to_best = np.argsort(jackknife_coherence)[::-1]
#
#         # Find cutoff indices
#         lower_quantile_cutoff = quantile_cutoffs[local_or_remote]["lower"]
#         n_clip_low = int(lower_quantile_cutoff * n_obs)
#         upper_quantile_cutoff = quantile_cutoffs[local_or_remote]["upper"]
#         n_clip_high = int(upper_quantile_cutoff * n_obs)
#
#         # Assign weights
#         logger.info(f"removing lowest {n_clip_low} of {n_obs} via jackknife coherence")
#         logger.info(
#             f"removing highest {n_obs - n_clip_high} of {n_obs} via jackknife coherence"
#         )
#         keepers = worst_to_best[n_clip_low:n_clip_high]
#
#         # Uncomment for sanity check
#         # (np.sign(np.diff(jackknife_coherence[worst_to_best])) < 0).all()
#         simple_coherence = estimate_simple_coherence(X, Y)
#         print("the largest partial coherence is due to removing the worst segment")
#         worst_ndx = worst_to_best[0]
#         print(f"Which has value worst_coh {simple_coherence[worst_ndx]}")
#         print("the last low-coh segment to remove has coherence: ")
#         last_low_ndx = worst_to_best[n_clip_low]
#         print(f"Which has value last_bad {simple_coherence[last_low_ndx]}")
#
#         print("the smallest partial coherence is due to removing the best segment")
#         best_ndx = worst_to_best[-1]
#         print(f"Which has value best_coh {simple_coherence[best_ndx]}")
#
#         print("the first high-coh segment to remove has coherence: ")
#         first_high_ndx = worst_to_best[n_obs - n_clip_high]
#         print(f"Which has value last_high {simple_coherence[first_high_ndx]}")
#
#         # Initialize a weight vector the length = num_observations
#         w = np.zeros(n_obs)
#         w[keepers] = 1
#         weights[local_or_remote][component] = w
#         cumulative_weights *= w
#     print(f"cumulative_weights shape {cumulative_weights.shape}")
#     return cumulative_weights
