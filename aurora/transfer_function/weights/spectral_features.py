# import numpy as np
#
# from aurora.transfer_function.regression.helper_functions import direct_solve_tf
# from aurora.transfer_function.regression.helper_functions import simple_solve_tf
#
#
# def estimate_time_series_of_impedances(
#     band_spectrogram, output_ch="ex", use_remote=True
# ):
#     """
#
#     Parameters
#     ----------
#     band_spectrogram: aurora.time_series.spectrogram.Spectrogram
#         A spectrogram of the frequency band to process
#
#     output_ch
#     use_remote
#
#     Returns
#     -------
#
#     solve:
#
#     [<rx,ex>]  = [<rx,hx>, <rx,hy>]   [Zxx]
#     [<ry,ex>]    [<ry,hx>, <ry,hy>]   [Zyx]
#
#     [a, b]
#     [c, d]
#
#     determinant  where det(A)  = 1/(ad-bc)
#
#
#     Requires some nomenclature setup... for now just hard code:/
#
#         TODO: Note that cross powers can be computed once only by using Spectrogram class
#         and spectrogram.cross_power("CH1", "CH2")
#         which returns self._ch1_ch2
#         which initialized to None and is written when requested
#
#
#     :return Zxx, Zxy:
#     """
#
#     def cross_power_series(ch1, ch2):
#         """<ch1.H ch2> summed along frequnecy"""
#         return (ch1.conjugate().transpose() * ch2).sum(dim="frequency")
#
#     band = band_spectrogram.dataset
#     # Start by computing relevant cross powers
#     if use_remote:
#         rx = band["rx"]
#         ry = band["ry"]
#     else:
#         rx = band["hx"]
#         ry = band["hy"]
#     rxex = cross_power_series(rx, band["ex"])
#     ryex = cross_power_series(ry, band["ex"])
#     rxhx = cross_power_series(rx, band["hx"])
#     ryhx = cross_power_series(ry, band["hx"])
#     rxhy = cross_power_series(rx, band["hy"])
#     ryhy = cross_power_series(ry, band["hy"])
#
#     N = len(rxex)
#     # Compute determinants (one per time window)
#     # Note that when no remote reference (rx=hx, ry=hy) then det is
#     # the product of the auto powers in h minus the product of the cross powers
#     det = rxhx * ryhy - rxhy * ryhx
#     det = np.real(det)
#
#     # Construct the Inverse matrices (2 x 2 x N)
#     inverse_matrices = np.zeros((2, 2, N), dtype=np.complex128)
#     inverse_matrices[0, 0, :] = ryhy / det
#     inverse_matrices[1, 1, :] = rxhx / det
#     inverse_matrices[0, 1, :] = -rxhy / det
#     inverse_matrices[1, 0, :] = -ryhx / det
#
#     Zxx = inverse_matrices[0, 0, :] * rxex.data + inverse_matrices[0, 1, :] * ryex.data
#     Zxy = inverse_matrices[1, 0, :] * rxex.data + inverse_matrices[1, 1, :] * ryex.data
#
#     # Below code is a spot check that the calculation of Zxx and Zxy from the "fast vectorized"
#     # method above is consistent with for-looping on np.solve
#
#     # Set up the problem in terms of linalg.solve()
#     # Y = X*b
#     # E = H z
#
#     # # Uncomment for sanity check test
#     # idx = np.random.randint(len(Zxx))  # 0  # a time-window index to check
#     # E = band["ex"][idx, :]
#     # H = band[["hx", "hy"]].to_array()[:, idx].T
#     # z_linalg = simple_solve_tf(E.data, H.data, None)
#     # z_direct = np.array([Zxx[idx], Zxy[idx]])
#     # assert np.isclose(np.abs(z_direct - z_linalg), 0, atol=1e-10).all()
#
#     return Zxx, Zxy
#
#
# def estimate_multiple_coherence(local_spectrogram, component, Z1, Z2):
#     """
#
#     From Z estimates obtained above, we will predict the output for each time window.
#     This means hx,hy (input channels) from each time window must be scaled by Zxx and Zy respectively
#     (TFs) from that time window.
#     For looping with matrix multiplication is too slow in python (probably fine in C/Fortran), and
#     it would be too memory intensive to pack a sparse array (although sparse package may do it!)
#     We can use numpy multiply() function if the arrays are properly shaped ...
#
#     TODO: add option for how TF is computed from cross-powers ala Sims and ala Vozoff
#     In general, to estimate Z we solve, for example:
#     Ex = Zxx Hx + Zxy Hy
#     The standard OLS single station solution is to set E, H as row vectors
#     Ex 1XN, H a 2xN, , H.H is Nx2 hermitian transpose of H
#     Ex * H.H = [Zxx Zxy] H*H.H  * is matrixmult
#     Ex * H.H *(H*H.H)^[-1] = [Zxx Zxy]
#     Replacing H.H with R.H results in a much more stable estimate of Z, if remote available
#
#         Z=EH*HH*−1=1DetHH*ExHx* ExHy* HyHy* −HxHy* −HyHx* HxHx*  Eqn7
#
#     Parameters
#     ----------
#     local_dataset
#     component
#     Z1
#     Z2
#
#     Returns
#     -------
#     multiple_coh
#
#     """
#     local_dataset = local_spectrogram.dataset
#     H = local_dataset[["hx", "hy"]].to_array()
#     hx = H.data[0, :, :].squeeze()
#     hy = H.data[1, :, :].squeeze()
#
#     # hx = band_dataset["hx"].data.T
#     # hy = band_dataset["hy"].data.T
#     E_pred = hx.T * Z1 + hy.T * Z2
#     # Careful that this is scaling each time window separately!!!
#
#     # E pred is the predicted Fourier coefficients
#     # residual = band_dataset[component] - E_pred
#     predicted_energy = (np.abs(E_pred) ** 2).sum(axis=0)
#     original_energy = (np.abs(local_dataset[component]) ** 2).sum(dim="frequency")
#     multiple_coh = predicted_energy / original_energy.data
#     return multiple_coh
