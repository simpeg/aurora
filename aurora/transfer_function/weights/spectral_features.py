import numpy as np

from aurora.transfer_function.regression.helper_functions import (
    solve_single_time_window,
)


def estimate_time_series_of_impedances(band, output_ch="ex", use_remote=True):
    """
    solve:

    [<rx,ex>]  = [<rx,hx>, <rx,hy>]   [Zxx]
    [<ry,ex>]    [<ry,hx>, <ry,hy>]   [Zyx]

    [a, b]
    [c, d]

    determinant  where det(A)  = 1/(ad-bc)

    :param band: band dataset: xarray, will be spectrogram in future
    :return:

    Requires some nomenclature setup... for now just hard code:/

        TODO: Note that cross powers can be computed once only by using Spectrogram class
        and spectrogram.cross_power("CH1", "CH2")
        which returns self._ch1_ch2
        which initialized to None and is written when requested

    :param band_dataset:
    :return:
    """

    def cross_power_series(ch1, ch2):
        """<ch1.H ch2> summed along frequnecy"""
        return (ch1.conjugate().transpose() * ch2).sum(dim="frequency")

    # Start by computing relevant cross powers
    if use_remote:
        rx = band["rx"]
        ry = band["ry"]
    else:
        rx = band["hx"]
        ry = band["hy"]
    rxex = cross_power_series(rx, band["ex"])
    ryex = cross_power_series(ry, band["ex"])
    rxhx = cross_power_series(rx, band["hx"])
    ryhx = cross_power_series(ry, band["hx"])
    rxhy = cross_power_series(rx, band["hy"])
    ryhy = cross_power_series(ry, band["hy"])

    N = len(rxex)
    # Compute determinants (one per time window)
    # Note that when no remote reference (rx=hx, ry=hy) then det is
    # the product of the auto powers in h minus the product of the cross powers
    det = rxhx * ryhy - rxhy * ryhx
    det = np.real(det)

    # Construct the Inverse matrices (2 x 2 x N)
    inverse_matrices = np.zeros((2, 2, N), dtype=np.complex128)
    inverse_matrices[0, 0, :] = ryhy / det
    inverse_matrices[1, 1, :] = rxhx / det
    inverse_matrices[0, 1, :] = -rxhy / det
    inverse_matrices[1, 0, :] = -ryhx / det

    Zxx = inverse_matrices[0, 0, :] * rxex.data + inverse_matrices[0, 1, :] * ryex.data
    Zxy = inverse_matrices[1, 0, :] * rxex.data + inverse_matrices[1, 1, :] * ryex.data

    # Below code is a spot check that the calculation of Zxx and Zxy from the "fast vectorized"
    # method above is consistent with for-looping on np.solve

    # Set up the problem in terms of linalg.solve()
    idx = 0  # a time-window index to check
    E = band["ex"][idx, :]
    H = band[["hx", "hy"]].to_array()[:, idx].T
    HH = H.conj().transpose()
    a = HH.data @ H.data
    b = HH.data @ E.data

    # Solve using direct inverse
    inv_a = np.linalg.inv(a)
    zz0 = inv_a @ b
    # solve using linalg.solve
    zz = solve_single_time_window(b, a, None)
    # compare the two solutions
    assert np.isclose(np.abs(zz0 - zz), 0, atol=1e-10).all()

    # Estimate multiple coherence with linalg.solve soln
    solved_residual = E - H[:, 0] * zz[0] - H[:, 1] * zz[1]
    solved_residual_energy = (np.abs(solved_residual) ** 2).sum()
    output_energy = (np.abs(E) ** 2).sum()
    multiple_coherence_1 = solved_residual_energy / output_energy
    print("solve", multiple_coherence_1)
    # Estimate multiple coherence with Zxx Zxy
    homebrew_residual = E - H[:, 0] * Zxx[0] - H[:, 1] * Zxy[0]
    homebrew_residual_energy = (np.abs(homebrew_residual) ** 2).sum()
    multiple_coherence_2 = homebrew_residual_energy / output_energy
    print(multiple_coherence_2)
    assert np.isclose(
        multiple_coherence_1.data, multiple_coherence_2.data, 0, atol=1e-14
    ).all()
    return Zxx, Zxy
