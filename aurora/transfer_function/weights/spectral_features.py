import numpy as np

from aurora.transfer_function.regression.helper_functions import direct_solve_tf
from aurora.transfer_function.regression.helper_functions import simple_solve_tf


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
    # Y = X*b
    # E = H z

    # Uncomment for sanity check test
    idx = np.random.randint(len(Zxx))  # 0  # a time-window index to check
    E = band["ex"][idx, :]
    H = band[["hx", "hy"]].to_array()[:, idx].T
    z_direct = direct_solve_tf(E.data, H.data)
    z_linalg = simple_solve_tf(E.data, H.data, None)
    z_tricky = np.array([Zxx[idx], Zxy[idx]])
    assert np.isclose(np.abs(z_direct - z_linalg), 0, atol=1e-10).all()
    assert np.isclose(np.abs(z_direct - z_tricky), 0, atol=1e-10).all()

    return Zxx, Zxy
