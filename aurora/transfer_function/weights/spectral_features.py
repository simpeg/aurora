import numpy as np
import xarray as xr
from aurora.transfer_function.regression.helper_functions import direct_solve_tf
from aurora.transfer_function.regression.helper_functions import simple_solve_tf


def estimate_time_series_of_impedances(
    band_spectrogram, output_ch="ex", use_remote=True
):
    """

    Parameters
    ----------
    band_spectrogram: aurora.time_series.spectrogram.Spectrogram
        A spectrogram of the frequency band to process

    output_ch
    use_remote

    Returns
    -------

    solve:

    [<rx,ex>]  = [<rx,hx>, <rx,hy>]   [Zxx]
    [<ry,ex>]    [<ry,hx>, <ry,hy>]   [Zyx]

    [a, b]
    [c, d]

    determinant  where det(A)  = 1/(ad-bc)


    Requires some nomenclature setup... for now just hard code:/

        TODO: Note that cross powers can be computed once only by using Spectrogram class
        and spectrogram.cross_power("CH1", "CH2")
        which returns self._ch1_ch2
        which initialized to None and is written when requested


    :return Zxx, Zxy:
    """
    # Start by computing relevant cross powers
    if use_remote:
        rx = "rx"
        ry = "ry"
    else:
        rx = "hx"
        ry = "hy"
    rxex = band_spectrogram.cross_power(rx, "ex")
    ryex = band_spectrogram.cross_power(ry, "ex")
    rxhx = band_spectrogram.cross_power(rx, "hx")
    ryhx = band_spectrogram.cross_power(ry, "hx")
    rxhy = band_spectrogram.cross_power(rx, "hy")
    ryhy = band_spectrogram.cross_power(ry, "hy")

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

    # # Uncomment for sanity check test
    # idx = np.random.randint(len(Zxx))  # 0  # a time-window index to check
    # E = band["ex"][idx, :]
    # H = band[["hx", "hy"]].to_array()[:, idx].T
    # z_linalg = simple_solve_tf(E.data, H.data, None)
    # z_direct = np.array([Zxx[idx], Zxy[idx]])
    # assert np.isclose(np.abs(z_direct - z_linalg), 0, atol=1e-10).all()

    return Zxx, Zxy


def estimate_multiple_coherence(local_spectrogram, component, Z1, Z2):
    """

    From Z estimates obtained above, we will predict the output for each time window.
    This means hx,hy (input channels) from each time window must be scaled by Zxx and Zy respectively
    (TFs) from that time window.
    For-looping with matrix multiplication is too slow in python (probably fine in C/Fortran), and
    it would be too memory intensive to pack a sparse array (although sparse package may do it!)
    We can use numpy multiply() function if the arrays are properly shaped ...

    TODO: add option for how TF is computed from cross-powers ala Sims and ala Vozoff
    In general, to estimate Z we solve, for example:
    Ex = Zxx Hx + Zxy Hy
    The standard OLS single station solution is to set E, H as row vectors
    Ex 1XN, H a 2xN, , H.H is Nx2 hermitian transpose of H
    Ex * H.H = [Zxx Zxy] H*H.H  * is matrixmult
    Ex * H.H *(H*H.H)^[-1] = [Zxx Zxy]
    Replacing H.H with R.H results in a much more stable estimate of Z, if remote available

        Z=EH*HH*−1=1DetHH*ExHx* ExHy* HyHy* −HxHy* −HyHx* HxHx*  Eqn7

    Parameters
    ----------
    local_dataset
    component
    Z1
    Z2

    Returns
    -------
    multiple_coh

    """
    local_dataset = local_spectrogram.dataset
    H = local_dataset[["hx", "hy"]].to_array()
    hx = H.data[0, :, :].squeeze()
    hy = H.data[1, :, :].squeeze()

    # hx = band_dataset["hx"].data.T
    # hy = band_dataset["hy"].data.T
    E_pred = hx.T * Z1 + hy.T * Z2
    # Careful that this is scaling each time window separately!!!

    # E pred is the predicted Fourier coefficients
    # residual = band_dataset[component] - E_pred
    predicted_energy = (np.abs(E_pred) ** 2).sum(axis=0)
    original_energy = (np.abs(local_dataset[component]) ** 2).sum(dim="frequency")
    multiple_coh = predicted_energy / original_energy.data
    return multiple_coh


def estimate_simple_coherence(band_spectrogram, use_remote=False, channel_pairs=None):
    """
    Simple coherence is defined in Bendat & Piersol Equation 3.43 as the square of the cross power, divided by
    the product of the individual autopowers.  Note that some applications refer to the sqrt of this function
    as the coherence.

    band_spectrogram: aurora.time_series.spectrogram.Spectrogram
        A spectrogram of the frequency band to process
    channel_pairs: None or iterable

    The channel pairs that we expect to be coherent are:
        - perpendicular E, H
        - parallel E, E
        - parallel H, H

    Thus common channel pairs will be:
    (ex, hy), (ey, hx), (ex, ry), (ey, rx), (hx, rx), (hy, ry)

    But we can accept any pairs
    Returns
    -------

    """
    coherences = xr.Dataset(
        coords={"time": band_spectrogram.dataset.time.data},
    )

    if channel_pairs is None:
        if use_remote:
            rx = "rx"
            ry = "ry"
        else:
            rx = "hx"
            ry = "hy"
        channel_pairs = (
            ("ex", "hy"),
            ("ey", "hx"),
            ("ex", ry),
            ("ey", rx),
            ("hx", rx),
            ("hy", ry),
        )
    for channel_pair in channel_pairs:
        ch1 = channel_pair[0]
        ch2 = channel_pair[1]
        name = f"{ch1}_{ch2}"
        xpwr = band_spectrogram.cross_power(ch1, ch2)
        auto1 = band_spectrogram.cross_power(ch1, ch1)
        auto2 = band_spectrogram.cross_power(ch2, ch2)
        coherence = xpwr.__abs__() ** 2 / (auto1 * auto2)
        coherences[name] = coherence
    return coherences
