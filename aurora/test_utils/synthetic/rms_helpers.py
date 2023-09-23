import numpy as np


def compute_rms(rho, phi, model_rho_a=100.0, model_phi=45.0, verbose=False):
    """
    This function being used to make comparative plots for synthetic data.  Could be
    used in general to compare different processing results.  For example by replacing
    model_rho_a and model_phi with other processing results, or other (
    non-uniform) model results.

    Parameters
    ----------
    rho: numpy.ndarray
        1D array of computed apparent resistivities (expected in Ohmm)
    phi: numpy.ndarrayx
        1D array of computed phases (expected in degrees)
    model_rho_a: float or numpy array
        if numpy array must be the same shape as rho
    model_phi: float or numpy array
        if numpy array must be the same shape as phi.
    Returns
    -------
    rho_rms: float
        rms misfit between the model apparent resistivity and the computed resistivity
    phi_rms: float
        rms misfit between the model phase (or phases) and the computed phase
    """
    rho_rms = np.sqrt(np.mean((rho - model_rho_a) ** 2))
    phi_rms = np.sqrt(np.mean((phi - model_phi) ** 2))
    if verbose:
        print(f"rho_rms = {rho_rms}")
        print(f"phi_rms = {phi_rms}")
    return rho_rms, phi_rms


def get_expected_rms_misfit(test_case_id, emtf_version=None):
    expected_rms_misfit = {}
    expected_rms_misfit["rho"] = {}
    expected_rms_misfit["phi"] = {}
    if test_case_id == "test1":
        if emtf_version == "fortran":
            expected_rms_misfit["rho"]["xy"] = 4.433905
            expected_rms_misfit["phi"]["xy"] = 0.910484
            expected_rms_misfit["rho"]["yx"] = 3.658614
            expected_rms_misfit["phi"]["yx"] = 0.844645
        elif emtf_version == "matlab":
            expected_rms_misfit["rho"]["xy"] = 2.706098
            expected_rms_misfit["phi"]["xy"] = 0.784229
            expected_rms_misfit["rho"]["yx"] = 3.745280
            expected_rms_misfit["phi"]["yx"] = 1.374938

    elif test_case_id == "test2r1":
        expected_rms_misfit["rho"]["xy"] = 3.971313
        expected_rms_misfit["phi"]["xy"] = 0.982613
        expected_rms_misfit["rho"]["yx"] = 3.967259
        expected_rms_misfit["phi"]["yx"] = 1.62881
    return expected_rms_misfit


def assert_rms_misfit_ok(
    expected_rms_misfit,
    xy_or_yx,
    rho_rms_aurora,
    phi_rms_aurora,
    rho_tol=1e-4,
    phi_tol=1e-4,
):
    """

    Parameters
    ----------
    expected_rms_misfit: dictionary
        precomputed RMS misfits for test data in rho and phi
    xy_or_yx: str
        mode
    rho_rms_aurora: float
    phi_rms_aurora: float
    """
    expected_rms_rho = expected_rms_misfit["rho"][xy_or_yx]
    expected_rms_phi = expected_rms_misfit["phi"][xy_or_yx]
    print(f"expected_rms_rho_{xy_or_yx} {expected_rms_rho}")
    print(f"expected_rms_phi_{xy_or_yx} {expected_rms_phi}")
    if not np.isclose(rho_rms_aurora - expected_rms_rho, 0, atol=rho_tol):
        print("==== AURORA ====\n")
        print(rho_rms_aurora)
        print("==== EXPECTED ====\n")
        print(expected_rms_rho)
        print("==== DIFFERENCE ====\n")
        print(rho_rms_aurora - expected_rms_rho)
        raise AssertionError("Expected misfit for resistivity is not correct")

    if not np.isclose(phi_rms_aurora - expected_rms_phi, 0, atol=rho_tol):
        print("==== AURORA ====\n")
        print(phi_rms_aurora)
        print("==== EXPECTED ====\n")
        print(expected_rms_phi)
        print("==== DIFFERENCE ====\n")
        print(phi_rms_aurora - expected_rms_phi)
        raise AssertionError("Expected misfit for phase is not correct")

    return
