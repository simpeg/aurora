"""
    Computes transfer functions from covariance matrix of cross powers.

    Development notes:
    It is not uncommon to want to compute simple, "non-robust" transfer functions from matrices of cross-powers during
    MT processing.  The preliminary estimates can be used as "features" that are in turn used to compute data weights
    prior to robust regression.

    There will probably be faster, vectorial ways to do this in future versions of the software, however, in this
    work-in-progress module are the functions that can be used to generate TF estimates from matrices of cross-powers
    using xarray DataArrays as the cross-power containers.

    The idea here is to boil the TF estimation from cross powers down to only two equations.
    Equations 10-15 in Sims et al. 1971 describe the 6 ways to compute the SS TF.
    Vozoff (1991) distills this to one equation (42) with six choices for a tuple (A,B)
    (A,B) = (Ex, Ey), (Ex, Hx), (Ex, Hy), (Ey, Hx), (Ey, Hy), (Hx, Hy)

    Vozoff Notation:
                <Ex,A*><Hy,B*> - <Ex,B*><Hy,A*>
     Z_xx =     ------------------------------       Equation V91_42a
                <Hx,A*><Hy,B*> - <Hx,B*><Hy,A*>


                <Ex,A*><Hx,B*> - <Ex,B*><Hx,A*>
     Z_xy =     ------------------------------        Equation V91_42b (modified to correct typo in publication)
                <Hy,A*><Hx,B*> - <Hy,B*><Hx,A*>


                <Ey,A*><Hy,B*> - <Ey,B*><Hy,A*>
     Z_yx =     ------------------------------         Equation V91_42c
                <Hx,A*><Hy,B*> - <Hx,B*><Hy,A*>


                <Ey,A*><Hx,B*> - <Ey,B*><Hx,A*>
     Z_yy =     ------------------------------          Equation V91_42d
                <Hy,A*><Hx,B*> - <Hy,B*><Hx,A*>

    Note that the Equations involving Ex (Z_xx, & Zxy) are identical to the Equations
    involving Ey (Z_yx, & Zyy) if we swap Ex for Ey.  I.e.
    - Equation V91_42a is the same as Equation V91_42c if we switch Ex for Ey.  Similarly,
    - Equation V91_42b is the same as Equation V91_42d if we switch Ex for Ey.
    - This motivates a similar formulation for the Tipper:


                <Hz,A*><Hy,B*> - <Hz,B*><Hy,A*>
     T_x =     ------------------------------
                <Hx,A*><Hy,B*> - <Hx,B*><Hy,A*>

                <Hz,A*><Hx,B*> - <Hz,B*><Hx,A*>
     T_y =     ------------------------------
                <Hy,A*><Hx,B*> - <Hy,B*><Hy,A*>

    However, even V91 can be further distilled to only two equations, with an argument defining the field component
    that is the "output" of the TF formulation Y = one of ("Ex", "Ey", "Hz")
    I.e. by the above logic, there is one more abstraction -- instead of six functions for TF, let Ex, Ey, Hz be called say, Y.

    References:
    Sims, W.E., F.X. Bostick, and H.W. SMITH. “THE ESTIMATION OF MAGNETOTELLURIC IMPEDANCE TENSOR ELEMENTS FROM MEASURED DATA” GEOPHYSICS 36, no. 5 (1971). https://doi.org/DOI:10.1190/1.1440225.
    @inbook{doi:10.1190/1.9781560802686.ch8,
    author = {K. Vozoff},
    title = {8. The Magnetotelluric Method},
    booktitle = {Electromagnetic Methods in Applied Geophysics: Volume 2, Application, Parts A and B},
    chapter = {},
    pages = {641-712},
    year = {2012},
    doi = {10.1190/1.9781560802686.ch8},
    URL = {https://library.seg.org/doi/abs/10.1190/1.9781560802686.ch8},
    eprint = {https://library.seg.org/doi/pdf/10.1190/1.9781560802686.ch8}
    }

    TODO: Consider making the _t__x, _t__y methods of a covariance matrix class.

    TODO: Expand this to a jupyter tutorial
     -  Add Sim's Single station estimators
     - Add more doc -- For any of the TF equations, two of 6 (A,B) combinations are unstable, two are upward biased, and two are downward biased.
     - Verify that setting (A=Ey, B=Ex) recovers Sims Equation 10
     - Verify that Sims' Equations 11-15 result from an appropriate choice of (A,B)

"""

import xarray as xr
from mt_metadata.transfer_functions.tf.transfer_function import TransferFunction
from typing import Optional, Union


def tf_from_cross_powers(
    sdm: xr.DataArray,
    station_id: str,
    remote: Optional[str] = "",
    #    style: Optional[str] = "",
    components: Optional[str] = "full",
    period: Optional[float] = -1.0,
    output_format: Optional[str] = "dict",
    join_char: Optional[str] = "_",
) -> Union[dict, TransferFunction]:
    """
    returns the requested tf components from the SDM

    Develpment notes:

    TODO add validator to ensure impedance components are present and tipper components are present

    TODO: Careful here with conjugates -- we need to establish a conjugation convention for the SDM.
     Let Xi be the data from ith row of X, and Xj be the data from the jth row of X.
     Then <Xi, Xj*> will be S[i,j]
     and <Xi*, Xj> will be S[j,i]
     Vozoff only uses the form where the second variable is conjugate, so we should be consistent throughout.
     The SDM is assumed to be formed from X@XH, so S[i,j] = Xi@XjH

    Parameters
    ----------
    sdm: xr.DataArray
        A covariance matrix of the Fourier coefficients for a particular frequency band.
    station_id: str
        The label of the station to get a TF for
    remote: str
        The label of the remote reference station to get a TF for.
    style: str
        There are six ways to estimate the single station TF
        Supported styles will be: SS[1,2,3,4,5,6], RRH, RRE, MV
        SS1-6 will use the equations from Sims (which can be simplified to fit Vozoff 1991)
        RRH will use the classic RR method
            (which simply replaces
    components: Optional[str] = "full",
        The transfer function components to return, this could be impedance, tipper, or full.
    period: float
        Not yet supported -- needed to convert tf to apparent resistivity
    join_char: str
        The character that links the station_id with the component name in the covariance matrix

    Returns
    -------

    """
    Ex, Ey, Hx, Hy, Hz, A, B = _channel_names(station_id, remote, join_char)

    tf = {}
    tf["z_xx"] = _tf__x(sdm, Y=Ex, Hx=Hx, Hy=Hy, A=A, B=B)
    tf["z_xy"] = _tf__y(sdm, Y=Ex, Hx=Hx, Hy=Hy, A=A, B=B)
    tf["z_yx"] = _tf__x(sdm, Y=Ey, Hx=Hx, Hy=Hy, A=A, B=B)
    tf["z_yy"] = _tf__y(sdm, Y=Ey, Hx=Hx, Hy=Hy, A=A, B=B)
    tf["t_zx"] = _tf__x(sdm, Y=Hz, Hx=Hx, Hy=Hy, A=A, B=B)
    tf["t_zy"] = _tf__y(sdm, Y=Hz, Hx=Hx, Hy=Hy, A=A, B=B)

    # TODO: Add this to a test module or ipynb.
    assert _zxx(sdm, Ex=Ex, Hx=Hx, Hy=Hy, A=A, B=B) == _tf__x(
        sdm, Y=Ex, Hx=Hx, Hy=Hy, A=A, B=B
    )
    assert _zxy(sdm, Ex=Ex, Hx=Hx, Hy=Hy, A=A, B=B) == _tf__y(
        sdm, Y=Ex, Hx=Hx, Hy=Hy, A=A, B=B
    )
    assert _zyx(sdm, Ey=Ey, Hx=Hx, Hy=Hy, A=A, B=B) == _tf__x(
        sdm, Y=Ey, Hx=Hx, Hy=Hy, A=A, B=B
    )
    assert _zyy(sdm, Ey=Ey, Hx=Hx, Hy=Hy, A=A, B=B) == _tf__y(
        sdm, Y=Ey, Hx=Hx, Hy=Hy, A=A, B=B
    )
    assert _tx(sdm, Hz=Hz, Hx=Hx, Hy=Hy, A=A, B=B) == _tf__x(
        sdm, Y=Hz, Hx=Hx, Hy=Hy, A=A, B=B
    )
    assert _ty(sdm, Hz=Hz, Hx=Hx, Hy=Hy, A=A, B=B) == _tf__y(
        sdm, Y=Hz, Hx=Hx, Hy=Hy, A=A, B=B
    )

    if output_format == "dict":
        return tf
    elif output_format == "mt_metadata":
        # To make an mt_metadata tf, we need to pass period
        msg = "mt_metadata format is coming soon -- work in progress"
        raise NotImplementedError(msg)

        # mtm_tf = TF(period=period)
        # output = {}


def _channel_names(
    station_id: str,
    remote: Optional[str] = "",
    join_char: Optional[str] = "_",
) -> tuple:
    """

    Parameters
    ----------
    sdm: xr.DataArray
        A covariance matrix of the Fourier coefficients for a particular frequency band.
    station_id: str
        The label of the station to get a TF for
    remote: str
        The label of the remote reference station to get a TF for.
    join_char: str
        The character that links the station_id with the component name in the covariance matrix

    Returns
    -------
    tuple: Ex, Ey, Hx, Hy, Hz, A, B the labels associated with the covariance matrix rows and columns
    """
    if remote:
        A = join_char.join((remote, "hx"))
        B = join_char.join((remote, "hy"))
    else:
        A = join_char.join((station_id, "hx"))
        B = join_char.join((station_id, "hy"))

    Ex = join_char.join((station_id, "ex"))
    Ey = join_char.join((station_id, "ey"))
    Hx = join_char.join((station_id, "hx"))
    Hy = join_char.join((station_id, "hy"))
    Hz = join_char.join((station_id, "hz"))
    return Ex, Ey, Hx, Hy, Hz, A, B


def _tf__x(sdm: xr.DataArray, Y: str, Hx: str, Hy: str, A: str, B: str) -> complex:
    """

    Compute the transfer function tensor element associated with Z_xx, Z_yx, or T_x:


                 <Y,A*><Hx,B*> -  <Y,B*><Hx,A*>
     tf__x =    ------------------------------            Generalized form of Vozoff, 1991, equation 42a
                <Hy,A*><Hx,B*> - <Hy,B*><Hx,A*>

    Parameters
    ----------
    sdm: xr.DataArray
        A covariance matrix of the Fourier coefficients for a particular frequency band.
    Y: str
        The label for the "output" channel at the station in the regression formula, as referenced in the sdm.
        This label should be associated with one of the station channels ["ex", "ey", or "hz"].
    Hx: str
        The label for accessing the hx data of the station associated with the TF element from the covariance matrix.
    Hy: str
        The label for accessing the hy data of the station associated with the TF element from the covariance matrix.
    A: str
        The label for accessing the first reference channel data.  For single station this is normally hx, for remote reference this is normally hx at the reference station.
    B: str
        The label for accessing the first reference channel data.  For single station this is normally hy, for remote reference this is normally hy at the reference station.

    Returns
    -------
    complex: transfer function tensor element Z_xx, Z_yx, or T_x

    """
    numerator = sdm.loc[Y, A] * sdm.loc[Hy, B] - sdm.loc[Y, B] * sdm.loc[Hy, A]
    denominator = sdm.loc[Hx, A] * sdm.loc[Hy, B] - sdm.loc[Hx, B] * sdm.loc[Hy, A]
    return numerator / denominator


def _tf__y(sdm: xr.DataArray, Y: str, Hx: str, Hy: str, A: str, B: str) -> complex:
    """
    Compute the transfer function tensor element associated with Z_xy, Z_yy, or T_y:

                 <Y,A*><Hx,B*> - <Y,B*><Hx,A*>
     tf__y =    ------------------------------        Generalized form of Vozoff, 1991, equation 42b
                <Hy,A*><Hx,B*> - <Hy,B*><Hx,A*>

    Parameters
    ----------
    sdm: xr.DataArray
        A covariance matrix of the Fourier coefficients for a particular frequency band.
    Y: str
        The label for the "output" channel at the station in the regression formula, as referenced in the sdm.
        This label should be associated with one of the station channels ["ex", "ey", or "hz"].
    Hx: str
        The label for accessing the hx data of the station associated with the TF element from the covariance matrix.
    Hy: str
        The label for accessing the hy data of the station associated with the TF element from the covariance matrix.
    A: str
        The label for accessing the first reference channel data.  For single station this is normally hx, for remote reference this is normally hx at the reference station.
    B: str
        The label for accessing the first reference channel data.  For single station this is normally hy, for remote reference this is normally hy at the reference station.

    Returns
    -------
    complex: transfer function tensor element Z_xy, Z_yy, or T_y:
    """
    numerator = sdm.loc[Y, A] * sdm.loc[Hx, B] - sdm.loc[Y, B] * sdm.loc[Hx, A]
    denominator = sdm.loc[Hy, A] * sdm.loc[Hx, B] - sdm.loc[Hy, B] * sdm.loc[Hx, A]
    return numerator / denominator


# =============================================================================
# test support methods
# =============================================================================
# TODO consider adding these to test_utils


def _zxx(sdm: xr.DataArray, Ex: str, Hx: str, Hy: str, A: str, B: str) -> complex:
    """

    Compute the transfer function tensor element Z_xx given by:

                <Ex,A*><Hy,B*> - <Ex,B*><Hy,A*>
     Z_xx =     ------------------------------       Equation V91_42a
                <Hx,A*><Hy,B*> - <Hx,B*><Hy,A*>

    Parameters
    ----------
    sdm: xr.DataArray
        A covariance matrix of the Fourier coefficients for a particular frequency band.
    Ex: str
        The label for accessing the ex data of the station associated with the TF element from the covariance matrix.
    Hx: str
        The label for accessing the hx data of the station associated with the TF element from the covariance matrix.
    Hy: str
        The label for accessing the hy data of the station associated with the TF element from the covariance matrix.
    A: str
        The label for accessing the first reference channel data.  For single station this is normally hx, for remote reference this is normally hx at the reference station.
    B: str
        The label for accessing the first reference channel data.  For single station this is normally hy, for remote reference this is normally hy at the reference station.

    Returns
    -------
    complex: transfer function tensor element Z_xx
    """
    numerator = sdm.loc[Ex, A] * sdm.loc[Hy, B] - sdm.loc[Ex, B] * sdm.loc[Hy, A]
    denominator = sdm.loc[Hx, A] * sdm.loc[Hy, B] - sdm.loc[Hx, B] * sdm.loc[Hy, A]
    return numerator / denominator


def _zxy(sdm: xr.DataArray, Ex: str, Hx: str, Hy: str, A: str, B: str) -> complex:
    """
    Compute the transfer function tensor element Z_xy given by:

                <Ex,A*><Hx,B*> - <Ex,B*><Hx,A*>
     Z_xy =     ------------------------------        Equation V91_42b
                <Hy,A*><Hx,B*> - <Hy,B*><Hx,A*>

    **Note**: There is a typo in Vozoff 1991, Equation 42b.
    In the denominator, the final term <Hx,A*> is published as <Hy,A*>.

    Parameters
    ----------
    sdm: xr.DataArray
        A covariance matrix of the Fourier coefficients for a particular frequency band.
    Ex: str
        The label for accessing the ex data of the station associated with the TF element from the covariance matrix.
    Hx: str
        The label for accessing the hx data of the station associated with the TF element from the covariance matrix.
    Hy: str
        The label for accessing the hy data of the station associated with the TF element from the covariance matrix.
    A: str
        The label for accessing the first reference channel data.  For single station this is normally hx, for remote reference this is normally hx at the reference station.
    B: str
        The label for accessing the first reference channel data.  For single station this is normally hy, for remote reference this is normally hy at the reference station.

    Returns
    -------
    complex: transfer function tensor element Z_xy

    """
    numerator = sdm.loc[Ex, A] * sdm.loc[Hx, B] - sdm.loc[Ex, B] * sdm.loc[Hx, A]
    denominator = sdm.loc[Hy, A] * sdm.loc[Hx, B] - sdm.loc[Hy, B] * sdm.loc[Hx, A]
    return numerator / denominator


def _zyx(sdm: xr.DataArray, Ey: str, Hx: str, Hy: str, A: str, B: str) -> complex:
    """

    Compute the transfer function tensor element Z_yx given by:

                <Ey,A*><Hy,B*> - <Ey,B*><Hy,A*>
     Z_yx =     ------------------------------         Equation V91_42c
                <Hx,A*><Hy,B*> - <Hx,B*><Hy,A*>

    Parameters
    ----------
    sdm: xr.DataArray
        A covariance matrix of the Fourier coefficients for a particular frequency band.
    Ex: str
        The label for accessing the ex data of the station associated with the TF element from the covariance matrix.
    Hx: str
        The label for accessing the hx data of the station associated with the TF element from the covariance matrix.
    Hy: str
        The label for accessing the hy data of the station associated with the TF element from the covariance matrix.
    A: str
        The label for accessing the first reference channel data.  For single station this is normally hx, for remote reference this is normally hx at the reference station.
    B: str
        The label for accessing the first reference channel data.  For single station this is normally hy, for remote reference this is normally hy at the reference station.

    Returns
    -------
    complex: transfer function tensor element Z_yx

    """
    numerator = sdm.loc[Ey, A] * sdm.loc[Hy, B] - sdm.loc[Ey, B] * sdm.loc[Hy, A]
    denominator = sdm.loc[Hx, A] * sdm.loc[Hy, B] - sdm.loc[Hx, B] * sdm.loc[Hy, A]
    return numerator / denominator


def _zyy(sdm: xr.DataArray, Ey: str, Hx: str, Hy: str, A: str, B: str) -> complex:
    """
    Compute the transfer function tensor element Z_yy given by:

                <Ey,A*><Hx,B*> - <Ey,B*><Hx,A*>
     Z_yy =     ------------------------------          Equation V91_42d
                <Hy,A*><Hx,B*> - <Hy,B*><Hx,A*>

    Parameters
    ----------
    sdm: xr.DataArray
        A covariance matrix of the Fourier coefficients for a particular frequency band.
    Ex: str
        The label for accessing the ex data of the station associated with the TF element from the covariance matrix.
    Hx: str
        The label for accessing the hx data of the station associated with the TF element from the covariance matrix.
    Hy: str
        The label for accessing the hy data of the station associated with the TF element from the covariance matrix.
    A: str
        The label for accessing the first reference channel data.  For single station this is normally hx, for remote reference this is normally hx at the reference station.
    B: str
        The label for accessing the first reference channel data.  For single station this is normally hy, for remote reference this is normally hy at the reference station.

    Returns
    -------
    complex: transfer function tensor element Z_yy

    """
    numerator = sdm.loc[Ey, A] * sdm.loc[Hx, B] - sdm.loc[Ey, B] * sdm.loc[Hx, A]
    denominator = sdm.loc[Hy, A] * sdm.loc[Hx, B] - sdm.loc[Hy, B] * sdm.loc[Hx, A]
    return numerator / denominator


def _tx(sdm: xr.DataArray, Hz: str, Hx: str, Hy: str, A: str, B: str) -> complex:
    """
    Compute the transfer function tensor element T_x given by:

                <Hz,A*><Hy,B*> - <Hz,B*><Hy,A*>
     T_x  =     ------------------------------
                <Hx,A*><Hy,B*> - <Hx,B*><Hy,A*>

    Parameters
    ----------
    sdm: xr.DataArray
        A covariance matrix of the Fourier coefficients for a particular frequency band.
    Ex: str
        The label for accessing the ex data of the station associated with the TF element from the covariance matrix.
    Hx: str
        The label for accessing the hx data of the station associated with the TF element from the covariance matrix.
    Hy: str
        The label for accessing the hy data of the station associated with the TF element from the covariance matrix.
    A: str
        The label for accessing the first reference channel data.  For single station this is normally hx, for remote reference this is normally hx at the reference station.
    B: str
        The label for accessing the first reference channel data.  For single station this is normally hy, for remote reference this is normally hy at the reference station.

    Returns
    -------
    complex: transfer function tensor element T_x

    """
    numerator = sdm.loc[Hz, A] * sdm.loc[Hy, B] - sdm.loc[Hz, B] * sdm.loc[Hy, A]
    denominator = sdm.loc[Hx, A] * sdm.loc[Hy, B] - sdm.loc[Hx, B] * sdm.loc[Hy, A]
    return numerator / denominator


def _ty(sdm: xr.DataArray, Hz: str, Hx: str, Hy: str, A: str, B: str) -> complex:
    """
    Compute the transfer function tensor element T_y given by:

                <Hz,A*><Hx,B*> - <Hz,B*><Hx,A*>
     T_y =       ------------------------------
                <Hy,A*><Hx,B*> - <Hy,B*><Hx,A*>

    Parameters
    ----------
    sdm: xr.DataArray
        A covariance matrix of the Fourier coefficients for a particular frequency band.
    Ex: str
        The label for accessing the ex data of the station associated with the TF element from the covariance matrix.
    Hx: str
        The label for accessing the hx data of the station associated with the TF element from the covariance matrix.
    Hy: str
        The label for accessing the hy data of the station associated with the TF element from the covariance matrix.
    A: str
        The label for accessing the first reference channel data.  For single station this is normally hx, for remote reference this is normally hx at the reference station.
    B: str
        The label for accessing the first reference channel data.  For single station this is normally hy, for remote reference this is normally hy at the reference station.

    Returns
    -------
    complex: transfer function tensor element T_y

    """
    numerator = sdm.loc[Hz, A] * sdm.loc[Hx, B] - sdm.loc[Hz, B] * sdm.loc[Hx, A]
    denominator = sdm.loc[Hy, A] * sdm.loc[Hx, B] - sdm.loc[Hy, B] * sdm.loc[Hx, A]
    return numerator / denominator
