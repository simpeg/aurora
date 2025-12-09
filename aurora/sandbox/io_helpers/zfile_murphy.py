"""
This module contains a class that was contributed by Ben Murphy for working with EMTF "Z-files"
"""

import pathlib
import re
from typing import Optional, Union

import numpy as np


class ZFile:
    def __init__(self, filename: Union[str, pathlib.Path]):
        """
        Constructor

        Parameters
        ----------
        filename: Union[str, pathlib.Path]
            The path to the z-file.

        """
        self.filename = filename
        self.station = ""
        self.decimation_levels = None
        self.lower_harmonic_indices = None
        self.upper_harmonic_indices = None
        self.f = None

    def open_file(self) -> None:
        """attempt to open file"""
        try:
            self.f = open(self.filename, "r")
        except IOError:
            raise IOError("File not found.")
        return

    def skip_header_lines(self) -> None:
        """Skip over the header when reading"""
        f = self.f
        f.readline()
        f.readline()
        f.readline()
        return

    def get_station_id(self) -> None:
        """get station ID from zfile"""
        f = self.f
        line = f.readline()
        if line.lower().startswith("station"):
            station = line.strip().split(":", 1)[1]
        else:
            station = line.strip()
        self.station = station

    def read_coordinates_and_declination(self) -> None:
        """read coordinates and declination"""
        f = self.f
        line = f.readline().strip().lower()
        match = re.match(
            r"\s*coordinate\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+"
            r"declination\s+(-?\d+\.?\d*)",
            line,
        )
        self.coordinates = (float(match.group(1)), float(match.group(2)))
        self.declination = float(match.group(3))
        return

    def read_number_of_channels_and_number_of_frequencies(self):
        """read_number_of_channels_and_number_of_frequencies"""
        f = self.f
        line = f.readline().strip().lower()
        match = re.match(
            r"\s*number\s+of\s+channels\s+(\d+)\s+number\s+of"
            r"\s+frequencies\s+(\d+)",
            line,
        )
        nchannels = int(match.group(1))
        nfreqs = int(match.group(2))
        self.nchannels = nchannels
        self.nfreqs = nfreqs

    def read_channel_information(self):
        """read_channel_information"""
        f = self.f
        self.orientation = np.zeros((self.nchannels, 2))
        self.channels = []
        for i in range(self.nchannels):
            line = f.readline().strip()
            match = re.match(
                r"\s*\d+\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+\w*\s+" r"(\w+)", line
            )
            self.orientation[i, 0] = float(match.group(1))
            self.orientation[i, 1] = float(match.group(2))
            if len(match.group(3)) > 2:
                # sometimes the channel ID comes out with extra stuff
                self.channels.append(match.group(3)[:2].title())
            else:
                self.channels.append(match.group(3).title())
        return

    def load(self):
        """load Z-file and populate attributes of class"""
        self.open_file()
        self.skip_header_lines()
        self.get_station_id()
        self.read_coordinates_and_declination()
        self.read_number_of_channels_and_number_of_frequencies()

        f = self.f

        # skip line
        f.readline()
        self.read_channel_information()
        f.readline()

        # initialize empty arrays for transfer functions
        # note that EMTF, and consequently this code, assumes two predictor
        #    channels (horizontal magnetics)
        # nchannels - 2 therefore is the number of predicted channels
        self.decimation_levels = np.zeros(self.nfreqs)
        self.periods = np.zeros(self.nfreqs)
        self.lower_harmonic_indices = np.zeros(self.nfreqs)
        self.upper_harmonic_indices = np.zeros(self.nfreqs)
        self.transfer_functions = np.zeros(
            (self.nfreqs, self.nchannels - 2, 2), dtype=np.complex64
        )

        # residual covariance -- square matrix with dimension as number of
        # predicted channels
        self.sigma_e = np.zeros(
            (self.nfreqs, self.nchannels - 2, self.nchannels - 2), dtype=np.complex64
        )

        # inverse coherent signal power -- square matrix, with dimension as the
        #    number of predictor channels
        # since EMTF and this code assume N predictors is 2,
        #    this dimension is hard-coded
        self.sigma_s = np.zeros((self.nfreqs, 2, 2), dtype=np.complex64)

        # now read data for each period
        for i in range(self.nfreqs):
            # extract period
            line = f.readline().strip()
            match = re.match(
                r"\s*period\s*:\s+(\d+\.?\d*)\s+" r"decimation\s+level", line
            )
            self.periods[i] = float(match.group(1))

            splitted_line1 = line.split("level")
            splitted_line2 = splitted_line1[1].split("freq")
            self.decimation_levels[i] = int(splitted_line2[0].strip())
            splitted_line1 = line.split("from")
            splitted_line2 = splitted_line1[1].split("to")
            self.lower_harmonic_indices[i] = int(splitted_line2[0].strip())
            self.upper_harmonic_indices[i] = int(splitted_line2[1].strip())
            # skip two lines
            f.readline()
            f.readline()

            # read transfer functions
            for j in range(self.nchannels - 2):
                comp1_r, comp1_i, comp2_r, comp2_i = f.readline().strip().split()
                self.transfer_functions[i, j, 0] = float(comp1_r) + 1.0j * float(
                    comp1_i
                )
                self.transfer_functions[i, j, 1] = float(comp2_r) + 1.0j * float(
                    comp2_i
                )

            # skip label line
            f.readline()

            # read inverse coherent signal power matrix
            val1_r, val1_i = f.readline().strip().split()
            val2_r, val2_i, val3_r, val3_i = f.readline().strip().split()
            self.sigma_s[i, 0, 0] = float(val1_r) + 1.0j * float(val1_i)
            self.sigma_s[i, 1, 0] = float(val2_r) + 1.0j * float(val2_i)
            self.sigma_s[i, 0, 1] = float(val2_r) - 1.0j * float(val2_i)
            self.sigma_s[i, 1, 1] = float(val3_r) + 1.0j * float(val3_i)

            # skip label line
            f.readline()

            # read residual covariance
            for j in range(self.nchannels - 2):
                values = f.readline().strip().split()
                for k in range(j + 1):
                    if j == k:
                        self.sigma_e[i, j, k] = float(values[2 * k]) + 1.0j * float(
                            values[2 * k + 1]
                        )
                    else:
                        self.sigma_e[i, j, k] = float(values[2 * k]) + 1.0j * float(
                            values[2 * k + 1]
                        )
                        self.sigma_e[i, k, j] = float(values[2 * k]) - 1.0j * float(
                            values[2 * k + 1]
                        )

        f.close()

    def impedance(self, angle: Optional[float] = 0.0):
        """
        Compute the Impedance and errors from the transfer function.
        - note u,v are identity matrices if angle=0

        Parameters
        ----------
        angle: float
            The angle about the vertical axis by which to rotate the Z tensor.

        Returns
        -------
        z: np.ndarray
            The impedance tensor
        error: np.ndarray
            Errors for the impedance tensor
        """
        # check to see if there are actually electric fields in the TFs
        if "Ex" not in self.channels and "Ey" not in self.channels:
            raise ValueError(
                "Cannot return apparent resistivity and phase "
                "data because these TFs do not contain electric "
                "fields as a predicted channel."
            )

        # transform the TFs first...
        # build transformation matrix for predictor channels
        #    (horizontal magnetic fields)
        hx_index = self.channels.index("Hx")
        hy_index = self.channels.index("Hy")
        u = np.eye(2, 2)
        u[hx_index, hx_index] = np.cos(
            (self.orientation[hx_index, 0] - angle) * np.pi / 180.0
        )
        u[hx_index, hy_index] = np.sin(
            (self.orientation[hx_index, 0] - angle) * np.pi / 180.0
        )
        u[hy_index, hx_index] = np.sin(
            (self.orientation[hy_index, 0] - angle) * np.pi / 180.0
        )
        u[hy_index, hy_index] = np.cos(
            (self.orientation[hy_index, 0] - angle) * np.pi / 180.0
        )
        u = np.linalg.inv(u)  # Identity if angle=0

        # build transformation matrix for predicted channels (electric fields)
        ex_index = self.channels.index("Ex")
        ey_index = self.channels.index("Ey")
        v = np.eye(self.transfer_functions.shape[1], self.transfer_functions.shape[1])
        v[ex_index - 2, ex_index - 2] = np.cos(
            (self.orientation[ex_index, 0] - angle) * np.pi / 180.0
        )
        v[ey_index - 2, ex_index - 2] = np.sin(
            (self.orientation[ex_index, 0] - angle) * np.pi / 180.0
        )
        v[ex_index - 2, ey_index - 2] = np.cos(
            (self.orientation[ey_index, 0] - angle) * np.pi / 180.0
        )
        v[ey_index - 2, ey_index - 2] = np.sin(
            (self.orientation[ey_index, 0] - angle) * np.pi / 180.0
        )

        # matrix multiplication...
        rotated_transfer_functions = np.matmul(
            v, np.matmul(self.transfer_functions, u.T)
        )
        rotated_sigma_s = np.matmul(u, np.matmul(self.sigma_s, u.T))
        rotated_sigma_e = np.matmul(v, np.matmul(self.sigma_e, v.T))

        # now pull out the impedance tensor
        z = np.zeros((self.periods.size, 2, 2), dtype=np.complex64)
        z[:, 0, 0] = rotated_transfer_functions[:, ex_index - 2, hx_index]  # Zxx
        z[:, 0, 1] = rotated_transfer_functions[:, ex_index - 2, hy_index]  # Zxy
        z[:, 1, 0] = rotated_transfer_functions[:, ey_index - 2, hx_index]  # Zyx
        z[:, 1, 1] = rotated_transfer_functions[:, ey_index - 2, hy_index]  # Zyy

        # and the variance information
        var = np.zeros((self.periods.size, 2, 2))
        var[:, 0, 0] = np.real(
            rotated_sigma_e[:, ex_index - 2, ex_index - 2]
            * rotated_sigma_s[:, hx_index, hx_index]
        )

        var[:, 0, 1] = np.real(
            rotated_sigma_e[:, ex_index - 2, ex_index - 2]
            * rotated_sigma_s[:, hy_index, hy_index]
        )
        var[:, 1, 0] = np.real(
            rotated_sigma_e[:, ey_index - 2, ey_index - 2]
            * rotated_sigma_s[:, hx_index, hx_index]
        )
        var[:, 1, 1] = np.real(
            rotated_sigma_e[:, ey_index - 2, ey_index - 2]
            * rotated_sigma_s[:, hy_index, hy_index]
        )

        error = np.sqrt(var)

        return z, error

    def tippers(self, angle=0.0):
        """compute the tipper from transfer function"""

        # check to see if there is a vertical magnetic field in the TFs
        if "Hz" not in self.channels:
            raise ValueError(
                "Cannot return tipper data because the TFs do not "
                "contain the vertical magnetic field as a "
                "predicted channel."
            )

        # transform the TFs first...
        # build transformation matrix for predictor channels
        #    (horizontal magnetic fields)
        hx_index = self.channels.index("Hx")
        hy_index = self.channels.index("Hy")
        u = np.eye(2, 2)
        u[hx_index, hx_index] = np.cos(
            (self.orientation[hx_index, 0] - angle) * np.pi / 180.0
        )
        u[hx_index, hy_index] = np.sin(
            (self.orientation[hx_index, 0] - angle) * np.pi / 180.0
        )
        u[hy_index, hx_index] = np.cos(
            (self.orientation[hy_index, 0] - angle) * np.pi / 180.0
        )
        u[hy_index, hy_index] = np.sin(
            (self.orientation[hy_index, 0] - angle) * np.pi / 180.0
        )
        u = np.linalg.inv(u)

        # don't need to transform predicated channels (assuming no tilt in Hz)
        hz_index = self.channels.index("Hz")
        v = np.eye(self.transfer_functions.shape[1], self.transfer_functions.shape[1])

        # matrix multiplication...
        rotated_transfer_functions = np.matmul(
            v, np.matmul(self.transfer_functions, u.T)
        )
        rotated_sigma_s = np.matmul(u, np.matmul(self.sigma_s, u.T))
        rotated_sigma_e = np.matmul(v, np.matmul(self.sigma_e, v.T))

        # now pull out tipper information
        tipper = np.zeros((self.periods.size, 2), dtype=np.complex64)
        tipper[:, 0] = rotated_transfer_functions[:, hz_index - 2, hx_index]  # Tx
        tipper[:, 1] = rotated_transfer_functions[:, hz_index - 2, hy_index]  # Ty

        # and the variance/error information
        var = np.zeros((self.periods.size, 2))
        var[:, 0] = np.real(
            rotated_sigma_e[:, hz_index - 2, hz_index - 2]
            * rotated_sigma_s[:, hx_index, hx_index]
        )  # Tx
        var[:, 1] = np.real(
            rotated_sigma_e[:, hz_index - 2, hz_index - 2]
            * rotated_sigma_s[:, hy_index, hy_index]
        )  # Ty
        error = np.sqrt(var)

        return tipper, error

    def apparent_resistivity(self, angle: float = 0.0):
        """compute the apparent resistivity from the impedance."""
        z_tensor, error = self.impedance(angle=angle)
        Zxy = z_tensor[:, 0, 1]
        Zyx = z_tensor[:, 1, 0]
        T = self.periods
        self.rxy = T * (abs(Zxy) ** 2) / 5.0
        self.ryx = T * (abs(Zyx) ** 2) / 5.0
        self.pxy = np.rad2deg(np.arctan(np.imag(Zxy) / np.real(Zxy)))
        self.pyx = np.rad2deg(np.arctan(np.imag(Zyx) / np.real(Zyx)))
        return

    def rho(self, mode):
        """
        Return the apparent resistivity for the given mode.

        Convenience function to help with streamlining synthetic tests - to be
        eventually replaced by functionality in mt_metadata.tf

        Parameters
        ----------
        mode: str
            "xy" or "yx"

        Returns
        -------
        rho
        """
        if mode == "xy":
            return self.rxy
        if mode == "yx":
            return self.ryx

    def phi(self, mode):
        """
        Return the phase for the given mode.

        Convenience function to help with streamlining synthetic tests - to be
        eventually replaced by functionality in mt_metadata.tf
        Parameters
        ----------
        mode: str
            "xy" or "yx"

        Returns
        -------
        phi
        """
        if mode == "xy":
            return self.pxy
        if mode == "yx":
            return self.pyx

    def compare_transfer_functions(
        self,
        other: "ZFile",
        interpolate_to: str = "self",
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> dict:
        """
        Compare transfer functions between two ZFile objects.

        Compares transfer_functions, sigma_e, and sigma_s arrays. If periods
        don't match, interpolates one onto the other.

        Parameters
        ----------
        other: ZFile
            The other ZFile object to compare against
        interpolate_to: str
            Which periods to interpolate to: "self", "other", or "common"
            - "self": interpolate other to self's periods
            - "other": interpolate self to other's periods
            - "common": use only common periods (no interpolation)
        rtol: float
            Relative tolerance for np.allclose, defaults to 1e-5
        atol: float
            Absolute tolerance for np.allclose, defaults to 1e-8

        Returns
        -------
        comparison: dict
            Dictionary containing:
            - "periods_match": bool, whether periods are identical
            - "transfer_functions_close": bool
            - "sigma_e_close": bool
            - "sigma_s_close": bool
            - "max_tf_diff": float, max absolute difference in transfer functions
            - "max_sigma_e_diff": float
            - "max_sigma_s_diff": float
            - "periods_used": np.ndarray of periods used for comparison
        """
        result = {}

        # Check if periods match
        periods_match = np.allclose(self.periods, other.periods, rtol=rtol, atol=atol)
        result["periods_match"] = periods_match

        if periods_match:
            # Direct comparison
            periods_used = self.periods
            tf1 = self.transfer_functions
            tf2 = other.transfer_functions
            se1 = self.sigma_e
            se2 = other.sigma_e
            ss1 = self.sigma_s
            ss2 = other.sigma_s
        else:
            # Need to interpolate
            if interpolate_to == "self":
                periods_used = self.periods
                tf1 = self.transfer_functions
                se1 = self.sigma_e
                ss1 = self.sigma_s
                tf2 = _interpolate_complex_array(
                    other.periods, other.transfer_functions, periods_used
                )
                se2 = _interpolate_complex_array(
                    other.periods, other.sigma_e, periods_used
                )
                ss2 = _interpolate_complex_array(
                    other.periods, other.sigma_s, periods_used
                )
            elif interpolate_to == "other":
                periods_used = other.periods
                tf2 = other.transfer_functions
                se2 = other.sigma_e
                ss2 = other.sigma_s
                tf1 = _interpolate_complex_array(
                    self.periods, self.transfer_functions, periods_used
                )
                se1 = _interpolate_complex_array(
                    self.periods, self.sigma_e, periods_used
                )
                ss1 = _interpolate_complex_array(
                    self.periods, self.sigma_s, periods_used
                )
            elif interpolate_to == "common":
                # Find common periods
                common_mask_self = np.isin(self.periods, other.periods)
                common_mask_other = np.isin(other.periods, self.periods)
                if not np.any(common_mask_self):
                    raise ValueError("No common periods found between the two ZFiles")
                periods_used = self.periods[common_mask_self]
                tf1 = self.transfer_functions[common_mask_self]
                se1 = self.sigma_e[common_mask_self]
                ss1 = self.sigma_s[common_mask_self]
                tf2 = other.transfer_functions[common_mask_other]
                se2 = other.sigma_e[common_mask_other]
                ss2 = other.sigma_s[common_mask_other]
            else:
                raise ValueError(
                    f"interpolate_to must be 'self', 'other', or 'common', got {interpolate_to}"
                )

        result["periods_used"] = periods_used

        # Compare arrays
        result["transfer_functions_close"] = np.allclose(tf1, tf2, rtol=rtol, atol=atol)
        result["sigma_e_close"] = np.allclose(se1, se2, rtol=rtol, atol=atol)
        result["sigma_s_close"] = np.allclose(ss1, ss2, rtol=rtol, atol=atol)

        # Calculate max differences
        result["max_tf_diff"] = np.max(np.abs(tf1 - tf2))
        result["max_sigma_e_diff"] = np.max(np.abs(se1 - se2))
        result["max_sigma_s_diff"] = np.max(np.abs(ss1 - ss2))

        return result


def _interpolate_complex_array(
    periods_from: np.ndarray, array_from: np.ndarray, periods_to: np.ndarray
) -> np.ndarray:
    """
    Interpolate complex array from one period axis to another.

    Uses linear interpolation on real and imaginary parts separately.

    Parameters
    ----------
    periods_from: np.ndarray
        Original periods (1D)
    array_from: np.ndarray
        Original array (can be multi-dimensional, first axis is periods)
    periods_to: np.ndarray
        Target periods (1D)

    Returns
    -------
    array_to: np.ndarray
        Interpolated array with shape (len(periods_to), ...)
    """
    # Handle multi-dimensional arrays
    shape_to = (len(periods_to),) + array_from.shape[1:]
    array_to = np.zeros(shape_to, dtype=array_from.dtype)

    # Flatten all dimensions except the first (periods)
    original_shape = array_from.shape
    array_from_flat = array_from.reshape(original_shape[0], -1)
    array_to_flat = array_to.reshape(shape_to[0], -1)

    # Interpolate each component
    for i in range(array_from_flat.shape[1]):
        # Interpolate real part
        array_to_flat[:, i].real = np.interp(
            periods_to, periods_from, array_from_flat[:, i].real
        )
        # Interpolate imaginary part
        if np.iscomplexobj(array_from):
            array_to_flat[:, i].imag = np.interp(
                periods_to, periods_from, array_from_flat[:, i].imag
            )

    # Reshape back
    array_to = array_to_flat.reshape(shape_to)

    return array_to


def compare_z_files(
    z_file_path1: Union[str, pathlib.Path],
    z_file_path2: Union[str, pathlib.Path],
    angle1: float = 0.0,
    angle2: float = 0.0,
    interpolate_to: str = "self",
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> dict:
    """
    Compare two z-files numerically.

    Loads both z-files and compares their transfer functions, sigma_e, and
    sigma_s arrays. If periods don't match, interpolates one onto the other.

    Parameters
    ----------
    z_file_path1: Union[str, pathlib.Path]
        Path to first z-file
    z_file_path2: Union[str, pathlib.Path]
        Path to second z-file
    angle1: float
        Rotation angle for first z-file, defaults to 0.0
    angle2: float
        Rotation angle for second z-file, defaults to 0.0
    interpolate_to: str
        Which periods to interpolate to: "self" (file1), "other" (file2), or "common"
    rtol: float
        Relative tolerance for comparison, defaults to 1e-5
    atol: float
        Absolute tolerance for comparison, defaults to 1e-8

    Returns
    -------
    comparison: dict
        Dictionary with comparison results including:
        - "periods_match": bool
        - "transfer_functions_close": bool
        - "sigma_e_close": bool
        - "sigma_s_close": bool
        - "max_tf_diff": float
        - "max_sigma_e_diff": float
        - "max_sigma_s_diff": float
        - "periods_used": np.ndarray

    Examples
    --------
    >>> result = compare_z_files("file1.zss", "file2.zss")
    >>> if result["transfer_functions_close"]:
    ...     print("Transfer functions match!")
    >>> print(f"Max difference: {result['max_tf_diff']}")
    """
    zfile1 = read_z_file(z_file_path1, angle=angle1)
    zfile2 = read_z_file(z_file_path2, angle=angle2)

    return zfile1.compare_transfer_functions(
        zfile2, interpolate_to=interpolate_to, rtol=rtol, atol=atol
    )


def read_z_file(z_file_path, angle=0.0) -> ZFile:
    """
    Reads a zFile and returns a ZFile object.

    Parameters
    ----------
    z_file_path: string or pathlib.Path
        The name of the EMTF-style z-file to operate on
    angle: float
        How much rotation to apply.  This is a kludge variable used to help compare
        legacy SPUD results which are rotated onto a cardinal grid, vs aurora which
        store the TF in the coordinate system of acquisition

    Returns
    -------
    z_obj: ZFile
        The zFile as an object.

    """
    z_obj = ZFile(z_file_path)
    z_obj.load()
    z_obj.apparent_resistivity(angle=angle)
    return z_obj
