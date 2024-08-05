"""
    This module contains a class that was contributed by Ben Murphy for working with EMTF "Z-files"
"""
import pathlib
from typing import Optional, Union
import re
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
        u[hy_index, hx_index] = np.cos(
            (self.orientation[hy_index, 0] - angle) * np.pi / 180.0
        )
        u[hy_index, hy_index] = np.sin(
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
