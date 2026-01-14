"""
This module contains an extension of aurora's TransferFunction base class.
This class can return estimates of standard error, apparent resistivity and phase.

Development Notes:
This class follows  Gary's legacy matlab code  TTFZ.m from
iris_mt_scratch/egbert_codes-20210121T193218Z-001/egbert_codes/matlabPrototype_10-13-20/TF/classes
TODO: This should be replaced by methods in mtpy.
"""

import numpy as np
import xarray as xr
from loguru import logger
from matplotlib import pyplot as plt

from aurora.transfer_function.base import TransferFunction


class TTFZ(TransferFunction):
    """
    subclass to support some more MT impedance specific functions  --
    initially just apparent resistivity and phase for diagonal elements.
    + rotation/fixed coordinate system

    TODO: This class should be deprecated and mt_metadata TF object should be used instead.

    """

    def __init__(self, *args, **kwargs):
        """
        Constructor

        Parameters
        ----------
        args: passed through to base class
        kwargs: passed through to base class
        """
        super(TTFZ, self).__init__(*args, **kwargs)

    def standard_error(self):
        """
        estimate the standard error, used for error bars and inversion.

        Development Notes:
        The standard error is normally thought of as the sqrt of the error variance.
        since the code here sets std_err = np.sqrt(np.abs(cov_ss_inv * cov_nn))
        that means the inverse signal covariance times the noise covariance is like the error variance.

        Returns
        -------
        standard_error: xr.DataArray
        """
        stderr = np.zeros(self.tf.data.shape)
        standard_error = xr.DataArray(
            stderr,
            dims=["output_channel", "input_channel", "period"],
            coords={
                "output_channel": self.tf_header.output_channels,
                "input_channel": self.tf_header.input_channels,
                "period": self.periods,
            },
        )
        for out_ch in self.tf_header.output_channels:
            for inp_ch in self.tf_header.input_channels:
                for T in self.periods:
                    cov_ss_inv = self.cov_ss_inv.loc[inp_ch, inp_ch, T]
                    cov_nn = self.cov_nn.loc[out_ch, out_ch, T]
                    std_err = np.sqrt(np.abs(cov_ss_inv * cov_nn))
                    standard_error.loc[out_ch, inp_ch, T] = std_err

        return standard_error

    def apparent_resistivity(self, channel_nomenclature, units="SI"):
        """
        Computes the apparent resistivity and phase.

        Development notes:
        Original Matlab Documentation:
        ap_res(...) : computes app. res., phase, errors, given imped., cov.
        %USAGE: [rho,rho_se,ph,ph_se] = ap_res(z,sig_s,sig_e,periods) ;
        % Z = array of impedances (from Z_***** file)
        % sig_s = inverse signal covariance matrix (from Z_****** file)
        % sig_e = residual covariance matrix (from Z_****** file)
        % periods = array of periods (sec)

        Parameters
        ----------
        units: str
            one of ["MT","SI"]
        channel_nomenclature:
        mt_metadata.processing.aurora.channel_nomenclature.ChannelNomenclature
            has a dict that maps the channel names in TF to the standard channel labellings.

        """
        ex, ey, hx, hy, hz = channel_nomenclature.unpack()
        rad_deg = 180 / np.pi
        # off - diagonal impedances
        self.rho = np.zeros((self.num_bands, 2))
        self.rho_se = np.zeros((self.num_bands, 2))
        self.phi = np.zeros((self.num_bands, 2))
        self.phi_se = np.zeros((self.num_bands, 2))
        Zxy = self.tf.loc[ex, hy, :].data
        Zyx = self.tf.loc[ey, hx, :].data

        # standard deviation of real and imaginary parts of impedance
        Zxy_se = self.standard_error().loc[ex, hy, :].data / np.sqrt(2)
        Zyx_se = self.standard_error().loc[ey, hx, :].data / np.sqrt(2)

        if units == "SI":
            rxy = 2e-7 * self.periods * (abs(Zxy) ** 2)
            ryx = 2e-7 * self.periods * (abs(Zyx) ** 2)
            # print("Correct the standard errors for SI units")
            Zxy_se *= 1e-3
            Zyx_se *= 1e-3
            rxy_se = 2 * np.sqrt(self.periods * rxy / 5) * Zxy_se
            ryx_se = 2 * np.sqrt(self.periods * ryx / 5) * Zyx_se
        elif units == "MT":
            rxy = 2e-1 * self.periods * (abs(Zxy) ** 2)
            ryx = 2e-1 * self.periods * (abs(Zyx) ** 2)
            rxy_se = 2 * np.sqrt(self.periods * rxy / 5) * Zxy_se
            ryx_se = 2 * np.sqrt(self.periods * ryx / 5) * Zyx_se
        else:
            logger.error("ERROR: only SI and MT units supported")
            raise Exception

        self.rho[:, :] = np.vstack((rxy, ryx)).T
        self.rho_se[:, :] = np.vstack((rxy_se, ryx_se)).T

        # phases
        pxy = rad_deg * np.arctan(np.imag(Zxy) / np.real(Zxy))
        pyx = rad_deg * np.arctan(np.imag(Zyx) / np.real(Zyx))
        self.phi[:, :] = np.vstack((pxy, pyx)).T

        pxy_se = rad_deg * Zxy_se / np.abs(Zxy)
        pyx_se = rad_deg * Zyx_se / np.abs(Zyx)

        self.phi_se = np.vstack((pxy_se, pyx_se)).T
        return

    def plot(self, out_filename=None, **kwargs):
        """Plot the transfer function using mtpy's built in plot function."""

        plot_object = RhoPlot(self)
        plt.ion()
        return plot_object.plot(
            station_id=self.tf_header.local_station.id,
            out_filename=out_filename,
            **kwargs,
        )


plt.ioff()


class RhoPlot(object):
    """
    TF plotting object class; some methods are only relevant to
    specific types of TFs (or for derived parameters such as rho/phi)

    Development Notes:
        This should be deprecated and replaced with MTpy
        The only place this class is used is in aurora/sandbox/plot_helpers.py in the
        plot_tf_obj method.

    """

    def __init__(self, tf_obj):
        """
        Constructor

        TODO: Replace tf_obj with mt_metadata tf if this method not replaced with mtpy.

        Parameters
        ----------
        tf_obj: aurora.transfer_function.TTFZ.TTFZ
            Object with TF information


        """
        self.tf = tf_obj
        self._blue = "steelblue"
        self._red = "firebrick"

    def err_log(
        self,
        x: np.ndarray,
        y: np.ndarray,
        yerr: np.ndarray,
        x_axis_limits: list,
        log_x_axis: bool = True,
        barsize: float = 0.0075,
    ):
        """
            Returns the coordinates for the line segments that make up the error bars.

        Development Notes:
        This function returns 6 numbers per data point.
        There is no documentation for what it does.
        A reasonable guess would be that the six numbers define 3 line segments.
        One line segment for the error bar, and one line segment at the top of the error bar, and one at the bottom.
        The vectors xb and yb each have six elements per data point assigned as follows
        xb = [x-dx, x+dx, x, x, x-dx, x+dx,]
        yb = [y-dy, y-dy, y-dy, y+dy, y+dy, y+dy,]
        and if log_x_axis is True
        [log(x)-dx, log(x)+dx, log(x), log(x), log(x)-dx, log(x)+dx,]

        Matlab Documentation
        err_log : used for plotting error bars with a y-axis log scale
        takes VECTORS x and y and outputs matrices (one row per data point) for
        plotting error bars ll = 'XLOG' for log X axis

        Parameters
        ----------
        x : np.ndarray
            The x-axis values.  Usually these are periods with units of seconds
        y : np.ndarray
            The x-axis values.  Usually apparent resistivity or phase
        yerr: np.ndarray
            A value associated with the error in the y measurement.
            It seems that this is the "half height" of the error bar.
        log_x_axis : bool
            If True the xaxis is logarithmic
            Not tested for False
        x_axis_limits: list
            The lower and upper limits for the xaxis in position 0, 1 respectively.
        barsize: float
            The width of the top and bottom horizontal error bar lines.

        Returns
        -------
        xb, yb: tuple
            Each is np.ndarray, 6 rows and one column per data point
            These are the six points needed to draw the error bars.
        """
        num_observations = len(x)
        xb = np.zeros((6, num_observations))
        yb = np.zeros((6, num_observations))
        if log_x_axis:
            dx = (
                np.log(x_axis_limits[1] / x_axis_limits[0]) * barsize
            )  # natural log in matlab & python
            xb[2, :] = np.log(x)
        else:
            dx = (x_axis_limits[1] - x_axis_limits[0]) * barsize
            xb[2, :] = x
        xb[3, :] = xb[2, :]
        xb[0, :] = xb[2, :] - dx
        xb[1, :] = xb[2, :] + dx
        xb[4, :] = xb[2, :] - dx
        xb[5, :] = xb[2, :] + dx

        if log_x_axis:
            xb = np.exp(xb)

        yb[0, :] = (y - yerr).T
        yb[1, :] = (y - yerr).T
        yb[2, :] = (y - yerr).T
        yb[3, :] = (y + yerr).T
        yb[4, :] = (y + yerr).T
        yb[5, :] = (y + yerr).T

        return xb, yb

    def phase_sub_plot(self, ax, ttl_str="", pred=None, linewidth=2):
        """
        place a phase subplot on given figure axis

        Development notes:
         Originally this took an optional input argument `axRect`
         but it was never used. It looks as it it was intended to be able to set the
         position of the figure.  There was also some hardcoded control of linewidth
         and markersize which has been removed for readability.


        Parameters
        ----------
        ax
        pred

        Returns
        -------

        """

        phi = self.tf.phi
        # rotate phases so all are positive:
        negative_phi_indices = np.where(phi < 0)[0]
        phi[negative_phi_indices] += 180.0

        Tmin, Tmax = self.set_period_limits()
        axis_limits = [Tmin, Tmax, 0, 90]

        [xb, yb] = self.err_log(
            np.transpose(self.tf.periods),
            self.tf.phi[:, 0],
            self.tf.phi_se[:, 0],
            axis_limits,
            log_x_axis=True,
        )

        ax.semilogx(xb, yb, ls="-", color=self._blue)
        ax.semilogx(self.tf.periods, phi[:, 0], marker="o", ls="--", color=self._blue)

        xb, yb = self.err_log(
            np.transpose(self.tf.periods),
            self.tf.phi[:, 1],
            self.tf.phi_se[:, 1],
            axis_limits,
            log_x_axis=True,
        )
        ax.semilogx(xb, yb, ls="-", color=self._red)
        ax.semilogx(self.tf.periods, phi[:, 1], marker="o", ls="--", color=self._red)
        # set(lines, 'LineWidth', 1, 'MarkerSize', 7);
        if pred is not None:
            plt.plot(pred.tf.periods, pred.tf.phi[:, 0], "b-")
            plt.plot(pred.tf.periods, pred.tf.phi[:, 1], "r-")

        # (lims_ph);
        ax.set_xlim(axis_limits[0], axis_limits[1])
        ax.set_ylim(axis_limits[2], axis_limits[3])

        # ax.set_subtitle( ttl_str, fontsize=14, fontweight="demi")
        # set(gca, 'FontWeight', 'bold', 'FontSize', 11, 'Xtick', xticks);
        ax.set_xlabel("Period (s)")
        ax.set_ylabel("Degrees")
        return ax

    def rho_sub_plot(self, ax, ttl_str="", pred=None):
        """
        Makes an apparent resistivity plot on the input axis.

        Matlab Documentation:
        Calls plotrhom, standard plotting routine; uses some other routines in
        EMTF/matlab/Zplt; this version is for putting multiple curves on the
        same plot ... set plotting limits now that rho is known


        Parameters
        ----------
        ax: matplotlib.axes._axes.Axes
        pred

        Returns
        -------

        """
        lims = self.set_lims()  # get the axes limits
        x_axis_limits = lims[0:2]
        y_axis_limits = lims[2:4]

        # get and plot error bars:
        [xb, yb] = self.err_log(
            self.tf.periods,
            self.tf.rho[:, 0],
            self.tf.rho_se[:, 0],
            x_axis_limits,
            log_x_axis=True,
        )
        ax.loglog(xb, yb, ls="--", color=self._blue)

        # plot rho dots
        ax.loglog(
            self.tf.periods,
            self.tf.rho[:, 0],
            marker="o",
            ls="--",
            color=self._blue,
            label="$Z_{xy}$",
        )

        [xb, yb] = self.err_log(
            self.tf.periods,
            self.tf.rho[:, 1],
            self.tf.rho_se[:, 1],
            x_axis_limits,
            log_x_axis=True,
        )
        ax.loglog(xb, yb, ls="-", color=self._red)
        ax.loglog(
            self.tf.periods,
            self.tf.rho[:, 1],
            marker="o",
            ls="--",
            color=self._red,
            label="$Z_{yx}$",
        )

        if pred is not None:
            ax.plot(
                pred.tf.periods,
                pred.tf.rho[:, 0],
                "b-",
                label="$Z_{xy}$",
            )
            ax.plot(
                pred.tf.periods,
                pred.tf.rho[:, 1],
                "r-",
                label="$Z_{yx}$",
            )

        # axis(lims_rho);
        ax.set_xlim(x_axis_limits[0], x_axis_limits[1])
        ax.set_ylim(y_axis_limits[0], y_axis_limits[1])
        ax.legend()
        ax.set_ylabel(r"$\Omega$-m")
        return ax

    def set_period_limits(self):
        """
        Returns a set of limits for the x-axis of plots based on periods to display.

        Original Matlab Notes:
            "set nicer period limits for logartihmic period scale plots"

        Returns
        -------
        Tmin, Tmax: tuple
            The minimum and maximum periods for the x-axis
        """

        x_min = self.tf.minimum_period
        x_max = self.tf.maximum_period

        Tmin = 10 ** (np.floor(np.log10(x_min) * 2) / 2)
        if (np.log10(x_min) - np.log10(Tmin)) < 0.15:
            Tmin = 10 ** (np.log10(Tmin) - 0.3)

        Tmax = 10 ** (np.ceil(np.log10(x_max) * 2) / 2)
        if (np.log10(Tmax) - np.log10(x_max)) < 0.15:
            Tmax = 10 ** (np.log10(Tmax) + 0.3)
        return Tmin, Tmax

    def set_rho_limits(self):
        """
        Returns a set of limits for the x-axis of plots based on periods to display.

        Original Matlab Notes:
            "set nicer period limits for logartihmic period scale plots"

        Returns
        -------
        Tmin, Tmax: tuple
            The minimum and maximum periods for the x-axis
        """
        y_min = max(self.tf.rho.min(), 1e-20)
        y_max = max(self.tf.rho.max(), 1e-20)

        yy_min = 10 ** (np.floor(np.log10(y_min)))
        if (np.log10(y_min) - np.log10(yy_min)) < 0.15:
            yy_min = 10 ** (np.log10(yy_min) - 0.3)

        yy_max = 10 ** (np.ceil(np.log10(y_max)))
        if (np.log10(yy_max) - np.log10(y_max)) < 0.15:
            yy_max = 10 ** (np.log10(yy_max) + 0.3)

        return yy_min, yy_max

    def set_lims(self) -> list:
        """
        Set limits for the plotting axes

        TODO: Add doc or start using MTpy

        Matlab Notes:
        set default limits for plotting; QD, derived from ZPLT use max/min limits of periods, rho to set limits

        function[lims, orient] = set_lims(obj)
        Returns
            lims : list
            x_max, x_min, y_max, y_min, 0, 90
            orient: 0

        Returns
        -------
        lims: list
            The plotting limits for period, rho and phi.
        """
        period_min, period_max = self.set_period_limits()  # get limits for the x-axis
        rho_min, rho_max = self.set_rho_limits()
        phi_min = 0
        phi_max = 90

        if abs(rho_max - rho_min) <= 1:
            rho_min = 0.01
            rho_max = 1e4
        lims = [period_min, period_max, rho_min, rho_max, phi_min, phi_max]

        # orient = 0.0
        return lims  # , orient

    def plot(self, station_id="Transfer Function", out_filename=None, **kwargs):
        """
        Plot the apparent resistivity and phase.

        Parameters
        ----------
        station_id: str

        Returns
        -------
        fig: matplotlib.figure.Figure
            The figure object containing the plots
        """
        fig, axs = plt.subplots(nrows=2)
        fig.suptitle(f"Station: {station_id}", fontsize=16, fontweight="demi")

        ax_res = self.rho_sub_plot(axs[0], ttl_str="", pred=None)
        ax_phase = self.phase_sub_plot(axs[1], ttl_str="", pred=None)

        for ax in [ax_res, ax_phase]:
            ax.grid(
                which="both", linestyle="--", linewidth=0.5, color="gray", alpha=0.7
            )
        plt.tight_layout()
        plt.show()

        if out_filename is not None:
            fig.savefig(out_filename, **kwargs)
        return fig
