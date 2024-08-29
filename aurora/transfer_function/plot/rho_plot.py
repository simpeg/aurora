"""
    This module contains functions for plotting apparent resistivity and phase.

This is based on Gary's RhoPlot.m in the matlab EMTF version. iris_mt_scratch/egbert_codes-20210121T193218Z-001/egbert_codes/matlabPrototype_10-13-20/TF/classes

TODO: replace with calls to mtpy
"""
import matplotlib.pyplot as plt
import numpy as np

from aurora.transfer_function.plot.error_bar_helpers import err_log

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

        [xb, yb] = err_log(
            np.transpose(self.tf.periods),
            self.tf.phi[:, 0],
            self.tf.phi_se[:, 0],
            axis_limits,
            log_x_axis=True,
        )

        ax.semilogx(xb, yb, "b-")
        ax.semilogx(self.tf.periods, phi[:, 0], "bo")

        xb, yb = err_log(
            np.transpose(self.tf.periods),
            self.tf.phi[:, 1],
            self.tf.phi_se[:, 1],
            axis_limits,
            log_x_axis=True,
        )
        ax.semilogx(xb, yb, "r-")
        ax.semilogx(self.tf.periods, phi[:, 1], "ro")
        # set(lines, 'LineWidth', 1, 'MarkerSize', 7);
        if pred is not None:
            plt.plot(pred.tf.periods, pred.tf.phi[:, 0], "b-", "linewidth", linewidth)
            plt.plot(pred.tf.periods, pred.tf.phi[:, 1], "r-", "linewidth", linewidth)

        # (lims_ph);
        ax.set_xlim(axis_limits[0], axis_limits[1])
        ax.set_ylim(axis_limits[2], axis_limits[3])
        title_pos_x = np.log(axis_limits[0]) + 0.1 * (
            np.log(axis_limits[1] / axis_limits[0])
        )
        title_pos_x = np.ceil(np.exp(title_pos_x))
        title_pos_y = axis_limits[2] + 0.8 * (axis_limits[3] - axis_limits[2])
        # ttl_str = f"$\phi$ : {self.tf.header.local_station_id}"\
        # + \"PKD"#self.tf.Header.LocalSite.SiteID
        ax.text(title_pos_x, title_pos_y, ttl_str, fontsize=14, fontweight="demi")
        # set(gca, 'FontWeight', 'bold', 'FontSize', 11, 'Xtick', xticks);
        ax.set_xlabel("Period (s)")
        ax.set_ylabel("Degrees")

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
        [xb, yb] = err_log(
            self.tf.periods,
            self.tf.rho[:, 0],
            self.tf.rho_se[:, 0],
            x_axis_limits,
            log_x_axis=True,
        )
        ax.loglog(xb, yb, "b-")

        # plot rho dots
        ax.loglog(self.tf.periods, self.tf.rho[:, 0], "bo")

        [xb, yb] = err_log(
            self.tf.periods,
            self.tf.rho[:, 1],
            self.tf.rho_se[:, 1],
            x_axis_limits,
            log_x_axis=True,
        )
        ax.loglog(xb, yb, "r-")
        ax.loglog(self.tf.periods, self.tf.rho[:, 1], "ro")

        if pred is not None:
            plt.plot(pred.tf.periods, pred.tf.rho[:, 0], "b-", "linewidth", 1.5)
            plt.plot(pred.tf.periods, pred.tf.rho[:, 1], "r-", "linewidth", 1.5)

        # axis(lims_rho);
        ax.set_xlim(x_axis_limits[0], x_axis_limits[1])
        ax.set_ylim(y_axis_limits[0], y_axis_limits[1])

        #
        title_pos_x = np.log(x_axis_limits[0]) + 0.1 * (
            np.log(x_axis_limits[1] / x_axis_limits[0])
        )
        title_pos_x = np.ceil(np.exp(title_pos_x))
        title_pos_y = y_axis_limits[0] + 0.8 * (y_axis_limits[1] - y_axis_limits[0])
        ttl_str = "\u03C1_a : " + ttl_str
        # c_title = "$\rho_a$ :" + "PKD"  # obj.tf.Header.LocalSite.SiteID
        ax.text(title_pos_x, title_pos_y, ttl_str, fontsize=14, fontweight="demi")
        # set(gca, 'FontWeight', 'bold', 'FontSize', 11, 'Xtick', xticks);
        ax.set_xlabel("Period (s)")
        ax.set_ylabel("$\Omega$-m")
        return

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

    # def get_xticks(self):
    #     xticks = 10.0 ** np.arange(-5, 6)
    #     cond1 = xticks >= self.tf.minimum_period
    #     cond2 = xticks <= self.tf.maximum_period
    #     xticks = xticks[cond1 & cond2]
    #     return xticks
