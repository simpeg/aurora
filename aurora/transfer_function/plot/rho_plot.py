"""
based on Gary's
RhoPlot.m in
iris_mt_scratch/egbert_codes-20210121T193218Z-001/egbert_codes/matlabPrototype_10-13-20/TF/classes
"""
import matplotlib.pyplot as plt
import numpy as np

from aurora.transfer_function.plot.error_bar_helpers import err_log

plt.ioff()


class RhoPlot(object):
    """
    TF plotting object class; some methods are only relevant to
    specific types of TFs (or for derived parameters such as rho/phi)

    """

    def __init__(self, tf_obj):
        self.tf = tf_obj

    def phase_plot(self, pred=None):
        """
        simple driver to plot phase only on a separate figures of fixed size

        hfig = phasePlot(obj, pred)
        hfig = figure('Position', [100, 100, 325, 200],
        'PaperPosition',[1, 1, 3.25, 2.], 'Tag', 'Phase Plot');
        axRect = [0.1446    0.2150    0.7604    0.7100];
        if nargin == 2
            phaseSubPlot(obj, hfig, axRect, pred);
        else
            phaseSubPlot(obj, hfig, axRect);
        end

        Returns
        -------

        """
        axRect = [0.1446, 0.2150, 0.7604, 0.7100]
        # plt.figure(22, figsize = (8.5, 11), dpi=300)
        fig, ax = plt.subplots()
        if pred is not None:
            self.phase_sub_plot(ax, axRect, pred=pred)
        else:
            self.phase_sub_plot(ax, axRect)

    def rho_plot2(self, pred=None):
        """
        one off, use tools in MTpy

        """
        axRect = [0.1446, 0.2150, 0.7604, 0.7100]
        # plt.figure(22, figsize = (8.5, 11), dpi=300)
        fig, ax = plt.subplots()
        if pred is not None:
            self.rho_sub_plot(ax, axRect, pred=pred)
        else:
            self.rho_sub_plot(ax, axRect)

    def rho_phi_plot(self, pred=None):
        """

        Parameters
        ----------
        pred

        Returns
        -------

        """
        rects = self.set_figure_size()
        fig, ax = plt.subplots(2, 1, 1)
        # hfig = figure('Position', rects.Screen, 'PaperPosition',
        #                rects.Paper, 'Tag', 'rho-phi plot');

        if pred is not None:
            h_phi = self.phase_sub_plot(ax, rects.phi, pred)
            h_rho = self.rho_sub_plot(ax, rects.rho, pred)
        else:
            h_phi = self.phase_sub_plot(ax, rects.phi)
            h_rho = self.rho_sub_plot(ax, rects.rho)

        H = {"fig": fig, "rho": h_rho, "phi": h_phi}
        return H

    # def rho_plot(self, **kwargs):
    #     """
    #     simple driver to plot rho only on a separate figures of fixed size
    #     function[hfig, hax] = rhoPlot(obj, varargin)
    #     Returns
    #     -------
    #
    #     """
    #     AddPred = 0
    #     inds = 1
    #     # <HANDLE KWARGS>
    #     # n_kwargs = len(**kwargs)
    #     # if np.mod(n_kwargs, 2)==1:
    #     #     print('Optional arguments must occur in pairs')
    #     #     raise Exception
    #     #
    #     # n_kwarg_pairs = n_kwargs/2
    #     # for k in range(0, n_kwargs, 2):
    #     #     option = lower(varargin{k});
    #     #     switch option
    #     #     case 'pred'
    #     #     AddPred = varargin{k + 1};
    #     #     case 'inds'
    #     #     inds = varargin{k + 1};
    #     # <HANDLE KWARGS>
    #     rects = self.set_figure_size()
    #     yFac = 1 - (rects.phi(1) + rects.phi(3))
    #     yMarg = (rects.rho(1) - (rects.phi(1) + rects.phi(3))) / yFac
    #     rectScreen = rects.Screen
    #     rectScreen[3] = rectScreen[3] * yFac
    #     rectPaper = rects.Paper
    #     rectPaper[3] = rectPaper[3] * yFac
    #     rectRho = rects.Rho
    #     rectRho[3] = rectRho[3] / yFac
    #     rectRho[1] = yMarg
    #     hfig = plt.figure(
    #         "Position", rectScreen, "PaperPosition", rectPaper, "Tag", "rho Plot"
    #     )
    #     if AddPred:
    #         hax = self.rho_sub_plot(hfig, rectRho, pred)
    #     else:
    #         if inds == 1:
    #             hax = self.rho_sub_plot(hfig, rectRho)
    #         else:
    #             hax = self.rho_sub_plots(hfig, inds)
    #     # grid on

    def phase_sub_plot(self, ax, ttl_str="", axRect=None, pred=None):
        """
        place a phase subplot on a figure, given figure handle and axis
        postion

        Parameters
        ----------
        self
        ax
        axRect
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
        # xticks = self.get_xticks()

        [xb, yb] = err_log(
            np.transpose(self.tf.periods),
            self.tf.phi[:, 0],
            self.tf.phi_se[:, 0],
            "XLOG",
            axis_limits,
        )
        # figure(ax); #need this?
        # set current axes
        # hax = plt.axes('Position', axRect);
        ax.semilogx(xb, yb, "b-")
        ax.semilogx(self.tf.periods, phi[:, 0], "bo")
        # print("OK, now set linewidth and markersize")
        # set(lines, 'LineWidth', 1, 'MarkerSize', 7);
        # hold on;
        xb, yb = err_log(
            np.transpose(self.tf.periods),
            self.tf.phi[:, 1],
            self.tf.phi_se[:, 1],
            "XLOG",
            axis_limits,
        )
        ax.semilogx(xb, yb, "r-")
        ax.semilogx(self.tf.periods, phi[:, 1], "ro")
        # set(lines, 'LineWidth', 1, 'MarkerSize', 7);
        if pred is not None:
            plt.plot(pred.tf.periods, pred.tf.phi[:, 0], "b-", "linewidth", 2)
            plt.plot(pred.tf.periods, pred.tf.phi[:, 1], "r-", "linewidth", 2)

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

    def rho_sub_plot(self, ax, ttl_str="", axRect=None, pred=None):
        """
        Calls plotrhom, standard plotting routine; uses some other routines in
        EMTF/matlab/Zplt; this version is for putting multiple curves on the
        same plot

        set plotting limits now that rho is known
        Parameters
        ----------
        ax
        axRect
        pred

        Returns
        -------

        """
        lims, orient = self.set_lims()
        # lims_rho = lims[0:3];
        axis_limits = lims[0:4]
        # xticks = self.get_xticks()
        # plot error bars:
        [xb, yb] = err_log(
            self.tf.periods,
            self.tf.rho[:, 0],
            self.tf.rho_se[:, 0],
            "XLOG",
            axis_limits,
        )
        ax.loglog(xb, yb, "b-")

        # plot rho dots
        ax.loglog(self.tf.periods, self.tf.rho[:, 0], "bo")

        [xb, yb] = err_log(
            self.tf.periods,
            self.tf.rho[:, 1],
            self.tf.rho_se[:, 1],
            "XLOG",
            axis_limits,
        )
        ax.loglog(xb, yb, "r-")
        ax.loglog(self.tf.periods, self.tf.rho[:, 1], "ro")

        if pred is not None:
            plt.plot(pred.tf.periods, pred.tf.rho[:, 0], "b-", "linewidth", 1.5)
            plt.plot(pred.tf.periods, pred.tf.rho[:, 1], "r-", "linewidth", 1.5)

        # axis(lims_rho);
        ax.set_xlim(axis_limits[0], axis_limits[1])
        ax.set_ylim(axis_limits[2], axis_limits[3])

        #
        title_pos_x = np.log(axis_limits[0]) + 0.1 * (
            np.log(axis_limits[1] / axis_limits[0])
        )
        title_pos_x = np.ceil(np.exp(title_pos_x))
        title_pos_y = axis_limits[2] + 0.8 * (axis_limits[3] - axis_limits[2])
        ttl_str = "\u03C1_a : " + ttl_str
        # c_title = "$\rho_a$ :" + "PKD"  # obj.tf.Header.LocalSite.SiteID
        ax.text(title_pos_x, title_pos_y, ttl_str, fontsize=14, fontweight="demi")
        # set(gca, 'FontWeight', 'bold', 'FontSize', 11, 'Xtick', xticks);
        ax.set_xlabel("Period (s)")
        ax.set_ylabel("$\Omega$-m")
        return

    def set_period_limits(self):  # function[Tmin, Tmax] = setTlim(obj)
        """
        set nicer period limits for logartihmic period scale plots
        Returns
        -------

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

    def set_lims(self):
        """
        set default limits for plotting; QD, derived from ZPLT
         use max/min limits of periods, rho to set limits

        function[lims, orient] = set_lims(obj)
        Returns
            lims : list
            x_max, x_min, y_max, y_min, 0, 90

            orient: 0

        TODO: maybe set this as a class?
        -------

        """
        xx_min, xx_max = self.set_period_limits()
        y_min = self.tf.rho.min()
        y_min = max(y_min, 1e-20)
        y_max = self.tf.rho.max()
        y_max = max(y_max, 1e-20)

        yy_min = 10 ** (np.floor(np.log10(y_min)))
        if (np.log10(y_min) - np.log10(yy_min)) < 0.15:
            yy_min = 10 ** (np.log10(yy_min) - 0.3)

        yy_max = 10 ** (np.ceil(np.log10(y_max)))
        if (np.log10(yy_max) - np.log10(y_max)) < 0.15:
            yy_max = 10 ** (np.log10(yy_max) + 0.3)

        if abs(yy_max - yy_min) > 1:
            lims = [xx_min, xx_max, yy_min, yy_max, 0, 90]
        else:
            lims = [xx_min, xx_max, 0.01, 1e4, 0, 90]

        orient = 0.0
        return lims, orient

    def set_figure_size(self):
        """
        rects is a dict
        Returns:
            rects : dict
            has keys: "Screen", "Paper", "Rho", "Phi"
        -------

        """
        lims, _ = self.set_lims()
        size_fac = 50
        paperSizeFac = 0.65
        one_dec = 1.6
        xdecs = np.log10(lims(1)) - np.log10(lims(0))
        one_dec = one_dec * 4 / xdecs
        ydecs = np.log10(lims[3]) - np.log10(lims[2])
        paper_width = xdecs * one_dec
        paper_height = (ydecs + 3) * one_dec
        paper_height = min([paper_height, 9])
        rectScreen = [0.5, 0.5, paper_width, paper_height] * size_fac
        rectPaper = [1.0, 1.0, paper_width * paperSizeFac, paper_height * paperSizeFac]

        rectRho = [0.15, 0.15 + 2.3 / (ydecs + 3), 0.8, ydecs / (ydecs + 3) * 0.8]
        rectPhi = [0.15, 0.15, 0.8, 2 / (ydecs + 3) * 0.8]
        rects = {
            "Screen": rectScreen,
            "Paper": rectPaper,
            "Rho": rectRho,
            "Phi": rectPhi,
        }
        return rects

    def get_xticks(self):
        xticks = 10.0 ** np.arange(-5, 6)
        cond1 = xticks >= self.tf.minimum_period
        cond2 = xticks <= self.tf.maximum_period
        xticks = xticks[cond1 & cond2]
        return xticks


def test():
    pass


def main():
    test()


if __name__ == "__main__":
    main()
