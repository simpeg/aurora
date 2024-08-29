"""
This module provides a TransferFunctionCollection container to hold:
1. TransferFunctionHeader
2. Dictionary of TransferFunction Objects

Development Notes:
 This class could use substantial updating.  There are two main issues to be aware of.

 First, the transfer function objects that is "collects" are themselves things
 that could use substantial updates.  This is because the aurora TransferFunction
 objects to not use the mt_metadata.transfer_function.TF and they should.  This
 is issue #203.

 Second, the Collection can have more than one value for a TF at a given
 frequency or period because the TFs can be estimated from different
 decimation levels.

 A single transfer function object is associated with a station, here called
 the "local_station". In a database of TFs could add a column for
 local_station and one for reference station.
"""
import pathlib

import numpy as np
import xarray as xr

from aurora.config.metadata.processing import Processing
from aurora.sandbox.io_helpers.zfile_murphy import ZFile
from aurora.transfer_function.plot.rho_phi_helpers import plot_phi
from aurora.transfer_function.plot.rho_phi_helpers import plot_rho
from aurora.general_helper_functions import FIGURES_PATH
from loguru import logger
from typing import Optional, Union

EMTF_REGRESSION_ENGINE_LABELS = {}
EMTF_REGRESSION_ENGINE_LABELS["RME"] = "Robust Single Station"
EMTF_REGRESSION_ENGINE_LABELS["RME_RR"] = "Robust Remote Reference"


class TransferFunctionCollection(object):
    def __init__(
        self,
        tf_dict: Optional[Union[dict, None]] = None,
        processing_config: Optional[Union[Processing, None]] = None,
    ):
        """
        Constructor.

        Parameters
        ----------
        kwargs

        tf_dict: dict
            This is a dictionary of TTFZ objects, one per decimation level.  They are
            keyed by the decimation_level_id, usually integers 0, 1, 2... n_dec
        processing_config: Processing
            The processing parameters associated with the TF objects in the dict.

        """
        self.tf_dict = tf_dict
        self.processing_config = processing_config
        self.merged_tf = None
        self.merged_cov_nn = None
        self.merged_cov_ss_inv = None

    @property
    def header(self) -> dict:
        """returns the tf_header (ordered dict) from the first tf in the dict"""
        return self.tf_dict[0].tf_header

    @property
    def local_station_id(self) -> str:
        """return the local station id"""
        return self.header.local_station.id

    @property
    def remote_station_id(self) -> str:
        """return the remote station id"""
        if self.header.remote_station:
            return self.header.remote_station[0].id
        else:
            return ""

    @property
    def total_number_of_frequencies(self) -> int:
        """
        Returns int:
            Count of number of frequencies over all TFs in dict
        """
        num_frequecies = 0
        for dec_level in self.tf_dict.keys():
            num_frequecies += len(self.tf_dict[dec_level].periods)
        return num_frequecies

    @property
    def channel_list(self) -> list:
        """
        Gets all channel names.
         - assumes the first TF in dict is representative of all

        Returns
        -------
        all_channels: list
            All the channel names for the first TF
        """
        tf_input_chs = self.tf_dict[0].transfer_function.input_channel.data.tolist()
        tf_output_chs = self.tf_dict[0].transfer_function.output_channel.data.tolist()
        all_channels = tf_input_chs + tf_output_chs
        return all_channels

    @property
    def total_number_of_channels(self) -> int:
        """Get a count of all channels (input and output) in the TF"""
        num_channels = 0
        num_channels += len(self.header.input_channels)
        num_channels += len(self.header.output_channels)
        return num_channels

    @property
    def number_of_decimation_levels(self) -> int:
        """
        Returns a count of number of decimation levels
        - assumes that each entry in dict is a separate decimation level.
        """
        return len(self.tf_dict)

    def get_merged_dict(self, channel_nomenclature) -> dict:
        """
        merges all decimation levels and returns a dict

        Returns
        -------
        output: dict
            merged tf values
            - this can be used to assign values to an mt_metadata.transfer_function.TF
        """
        output = {}
        self._merge_decimation_levels()
        self.check_all_channels_present(channel_nomenclature)
        output["tf"] = self.merged_tf
        output["cov_ss_inv"] = self.merged_cov_ss_inv
        output["cov_nn"] = self.merged_cov_nn
        return output

    def _merge_decimation_levels(self) -> None:
        """
        Merge all decimation levels into a single xarray for export.

        Development Notes:
         The output of this may become its own class, MergedTransferFunction

         One concern here is that the same period can be estimted at more then one
         decimation level, making the frequency or period axis not of the same order
         as the number of estimates.  Ways around this:
         1. We can multi-index the period axis with decimation level
         2. Make sure that the processing config does not dupliacate periods
         3. Take the average estimate over all periods heving the same value
         4. Drop all but one estimate accoring to a rule (keep most observations say)

         Does TFXML support multiple estimates? How is it dealt with there?
         There is an issue here -

        Flow:
        starting from self.tf_dict, cast each decimation level's tf, which is
        an xr.DataArray to xr.Dataset, forming a list of datasets.
        This dataset list is then combined over all periods forming a representation
        of the TTFZ which is merged over all the decimation levels.

        result is to build:
        merged_tf
        merged_cov_ss_inv
        merged_cov_nn
        """
        n_dec = self.number_of_decimation_levels

        tmp = [self.tf_dict[i].tf.to_dataset("period") for i in range(n_dec)]
        merged_tf = xr.combine_by_coords(tmp)
        merged_tf = merged_tf.to_array(dim="period")
        self.merged_tf = merged_tf

        tmp = [self.tf_dict[i].cov_ss_inv.to_dataset("period") for i in range(n_dec)]
        merged_cov_ss_inv = xr.combine_by_coords(tmp)
        merged_cov_ss_inv = merged_cov_ss_inv.to_array("period")
        self.merged_cov_ss_inv = merged_cov_ss_inv

        tmp = [self.tf_dict[i].cov_nn.to_dataset("period") for i in range(n_dec)]
        merged_cov_nn = xr.combine_by_coords(tmp)
        merged_cov_nn = merged_cov_nn.to_array("period")
        self.merged_cov_nn = merged_cov_nn

        return

    def check_all_channels_present(self, channel_nomenclature) -> None:
        """
        Checks if TF has tipper.  If not, fill in the tipper data with NaN and also
         update the noise covariance matrix so shape is as expected by mt_metadata.

        Development Notes:
            For testing, at least some Parkfield tests have no hz.
            Synthetic tests do not (currently) have missing hz.

        Parameters
        ----------
        channel_nomenclature: mt_metadata.transfer_functions.processing.aurora.channel_nomenclature.ChannelNomenclature
            Scheme according to how channels are named

        """
        ex, ey, hx, hy, hz = channel_nomenclature.unpack()
        if hz not in self.merged_tf.output_channel.data.tolist():
            output_channels_original = self.merged_tf.output_channel.data.tolist()
            output_channels = [f"tmp___{x}" for x in output_channels_original]
            output_channels[0] = hz
            tmp = self.merged_tf.copy(deep=True)
            tmp = tmp.assign_coords({"output_channel": output_channels})
            tmp.data *= np.nan
            tmp = tmp.to_dataset("period")
            tmp = tmp.merge(self.merged_tf.to_dataset("period"))
            tmp = tmp.to_array("period")
            output_channels = tmp.output_channel.data.tolist()
            output_channels = [x for x in output_channels if x[0:6] != "tmp___"]
            tmp = tmp.sel(output_channel=output_channels)
            self.merged_tf = tmp

            n_output_ch = len(self.merged_tf.output_channel)  # this should be 3
            n_periods = len(self.merged_tf.period)
            cov_nn_dims = (n_output_ch, n_output_ch, n_periods)
            cov_nn_data = np.zeros(cov_nn_dims, dtype=np.complex128)

            cov_nn = xr.DataArray(
                cov_nn_data,
                dims=["output_channel_1", "output_channel_2", "period"],
                coords={
                    "output_channel_1": self.merged_tf.output_channel.data,
                    "output_channel_2": self.merged_tf.output_channel.data,
                    "period": self.merged_tf.period,
                },
            )

            # to dataset and back makes the coords line up -- this does not seem robust
            cov_nn = cov_nn.to_dataset("period").to_array("period")

            for out_ch_1 in cov_nn.output_channel_1.data.tolist():
                for out_ch_2 in cov_nn.output_channel_1.data.tolist():
                    try:
                        values = self.merged_cov_nn.loc[:, out_ch_1, out_ch_2]
                        cov_nn.loc[:, out_ch_1, out_ch_2] = values
                    except KeyError:
                        pass
            self.merged_cov_nn = cov_nn
        if ex not in self.merged_tf.output_channel.data.tolist():
            # placeholder: logic for missing ex would go here
            msg = "No handling for missing ex channel"
            raise NotImplementedError(msg)
        if ey not in self.merged_tf.output_channel.data.tolist():
            # placeholder: logic for missing ey would go here
            msg = "No handling for missing ey channel"
            raise NotImplementedError(msg)

    def rho_phi_plot(
        self,
        xy_or_yx: str,
        show: Optional[bool] = True,
        aux_data: Optional[Union[ZFile, None]] = None,
        ttl_str: Optional[str] = "",
        x_axis_fontsize: Optional[float] = 25,
        y_axis_fontsize: Optional[float] = 25,
        ttl_fontsize: Optional[float] = 16,
        markersize: Optional[float] = 10,
        rho_ylims: Optional[list] = [10, 1000],
        phi_ylims: Optional[list] = [0, 90],
        figures_path: Optional[Union[str, pathlib.Path]] = FIGURES_PATH,
        figure_basename: Optional[str] = "",
    ) -> None:
        """
        One-off plotting method intended only for the synthetic test data for aurora dev

        Parameters
        ----------
        xy_or_yx: str
            Must be "xy" or "yx" -- the mode of the TF to plot
        show: Optional[bool]
            If True show figure on screen
        aux_data: Optional[Union[ZFile, None]]
            If not none, plot the data from the z-file as well as data from self
        ttl_str: Optional[str]
            If provided, ttl_str will be appended to the plot title
        x_axis_fontsize: int
            Font size for the x-axis
        y_axis_fontsize: int
            Font size for the y-axis
        ttl_fontsize: int
            Font size for the title
        markersize: int
            the marker size for the plot symbols
        rho_ylims: list
            The lower and upper limits for the y-axis on rho plot
        phi_ylims: list
            The lower and upper limits for the y-axis on phi plot
        figures_path: Union[str, pathlib.Path]
            Where to save the figure
        figure_basename:
            What the name of the figure will be.
            If not given, a default name is assigned

        """

        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(nrows=2, figsize=(11, 8.5), dpi=300)

        color_cyc = {0: "r", 1: "orange", 2: "g", 3: "b", 4: "k"}
        for i_dec in self.tf_dict.keys():
            tf = self.tf_dict[i_dec]

            if xy_or_yx == "xy":
                aurora_rho = tf.rho[:, 0]
            else:
                aurora_rho = tf.rho[:, 1]
            plot_rho(
                axs[0],
                tf.periods,
                aurora_rho,
                marker="o",
                color=color_cyc[i_dec],
                linestyle="None",
                label=f"aurora {i_dec}",
                markersize=markersize,
            )

            if xy_or_yx == "xy":
                aurora_phi = tf.phi[:, 0]
            else:
                aurora_phi = tf.phi[:, 1]
            # rotate phases so all are positive:
            negative_phi_indices = np.where(aurora_phi < 0)[0]
            aurora_phi[negative_phi_indices] += 180.0
            plot_phi(
                axs[1],
                tf.periods,
                aurora_phi,
                marker="o",
                color=color_cyc[i_dec],
                linestyle="None",
                label=f"aurora {i_dec}",
                markersize=markersize,
            )

        if aux_data:
            decimation_levels = list(set(aux_data.decimation_levels))
            shape_cyc = {0: "s", 1: "v", 2: "*", 3: "^"}
            if xy_or_yx == "xy":
                emtf_rho = aux_data.rxy
                emtf_phi = aux_data.pxy
            elif xy_or_yx == "yx":
                emtf_rho = aux_data.ryx
                emtf_phi = aux_data.pyx

            axs[0].loglog(axs[0].get_xlim(), 100 * np.ones(2), color="k")
            axs[1].semilogx(axs[1].get_xlim(), 45 * np.ones(2), color="k")
            for i_dec in decimation_levels:

                ndx = np.where(aux_data.decimation_levels == i_dec)[0]
                axs[0].loglog(
                    aux_data.periods[ndx],
                    emtf_rho[ndx],
                    marker=shape_cyc[i_dec - 1],
                    color="k",
                    linestyle="None",
                    label=f"emtf " f"{int(i_dec-1)}",
                    markersize=markersize,
                )
                axs[1].semilogx(
                    aux_data.periods[ndx],
                    emtf_phi[ndx],
                    marker=shape_cyc[i_dec - 1],
                    color="k",
                    linestyle="None",
                    markersize=markersize,
                    label=f"emtf " f"{int(i_dec-1)}",
                )

        axs[0].legend(ncol=2)
        axs[1].legend(ncol=2)

        axs[1].set_xlabel("Period (s)", fontsize=x_axis_fontsize)
        axs[0].set_ylabel(r"$\Omega$-m", fontsize=y_axis_fontsize)
        axs[1].set_ylabel("Degrees", fontsize=y_axis_fontsize)

        ttl_str = f"{tf.tf_header.local_station.id} {xy_or_yx} \n{ttl_str}"
        axs[0].set_title(ttl_str, fontsize=ttl_fontsize)
        if rho_ylims is not None:
            axs[0].set_ylim(rho_ylims)
        if phi_ylims is not None:
            axs[1].set_ylim(phi_ylims)

        if figures_path is None:
            logger.info("figures path is not defined -- skipping saving figures")
            return
        else:
            if not figure_basename:
                figure_basename = f"{self.local_station_id}_{xy_or_yx}.png"
            out_file = figures_path.joinpath(figure_basename)
            plt.savefig(out_file)
            if show:
                plt.show()
            plt.close(fig)
