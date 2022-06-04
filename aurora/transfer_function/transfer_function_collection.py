"""
This is a container to hold:
1. TransferFunctionHeader
2. Dictionary of TransferFunction Objects

Note that a single transfer function object is associated with a station,
which we call the "local_station".  In a database of TFs we could add a column
for local_station and one for reference station.
"""

import fortranformat as ff
import numpy as np
import xarray as xr

from pathlib import Path

from aurora.transfer_function.emtf_z_file_helpers import (
    make_orientation_block_of_z_file,
)
from aurora.transfer_function.plot.rho_phi_helpers import plot_phi
from aurora.transfer_function.plot.rho_phi_helpers import plot_rho


EMTF_REGRESSION_ENGINE_LABELS = {}
EMTF_REGRESSION_ENGINE_LABELS["RME"] = "Robust Single station"


class TransferFunctionCollection(object):
    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        kwargs

        tf_dict: dict
            This is a dictionary of TTFZ objects, one per decimation level.  They are
            keyed by the decimation_level_id, usually integers 0, 1, 2... n_dec

        """
        self.header = kwargs.get("header", None)
        self.tf_dict = kwargs.get("tf_dict", None)
        self.labelled_tf = None
        self.merged_tf = None
        self.merged_cov_nn = None
        self.merged_cov_ss_inv = None

    @property
    def local_station_id(self):
        """
        TODO: make this take the station_id directly from the header
        Returns
        -------

        """
        return self.tf_dict[0].tf_header.local_station_id

    @property
    def reference_station_id(self):
        """
        TODO: make this take the station_id directly from the header
        Returns
        -------

        """
        return self.tf_dict[0].tf_header.reference_station_id

    @property
    def total_number_of_frequencies(self):
        num_frequecies = 0
        for dec_level in self.tf_dict.keys():
            num_frequecies += len(self.tf_dict[dec_level].periods)
        return num_frequecies

    @property
    def channel_list(self):
        tf_input_chs = self.tf_dict[0].transfer_function.input_channel.data.tolist()
        tf_output_chs = self.tf_dict[0].transfer_function.output_channel.data.tolist()
        all_channels = tf_input_chs + tf_output_chs
        return all_channels

    @property
    def total_number_of_channels(self):
        num_channels = 0
        num_channels += self.header.num_input_channels
        num_channels += self.header.num_output_channels
        return num_channels

    @property
    def number_of_decimation_levels(self):
        return len(self.tf_dict)

    def get_merged_dict(self):
        output = {}
        self._merge_decimation_levels()
        self.check_all_channels_present()
        # self.relabel_merged_decimation_levels_for_export()
        output["tf"] = self.merged_tf
        output["tf_xarray"] = self.labelled_tf
        output["cov_ss_inv"] = self.merged_cov_ss_inv
        output["cov_nn"] = self.merged_cov_nn
        return output

    def _merge_decimation_levels(self):
        """
        Addressing Aurora Issue #93
        Will merge all decimation levels into a single 3D xarray for output.
        The output of this may become its own class, MergedTransferFunction

        One concern here is that the same period can be estiamted at more then one
        decimation level, making the frequency or period axis not of the same order
        as the number of estimates.  Ways around this:
         1. We can stack that axis with decimation level
         2. Make sure that the processing config does not dupliacate periods
         3. Take the average estimate over all periods heving the same value
         4. Drop all but one estimate accoring to a rule (keep most observations say)

        Does TFXML support multiple estimates? How is it dealt with there?
        There is an issue here -

        Flow:
        starting from self.tf_dict, we cast each decimation level's tf, which is
        nativley an xr.DataArray to xr.Dataset, forming a list of datasets.
        This dataset list is then combined over all periods forming a representation
        of the TTFZ which is merged over all the decimation levels.

        2021-09-25: probably break this into two methods
        The first will crate the merged object, and the second will

        Returns xarray.dataset
        -------

        """

        # <MERGE DECIMATION LEVELS>
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
        # </MERGE DECIMATION LEVELS>

        return

    def check_all_channels_present(self):
        if "hz" not in self.merged_tf.output_channel.data.tolist():
            output_channels_original = self.merged_tf.output_channel.data.tolist()
            output_channels = [f"tmp___{x}" for x in output_channels_original]
            output_channels[0] = "hz"
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

            n_output_ch = len(self.merged_tf.output_channel)  # 3
            n_periods = len(self.merged_tf.period)
            cov_nn_dims = (n_output_ch, n_output_ch, n_periods)
            cov_nn_data = np.zeros(cov_nn_dims, dtype=np.complex128)
            import xarray as xr

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

    def relabel_merged_decimation_levels_for_export(self):
        """
        This method was specifcally related to issue #93, but may not be needed afterall
        Returns
        -------

        """

        if self.merged_tf is None:
            self._merge_decimation_levels()

        # <MAKE XARRAY WITH tzx, tzy, zxx, zxy, zyx, zyy NOMENCLATURE>
        tmp_tipper = self.merged_tf.sel(output_channel="hz")
        tmp_tipper = tmp_tipper.reset_coords(drop="output_channel")
        tmp_tipper = tmp_tipper.to_dataset("input_channel")
        tf_xarray = tmp_tipper.rename({"hx": "tzx", "hy": "tzy"})

        zxx = self.merged_tf.sel(output_channel="ex", input_channel="hx")
        zxx = zxx.reset_coords(drop=["input_channel", "output_channel"])
        zxx = zxx.to_dataset(name="zxx")
        if tf_xarray:
            tf_xarray = tf_xarray.merge(zxx)
        else:
            tf_xarray = zxx
        zxy = self.merged_tf.sel(output_channel="ex", input_channel="hy")
        zxy = zxy.reset_coords(drop=["input_channel", "output_channel"])
        zxy = zxy.to_dataset(name="zxy")
        tf_xarray = tf_xarray.merge(zxy)

        zyx = self.merged_tf.sel(output_channel="ey", input_channel="hx")
        zyx = zyx.reset_coords(drop=["input_channel", "output_channel"])
        zyx = zyx.to_dataset(name="zyx")
        tf_xarray = tf_xarray.merge(zyx)

        zyy = self.merged_tf.sel(output_channel="ey", input_channel="hy")
        zyy = zyy.reset_coords(drop=["input_channel", "output_channel"])
        zyy = zyy.to_dataset(name="zyy")
        tf_xarray = tf_xarray.merge(zyy)
        self.labelled_tf = tf_xarray

        # </MAKE XARRAY WITH tzx, tzy, zxx, zxy, zyx, zyy NOMENCLATURE>

        return

    def write_emtf_z_file(self, z_file_path, run_obj=None, orientation_strs=None):
        """
        Could probably move this into EMTFUtils() class
        Based on EMTF/T/wrt_z.f

        Issues to review:
        This seems to insist that channels be ordered:
        Hx, Hy, Hz, Ex, Ey

        z_file_path : Path or str

        Sample output for a band:
        period :      4.65455    decimation level   1    freq. band from   25 to   30
        number of data point   2489 sampling freq.   1.000 Hz
         Transfer Functions
          0.2498E+00  0.1966E-03  0.3859E-04  0.2519E+00
         -0.1458E-01 -0.2989E-01 -0.7283E+01 -0.7313E+01
          0.7311E+01  0.7338E+01 -0.4087E-01 -0.1031E-01
         Inverse Coherent Signal Power Matrix
          0.3809E-07 -0.6261E-18
         -0.3095E-09  0.4505E-09  0.3764E-07  0.7792E-17
         Residual Covariance
          0.3639E+02  0.0000E+00
         -0.2604E+03  0.2280E+03  0.3090E+05  0.0000E+00
          0.2483E+03  0.2688E+03  0.2660E+03 -0.6791E+03  0.3161E+05  0.0000E+00

        Returns
        -------

        """
        if isinstance(z_file_path, Path):
            parent = z_file_path.parent
            parent.mkdir(exist_ok=True)
        f = open(z_file_path, "w")
        f.writelines(" **** IMPEDANCE IN MEASUREMENT COORDINATES ****\n")
        f.writelines(" ********** WITH FULL ERROR COVARINCE**********\n")

        #processing scheme
        try:
            processing_scheme = EMTF_REGRESSION_ENGINE_LABELS[
                self.header.processing_scheme
            ]
        except KeyError:
            processing_scheme = self.header.processing_scheme

        # data_format = ff.FortranRecordWriter('(a80)')
        # line = f"{data_format.write([processing_scheme])}\n"
        line = f"{processing_scheme}"
        line += (80 - len(line)) * " " + "\n"
        f.writelines(line)

        # <station>
        # format('station    :', a20)
        station_line = f"station    :{self.header.local_station_id}"
        station_line += (32 - len(station_line)) * " " + "\n"
        f.writelines(station_line)
        # </station>

        # <location>
        # 105   format('coordinate ',f9.3,1x,f9.3,1x,' declination ',f8.2)
        # could also use self.header.local_station object here
        if run_obj is None:
            latitude = 1007.996
            longitude = 0.000
            declination = 0.00
        else:
            latitude = run_obj.station_group.metadata.location.latitude
            longitude = run_obj.station_group.metadata.location.longitude
            declination = run_obj.station_group.metadata.location.declination.value
            if declination is None:
                declination = 0.0

        location_str = (
            f"coordinate  {latitude}  {longitude}  declination" f"  {declination}\n"
        )
        f.writelines(location_str)
        # </location>

        # num channels and num frequencies
        # 110   format('number of channels ',i3,2x,' number of frequencies ',i4)
        num_frequencies = self.total_number_of_frequencies
        num_channels_str = f"number of channels   {self.total_number_of_channels}"
        num_frequencies_str = f"number of frequencies   {num_frequencies}"
        out_str = f"{num_channels_str}   {num_frequencies_str}\n"
        f.writelines(out_str)

        #Orientations and tilts
        print("Make the channel list be only the active channels for a z-file")
        assert self.total_number_of_channels == len(self.channel_list)
        ch_list = self.channel_list
        f.writelines(" orientations and tilts of each channel \n")
        if orientation_strs is None:
            orientation_strs = make_orientation_block_of_z_file(
                run_obj, channel_list=ch_list
            )
        f.writelines(orientation_strs)


        f.writelines("\n")

        # <DATA READ>
        # Given that the channel ordering is fixed (hxhyhzexey) and that hxhy
        # are always the input channels, the TF is ordered hzexey or exey
        # depending on 2 or 3 channels.
        # 120   format('period : ',f12.5,3x,' decimation level ',i3,3x,+       '
        # freq. band from ',i4,' to ',i4)

        data_format = ff.FortranRecordWriter("(16E12.4)")
        for i_dec in self.tf_dict.keys():
            tf = self.tf_dict[i_dec]
            tf_xr = tf.transfer_function
            cov_ss_xr = tf.cov_ss_inv
            cov_nn_xr = tf.cov_nn
            periods = tf.frequency_bands.band_centers(frequency_or_period="period")
            periods = np.flip(periods)  # EMTF works in increasing period
            dec_level_config = tf.processing_config.decimations[i_dec]

            for band in tf.frequency_bands.bands(direction="increasing_period"):
                #print(f"band {band}")
                line1 = f"period :      {band.center_period:.5f}    "
                line1 += f"decimation level   {i_dec+1}     "

                sample_rate = dec_level_config.decimation.sample_rate
                num_samples_window = dec_level_config.window.num_samples
                freqs = np.fft.fftfreq(num_samples_window, 1.0 / sample_rate)
                fc_indices = band.fourier_coefficient_indices(freqs)
                fc_indices_str = f"{fc_indices[0]} to   {fc_indices[-1]}"
                line1 += f"freq. band from   {fc_indices_str}\n"
                f.writelines(line1)

                # freq_index = tf.frequency_index(band.center_frequency)
                # num_segments = tf.num_segments.data[0, freq_index]
                period_index = tf.period_index(band.center_period)
                num_segments = tf.num_segments.data[0, period_index]
                line2 = f"number of data point    {int(num_segments)} "
                line2 += f"sampling freq.   {sample_rate} Hz\n"
                f.writelines(line2)


                # write the tf:
                # rows are output channels (hz, ex, ey),
                # columns are input channels (hx, hy)
                f.writelines("  Transfer Functions\n")

                line = ""
                for out_ch in tf.tf_header.output_channels:
                    for inp_ch in tf.tf_header.input_channels:
                        chchtf = tf_xr.loc[out_ch, inp_ch, :]
                        real_part = np.real(chchtf.data[period_index])
                        imag_part = np.imag(chchtf.data[period_index])
                        line += f"{data_format.write([real_part])}"
                        line += f"{data_format.write([imag_part])}"
                    line += "\n"
                f.writelines(line)

                f.writelines("    Inverse Coherent Signal Power Matrix\n")
                line = ""
                for i, inp_ch1 in enumerate(tf.tf_header.input_channels):
                    for inp_ch2 in tf.tf_header.input_channels[: i + 1]:
                        cond1 = cov_ss_xr.input_channel_1 == inp_ch1
                        cond2 = cov_ss_xr.input_channel_2 == inp_ch2
                        chchss = cov_ss_xr.where(cond1 & cond2, drop=True)
                        chchss = chchss.data.squeeze()
                        real_part = np.real(chchss[period_index])
                        imag_part = np.imag(chchss[period_index])
                        line += f"{data_format.write([real_part])}"
                        line += f"{data_format.write([imag_part])}"
                    line += "\n"
                f.writelines(line)

                f.writelines("  Residual Covariance\n")
                line = ""
                for i, out_ch1 in enumerate(tf.tf_header.output_channels):
                    for out_ch2 in tf.tf_header.output_channels[: i + 1]:
                        cond1 = cov_nn_xr.output_channel_1 == out_ch1
                        cond2 = cov_nn_xr.output_channel_2 == out_ch2
                        chchnn = cov_nn_xr.where(cond1 & cond2, drop=True)
                        chchnn = chchnn.data.squeeze()
                        if np.isnan(chchnn[period_index]):
                            real_part = -1.0
                            imag_part = -1.0
                        else:
                            real_part = np.real(chchnn[period_index])
                            imag_part = np.imag(chchnn[period_index])
                        line += f"{data_format.write([real_part])}"
                        line += f"{data_format.write([imag_part])}"
                    line += "\n"
                f.writelines(line)

        f.close()

        return

    def rho_phi_plot(
        self,
        show=True,
        aux_data=None,
        xy_or_yx="xy",
        ttl_str="",
        x_axis_fontsize=25,
        y_axis_fontsize=25,
        ttl_fontsize=16,
        markersize=10,
        rho_ylims=[10, 1000],
        phi_ylims=[0, 90],
        **kwargs,
    ):
        """
        One-off plotting method intended only for the synthetic test data for aurora dev
        Parameters
        ----------
        show
        aux_data
        xy_or_yx
        ttl_str

        Returns
        -------

        """

        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(nrows=2, figsize=(11, 8.5), dpi=300)
        # plotter.rho_sub_plot(axs[0], ttl_str=ttl_str)
        # plotter.phase_sub_plot(axs[1], ttl_str=ttl_str)

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
            #            try:
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

        ttl_str = f"{tf.tf_header.local_station_id} {xy_or_yx} \n{ttl_str}"
        axs[0].set_title(ttl_str, fontsize=ttl_fontsize)
        if rho_ylims is not None:
            axs[0].set_ylim(rho_ylims)
        if phi_ylims is not None:
            axs[1].set_ylim(phi_ylims)

        from aurora.general_helper_functions import FIGURES_PATH

        default_figure_basename = f"{self.local_station_id}_{xy_or_yx}.png"
        figure_basename = kwargs.get("figure_basename", default_figure_basename)
        figure_path = kwargs.get("figure_path", FIGURES_PATH)
        # figure_basename = f"synthetic_{tf.tf_header.local_station_id}_{xy_or_yx}.png"
        out_file = figure_path.joinpath(figure_basename)
        plt.savefig(out_file)
        if show:
            plt.show()
