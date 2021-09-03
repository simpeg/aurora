"""
Placeholder class.  Will probably evolve structure in time.
This is a container to hold:
1. TransferFunctionHeader
2. Dictionary of TransferFunction Objects

Note that a single transfer function object is associated with a station,
which we call the "local_station".  In a database of TFs we could add a column
for local_station and one for reference station.
"""
import fortranformat as ff
import numpy as np
from aurora.transfer_function.emtf_z_file_helpers import (
    make_orientation_block_of_z_file,
)

EMTF_REGRESSION_ENGINE_LABELS = {}
EMTF_REGRESSION_ENGINE_LABELS["RME"] = "Robust Single station"


class TransferFunctionCollection(object):
    def __init__(self, **kwargs):
        self.header = kwargs.get("header", None)
        self.tf_dict = kwargs.get("tf_dict", None)

    @property
    def total_number_of_frequencies(self):
        num_frequecies = 0
        for dec_level in self.tf_dict.keys():
            num_frequecies += len(self.tf_dict[dec_level].periods)
        return num_frequecies

    @property
    def total_number_of_channels(self):
        num_channels = 0
        num_channels += self.header.num_input_channels
        num_channels += self.header.num_output_channels
        return num_channels

    def write_emtf_z_file(self, z_file_path, run_obj=None):
        """
        Could probably move this into EMTFUtils() class
        Based on EMTF/T/wrt_z.f

        Issues to review:
        This seems to insist that channels be ordered:
        Hx, Hy, Hz, Ex, Ey

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
        f = open(z_file_path, "w")
        f.writelines(" **** IMPEDANCE IN MEASUREMENT COORDINATES ****\n")
        f.writelines(" ********** WITH FULL ERROR COVARINCE**********\n")

        # <processing scheme>
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
        # </processing scheme>

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

        # <num channels / num frequencies>
        # 110   format('number of channels ',i3,2x,' number of frequencies ',i4)
        num_frequencies = self.total_number_of_frequencies
        num_channels_str = f"number of channels   {self.total_number_of_channels}"
        num_frequencies_str = f"number of frequencies   {num_frequencies}"
        out_str = f"{num_channels_str}   {num_frequencies_str}\n"
        f.writelines(out_str)
        # </num channels / num frequencies>

        # <Orientations and tilts>
        print("CHANNEL ORIENTATION METADATA NEEDED")
        f.writelines(" orientations and tilts of each channel \n")
        orientation_strs = make_orientation_block_of_z_file(run_obj)
        f.writelines(orientation_strs)
        # </Orientations and tilts>

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
            for band in tf.frequency_bands.bands(direction="increasing_period"):
                line1 = f"period :      {band.center_period:.5f}    "
                line1 += f"decimation level   {i_dec+1}:    "
                # <Make a method of processing config?>
                sample_rate = tf.processing_config.sample_rate
                num_samples_window = tf.processing_config.num_samples_window
                freqs = np.fft.fftfreq(num_samples_window, 1.0 / sample_rate)
                fc_indices = band.fourier_coefficient_indices(freqs)
                # </Make a method of processing config?>
                fc_indices_str = f"{fc_indices[0]} to   {fc_indices[-1]}"
                line1 += f"freq. band from   {fc_indices_str}\n"
                f.writelines(line1)
                freq_index = tf.frequency_index(band.center_frequency)
                num_segments = tf.num_segments.data[0, freq_index]
                line2 = f"number of data point    {int(num_segments)} "
                line2 += f"sampling freq.   {tf.processing_config.sample_rate} Hz\n"
                f.writelines(line2)

                f.writelines("  Transfer Functions\n")
                # write the tf:
                # rows are output channels (hz, ex, ey),
                # columns are input channels (hx, hy)
                period_index = tf.period_index(band.center_period)
                line = ""
                for out_ch in tf.tf_header.output_channels:
                    for inp_ch in tf.tf_header.input_channels:
                        print(out_ch, inp_ch)
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
            axs[0].loglog(
                tf.periods,
                aurora_rho,
                marker="o",
                color=color_cyc[i_dec],
                linestyle="None",
                label=f"aurora {i_dec}",
                markersize=markersize,
            )
            axs[0].tick_params(axis="both", which="major", labelsize=16)
            # ax.tick_params(axis='both', which='minor', labelsize=8)

            if xy_or_yx == "xy":
                aurora_phi = tf.phi[:, 0]
            else:
                aurora_phi = tf.phi[:, 1]
            # rotate phases so all are positive:
            negative_phi_indices = np.where(aurora_phi < 0)[0]
            aurora_phi[negative_phi_indices] += 180.0
            axs[1].semilogx(
                tf.periods,
                aurora_phi,
                marker="o",
                color=color_cyc[i_dec],
                linestyle="None",
                label=f"aurora {i_dec}",
                markersize=markersize,
            )
            axs[1].tick_params(axis="both", which="major", labelsize=16)

        if aux_data:
            #            try:
            decimation_levels = list(set(aux_data.decimation_levels))
            shape_cyc = {0: "s", 1: "v", 2: "*", 3: "^"}
            if xy_or_yx == "xy":
                emtf_rho = aux_data.rxy
                emtf_phi = aux_data.pxy
            else:
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
        # except:
        #     # for i_dec in aux_data.decimation_levels
        #     axs[0].loglog(
        #         aux_data.periods,
        #         aux_data.rxy,
        #         marker="s",
        #         color="k",
        #         linestyle="None",
        #     )
        #     axs[1].semilogx(
        #         aux_data.periods,
        #         aux_data.pxy,
        #         marker="s",
        #         color="k",
        #         linestyle="None",
        #     )
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

        figure_basename = f"synthetic_{tf.tf_header.local_station_id}_{xy_or_yx}.png"
        out_file = FIGURES_PATH.joinpath(figure_basename)
        plt.savefig(out_file)
        if show:
            plt.show()
