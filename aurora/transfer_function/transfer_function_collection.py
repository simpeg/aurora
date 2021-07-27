"""
Placeholder class.  Will probably evolve structure in time.
This is a container to hold:
1. TransferFunctionHeader
2. Dictionary of TransferFunction Objects

Note that a single transfer function object is associated with a station,
which we call the "local_station".  In a database of TFs we could add a column
for local_station and one for reference station.
"""
import numpy as np
from aurora.transfer_function.emtf_z_file_helpers import make_orientation_block_of_z_file

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
        first cut - could probably move this into EMTFUtils() class

        Issues to review:
        This seems to insist that channels be ordered:
        Hx, Hy, Hz, Ex, Ey
        Returns
        -------

        """
        f = open(z_file_path, "w")
        f.writelines(" **** IMPEDANCE IN MEASUREMENT COORDINATES ****\n")
        f.writelines(" ********** WITH FULL ERROR COVARINCE**********\n")

        # <processing scheme>
        try:
            processing_scheme = EMTF_REGRESSION_ENGINE_LABELS[
                self.header.processing_scheme]
        except:
            processing_scheme = self.header.processing_scheme
        f.writelines(f"{processing_scheme}\n")
        f.writelines(f"station    :{self.header.local_station_id}\n")
        # </processing scheme>

        # <location>
        #could also use self.header.local_station object here
        if run_obj is None:
            latitude = 1007.996
            longitude = 0.000
            declination = 0.00
        else:
            latitude = run_obj.station_group.metadata.location.latitude
            longitude = run_obj.station_group.metadata.location.longitude
            declination = run_obj.station_group.metadata.location\
                .declination.value
            if declination is None:
                declination = 0.0

        location_str = f"coordinate  {latitude}  {longitude}  {declination}\n"
        f.writelines(location_str)
        # </location>

        # <num channels / num frequencies>
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


        #<DATA READ>
        #Given that the channel ordering is fixed (hxhyhzexey) and that hxhy
        # are always the input channels, the TF is ordered hzexey or exey
        # depending on 2 or 3 channels.
        for i_dec in self.tf_dict.keys():
            tf = self.tf_dict[i_dec]
            tf_xr = tf.to_xarray()
            cov_ss_xr = tf.cov_ss_to_xarray()
            cov_nn_xr = tf.cov_nn_to_xarray()
            periods = tf.frequency_bands.band_centers(frequency_or_period="period")
            periods = np.flip(periods) #EMTF works in increasing period
            #for i_period, period in periods:
            for i_band, band in enumerate(tf.frequency_bands.bands(
                    direction="increasing_period")):

                print(i_band, band)
                line1 = f"period :      {band.center_period:.5f}    "
                line1 += f"decimation level   {i_dec+1}:    "
                #<Make a method of processing config?>
                freqs = num_frequencies_str = np.fft.fftfreq(
                    tf.processing_config.num_samples_window, 1./tf.processing_config.sample_rate)
                fc_indices = band.fourier_coefficient_indices(freqs)
                #<Make a method of processing config?>
                fc_indices_str = f"{fc_indices[0]} to   {fc_indices[-1]}"
                line1 += f"freq. band from   {fc_indices_str}\n"
                f.writelines(line1)
                freq_index = tf.frequency_index(band.center_frequency)
                num_segments = tf.num_segments.data[0, freq_index]
                line2 = f"number of data point    {num_segments} "
                line2 += f"sampling freq.   {tf.processing_config.sample_rate} Hz\n"
                f.writelines(line2)


                f.writelines(f"  Transfer Functions\n")
                #write the tf:
                #rows are output channels (hz, ex, ey), cols are input
                # channels (hx, hy)
                period_index = tf.period_index(band.center_period)
                line = ""
                for out_ch in tf.tf_header.output_channels:
                    for inp_ch in tf.tf_header.input_channels:
                        print(out_ch, inp_ch)
                        chchtf = tf_xr.loc[out_ch, inp_ch, :]

                        line += f"{np.real(chchtf.data[period_index]):.5f}  "
                        line += f"{np.imag(chchtf.data[period_index]):.5f}  "
                    line += "\n"
                f.writelines(line)


                f.writelines(f"    Inverse Coherent Signal Power Matrix\n")
                line = ""
                for i, inp_ch1 in enumerate(tf.tf_header.input_channels):
                    for inp_ch2 in tf.tf_header.input_channels[:i+1]:
                        cond1 = cov_ss_xr.input_channel_1 == inp_ch1
                        cond2 = cov_ss_xr.input_channel_2 == inp_ch2
                        chchss = cov_ss_xr.where(cond1 & cond2, drop=True)
                        chchss = chchss.data.squeeze()
                        line += f"{chchss.data[period_index]:.5f}  "
                    line += "\n"
                f.writelines(line)

                f.writelines(f"  Residual Covariance\n")
                line = ""
                for i, out_ch1 in enumerate(tf.tf_header.output_channels):
                    for out_ch2 in tf.tf_header.output_channels[:i+1]:
                        cond1 = cov_nn_xr.output_channel_1 == out_ch1
                        cond2 = cov_nn_xr.output_channel_2 == out_ch2
                        chchnn = cov_nn_xr.where(cond1 & cond2, drop=True)
                        chchnn = chchnn.data.squeeze()
                        line += f"{chchnn.data[period_index]:.5f}  "
                    line += "\n"
                f.writelines(line)

            pass
        #    for
        #     period :      4.65455    decimation level   1    freq. band from   25 to   30
        # number of data point   2489 sampling freq.   1.000 Hz
        #  Transfer Functions
        #   0.2498E+00  0.1966E-03  0.3859E-04  0.2519E+00
        #  -0.1458E-01 -0.2989E-01 -0.7283E+01 -0.7313E+01
        #   0.7311E+01  0.7338E+01 -0.4087E-01 -0.1031E-01
        #  Inverse Coherent Signal Power Matrix
        #   0.3809E-07 -0.6261E-18
        #  -0.3095E-09  0.4505E-09  0.3764E-07  0.7792E-17
        #  Residual Covariance
        #   0.3639E+02  0.0000E+00
        #  -0.2604E+03  0.2280E+03  0.3090E+05  0.0000E+00
        #   0.2483E+03  0.2688E+03  0.2660E+03 -0.6791E+03  0.3161E+05  0.0000E+00



        f.close()