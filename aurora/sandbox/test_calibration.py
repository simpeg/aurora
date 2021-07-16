import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
plt.ion()


def load_bf4_fap_for_parkfield_test_using_mt_metadata(frequencies):
    """

    Returns
    -------

    """
    from aurora.time_series.filters.filter_helpers import \
        make_frequency_response_table_filter
    bf4_obj = make_frequency_response_table_filter(case="bf4")
    bf4_resp = bf4_obj.complex_response(frequencies)
    bf4_resp *= 421721.0  # CPV compensation
    return bf4_resp


# def load_bf4_fap_for_parkfield_test_using_qf_methods(frequencies):
#     """
#     This function to be removed/deprecated as soon as we have a FAP table
#     consistent with BF4 response archived at IRIS
#     Parameters
#     ----------
#     frequencies
#
#     Returns
#     -------
#
#     """
#     from .qf.instrument import DeployedInstrument
#     from .qf.instrument import Instrument
#     bf4_coil = Instrument(make="emi", model="bf4", serial_number=9819,
#                           channel=0, epoch=0)
#     deployed_bf4 = DeployedInstrument(sensor=bf4_coil)
#     deployed_bf4.get_response_function()
#     bf4_resp = deployed_bf4.response_function(frequencies)
#     bf4_resp *= 421721.0  # CPV compensation
#     return bf4_resp



def parkfield_sanity_check(fft_obj, run_obj, show_response_curves=False,
                           show_spectra=True, figures_path=Path(""),
                           include_decimation=True):
    """
    loop over channels in fft obj and make calibrated spectral plots
    
    Parameters
    ----------
    fft_obj

    Returns
    -------

    """

    frequencies = fft_obj.frequency.data[1:] #drop DC, add flag for dropping DC
    figures_path.mkdir(parents=True, exist_ok=True)

    channel_keys = list(fft_obj.data_vars.keys())
    print(f"channel_keys: {channel_keys}")
    for key in channel_keys:
        print(f"CHANNEL {key}")
        if key[0].lower() == "h":
            bf4 = True
            response_units = "V/nT"
            spectral_units = "nT/sqrt(Hz)"
        else:
            bf4 = False
            response_units = "V/mV/km"
            spectral_units = "mv/km/sqrt(Hz)"

        channel = run_obj.get_channel(key)

        # <PZRSP>
        #owing to issue #32 in mth5 I am going to modify this to use a different
        #way to calibrate for now
        pz_calibration_response = channel.channel_response_filter.complex_response(
            frequencies, include_decimation=include_decimation)

        # </PZRSP>

        # <FAP RSP>
        if bf4:
            bf4_resp = load_bf4_fap_for_parkfield_test_using_mt_metadata(frequencies)
            #bf4_resp = load_bf4_fap_for_parkfield_test_using_qf_methods(
            # frequencies)
            abs_bf4_resp = np.abs(bf4_resp)

        # </FAP RSP>

        # <CF RESPONSES>
        plt.figure(1)
        plt.clf()
        plt.loglog(frequencies, np.abs(pz_calibration_response),
                   label='pole-zero')
        if bf4:
            plt.loglog(frequencies, np.abs(bf4_resp), label='fap EMI')
        plt.legend()
        plt.title(f"Calibration Response Functions {key}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel(f"Response {response_units}")
        png_name = f"{key}_response_function.png"
        plt.savefig(figures_path.joinpath(png_name))
        if show_response_curves:
            plt.show()
        # </CF RESPONSES>


        n_smooth = 131
        show_raw = 0
        raw_spectral_density = fft_obj[key].data[:, 1:]
        # raw_spectral_density = np.abs(fft_obj[key].data[:, 1:])
        raw_spectral_density = raw_spectral_density.squeeze()  # only for full window - need to recognize this or workaround
        calibrated_data_pz = raw_spectral_density / pz_calibration_response
        # calibrated_data_pz = raw_spectral_density / abs_pz_calibration_response
        if bf4:
            calibrated_data_fap = raw_spectral_density / abs_bf4_resp

        plt.figure(2)
        plt.clf()
        if n_smooth:
            import scipy.signal as ssig
            smooth_calibrated_data_pz = ssig.medfilt(np.abs(calibrated_data_pz),
                                                     n_smooth)
            if bf4:
                smooth_calibrated_data_fap = ssig.medfilt(
                np.abs(calibrated_data_fap), n_smooth)
        if show_raw:
            plt.loglog(frequencies, calibrated_data_pz, color='b',
                       label='pole-zero')
            if bf4:
                plt.loglog(frequencies, calibrated_data_fap, color='r',
                       label='fap EMI')
        if n_smooth:
            plt.loglog(frequencies, smooth_calibrated_data_pz, color='b',
                       label='smooth pole-zero')
            if bf4:
                plt.loglog(frequencies, smooth_calibrated_data_fap, color='r',
                       label='fap EMI')

        plt.legend()
        plt.grid(True, which="both", ls="-")
        plt.title(f"Calibrated Spectra {key}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel(f"{spectral_units}")
        png_name = f"{key}_calibrated_spectra.png"
        plt.savefig(figures_path.joinpath(png_name))
        if show_spectra:
            plt.show()
