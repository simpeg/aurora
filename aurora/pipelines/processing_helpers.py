


def calibrate_stft_obj(stft_obj, run_obj, units="MT"):
    """

    Parameters
    ----------
    stft_obj
    run_obj
    units

    Returns
    -------

    """
    for channel_id in stft_obj.keys():
        mth5_channel = run_obj.get_channel(channel_id)
        channel_filter = mth5_channel.channel_response_filter
        calibration_response = channel_filter.complex_response(
            stft_obj.frequency.data)

        if units == "SI":
            print("Warning: SI Units are not robustly supported issue #36")
            #This is not robust, and is really only here for the parkfield test
            #We should add units support as a general fix and handle the
            # parkfield case by converting to "MT" units in calibration filters
            if channel_id[0].lower() == 'h':
                calibration_response /= 1e-9  # SI Units
        stft_obj[channel_id].data /= calibration_response
    return stft_obj