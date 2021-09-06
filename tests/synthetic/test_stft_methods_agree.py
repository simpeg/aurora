"""
See aurora issue #3.  This test confirms that the internal aurora stft
method returns the same array as scipy.signal.spectrogram
"""
import numpy as np

from aurora.general_helper_functions import TEST_PATH
from aurora.pipelines.process_mth5 import get_data_from_decimation_level_from_mth5
from aurora.pipelines.process_mth5 import initialize_pipeline
from aurora.pipelines.process_mth5 import prototype_decimate
from aurora.pipelines.time_series_helpers import run_ts_to_stft
from aurora.pipelines.time_series_helpers import run_ts_to_stft_scipy


def test_stft_methods_agree():
    run_config = TEST_PATH.joinpath(
        "synthetic", "config", "test1_run_config_standard.json"
    )
    run_id = "001"
    mth5_path = TEST_PATH.joinpath("synthetic", "data", "test1.h5")
    run_config, mth5_obj = initialize_pipeline(run_config, mth5_path=mth5_path)

    for dec_level_id in run_config.decimation_level_ids:
        processing_config = run_config.decimation_level_configs[dec_level_id]
        processing_config.local_station_id = run_config.local_station_id

        if dec_level_id == 0:
            local, remote = get_data_from_decimation_level_from_mth5(
                processing_config, mth5_obj, run_id
            )
        else:
            local = prototype_decimate(processing_config, local)
            # if processing_config.reference_station_id:
            #     remote = prototype_decimate(processing_config, remote)

        # </GET DATA>
        # local_run_obj = local["run"]
        local_run_xrts = local["mvts"]
        processing_config.extra_pre_fft_detrend_type = ""
        local_stft_obj = run_ts_to_stft(processing_config, local_run_xrts)
        local_stft_obj2 = run_ts_to_stft_scipy(processing_config, local_run_xrts)
        stft_difference = local_stft_obj - local_stft_obj2
        stft_difference = stft_difference.to_array()

        # drop dc term
        stft_difference = stft_difference.where(
            stft_difference.frequency > 0, drop=True
        )

        assert np.isclose(stft_difference, 0).all()

        print("stft aurora method agrees with scipy.signal.spectrogram")
    return


def main():
    test_stft_methods_agree()


if __name__ == "__main__":
    main()
