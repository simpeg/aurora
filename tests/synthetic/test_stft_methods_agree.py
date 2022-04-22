"""
See aurora issue #3.  This test confirms that the internal aurora stft
method returns the same array as scipy.signal.spectrogram
"""
import numpy as np

from aurora.general_helper_functions import TEST_PATH
from aurora.pipelines.process_mth5 import initialize_pipeline
from aurora.pipelines.time_series_helpers_new import get_data_from_mth5_new
from aurora.pipelines.time_series_helpers_new import prototype_decimate
from aurora.pipelines.time_series_helpers_new import run_ts_to_stft
from aurora.pipelines.time_series_helpers_new import run_ts_to_stft_scipy
from aurora.test_utils.synthetic.make_mth5_from_asc import create_test1_h5
from aurora.test_utils.synthetic.make_processing_configs_new import create_test_run_config
from aurora.tf_kernel.helpers import extract_run_summaries_from_mth5s
from mth5.mth5 import MTH5
from mth5.helpers import close_open_files


def test_stft_methods_agree():
    mth5_path = create_test1_h5()
    mth5_paths = [mth5_path, ]
    super_summary = extract_run_summaries_from_mth5s(mth5_paths)
    dataset_df = super_summary[super_summary.station_id=="test1"]
    dataset_df["remote"] = False

    processing_config = create_test_run_config("test1", dataset_df)

    mth5_obj = MTH5(file_version="0.1.0")
    mth5_obj.open_mth5(mth5_path, mode="a")

    run_id = "001"
    for dec_level_id, dec_config in enumerate(processing_config.decimations):

        if dec_level_id == 0:
            run_dict = get_data_from_mth5_new(mth5_obj, "test1", run_id, 1.0)
        else:
            run_dict = prototype_decimate(dec_config.decimation, run_dict)

        local_run_xrts = run_dict["mvts"]
        dec_config.extra_pre_fft_detrend_type = ""
        local_stft_obj = run_ts_to_stft(dec_config, local_run_xrts)
        local_stft_obj2 = run_ts_to_stft_scipy(dec_config, local_run_xrts)
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
